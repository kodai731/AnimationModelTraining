from __future__ import annotations

import argparse
import gc
import math
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from anim_ml.data.rig_dataset import RigPropagationDataset
from anim_ml.models.rig_propagation.model import RigPropagationConfig, RigPropagationModel
from anim_ml.paths import resolve_data_path
from anim_ml.utils.checkpoint import (
    CheckpointState,
    load_training_checkpoint,
    restore_checkpoint_state,
    save_training_checkpoint,
)
from anim_ml.utils.device import detect_training_device, supports_pin_memory
from anim_ml.utils.batch_budget import resolve_batch_size
from anim_ml.utils.memory_budget import create_memory_budget
from anim_ml.utils.optimizer import DmlAdamW
from anim_ml.utils.preparation_log import PreparationLog
from anim_ml.utils.timing_log import BatchTimingLog, TimingLog


@dataclass
class LossWeights:
    rotation: float = 1.0
    confidence: float = 0.2


@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    early_stopping_patience: int = 0
    loss_weights: LossWeights = field(default_factory=LossWeights)


@dataclass
class DataConfig:
    train_files: list[str] = field(default_factory=lambda: [])
    val_split: str = "val"
    num_workers: int = 4


@dataclass
class OutputConfig:
    checkpoint_dir: str = "runs/rig_propagation"
    save_every_epochs: int = 10
    log_every_steps: int = 100


@dataclass
class TrainConfig:
    model: RigPropagationConfig = field(default_factory=RigPropagationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: str | Path) -> TrainConfig:
    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    model_cfg = RigPropagationConfig(**raw.get("model", {}))

    training_raw = raw.get("training", {})
    loss_raw = training_raw.pop("loss_weights", {})
    loss_weights = LossWeights(**loss_raw)
    training_cfg = TrainingConfig(**training_raw, loss_weights=loss_weights)

    data_cfg = DataConfig(**raw.get("data", {}))
    output_cfg = OutputConfig(**raw.get("output", {}))

    return TrainConfig(
        model=model_cfg,
        training=training_cfg,
        data=data_cfg,
        output=output_cfg,
    )


def compute_rotation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    dot = (pred * target).sum(dim=-1)
    loss = 1.0 - dot.abs()
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1.0)
    return loss.mean()


def compute_loss(
    rotation_deltas: torch.Tensor,
    confidence: torch.Tensor,
    target_deltas: torch.Tensor,
    confidence_targets: torch.Tensor,
    weights: LossWeights,
    joint_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    rotation_loss = compute_rotation_loss(rotation_deltas, target_deltas, joint_mask)

    eps = 1e-7
    p = confidence.squeeze(-1).clamp(eps, 1 - eps)
    confidence_loss_raw = -(
        confidence_targets * p.log() + (1 - confidence_targets) * (1 - p).log()
    )
    if joint_mask is not None:
        confidence_loss = (confidence_loss_raw * joint_mask).sum() / joint_mask.sum().clamp(min=1.0)
    else:
        confidence_loss = confidence_loss_raw.mean()

    total = weights.rotation * rotation_loss + weights.confidence * confidence_loss

    metrics = {
        "loss/total": total.item(),
        "loss/rotation": rotation_loss.item(),
        "loss/confidence": confidence_loss.item(),
    }
    return total, metrics


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> tuple[DmlAdamW, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = DmlAdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_one_epoch(
    model: RigPropagationModel,
    train_dataset: RigPropagationDataset,
    optimizer: DmlAdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: TrainConfig,
    device: torch.device,
    epoch: int,
    pin_memory: bool = False,
    batch_timing: BatchTimingLog | None = None,
    global_step: int = 0,
) -> tuple[dict[str, float], int]:
    model.train()
    total_loss = 0.0
    num_batches = 0

    if batch_timing:
        batch_timing.begin_epoch(epoch)

    num_chunks = 1 if train_dataset.is_fully_loaded else train_dataset.num_chunks
    epoch_step = 0

    train_dataset.begin_epoch(epoch)
    chunk_order = list(range(num_chunks))
    random.Random(epoch).shuffle(chunk_order)

    for chunk_idx in chunk_order:
        if num_chunks > 1:
            train_dataset.reload_chunk(chunk_idx)

        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=True,
        )

        data_wait = 0.0
        for batch in loader:
            if batch_timing:
                data_wait = time.perf_counter() - batch_timing.data_start

            joint_features = batch["joint_features"].to(device, non_blocking=True)
            topology_features = batch["topology_features"].to(device, non_blocking=True)
            bone_name_tokens = batch["bone_name_tokens"].to(device, non_blocking=True)
            joint_mask = batch["joint_mask"].to(device, non_blocking=True)
            target_deltas = batch["target_deltas"].to(device, non_blocking=True)
            confidence_targets = batch["confidence_targets"].to(device, non_blocking=True)
            source_indices = batch["source_indices"][0].to(device, non_blocking=True)
            target_indices = batch["target_indices"][0].to(device, non_blocking=True)
            edge_direction = batch["edge_direction"][0].to(device, non_blocking=True)
            edge_mask = batch["edge_mask"][0].to(device, non_blocking=True).float()

            compute_start = time.perf_counter()

            rotation_deltas, confidence = model(
                joint_features, topology_features, bone_name_tokens, joint_mask,
                source_indices, target_indices, edge_direction, edge_mask,
            )
            loss, metrics = compute_loss(
                rotation_deltas, confidence, target_deltas,
                confidence_targets, config.training.loss_weights, joint_mask,
            )

            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()  # type: ignore[no-untyped-call]
            scheduler.step()

            if batch_timing:
                compute_elapsed = time.perf_counter() - compute_start
                batch_timing.record_step(epoch_step, data_wait, compute_elapsed)
                batch_timing.mark_data_start()

            total_loss += metrics["loss/total"]
            num_batches += 1
            epoch_step += 1

            if epoch_step % config.output.log_every_steps == 0:
                avg = total_loss / num_batches
                chunk_info = f" [chunk {chunk_idx + 1}/{num_chunks}]" if num_chunks > 1 else ""
                print(f"  epoch {epoch} step {epoch_step}{chunk_info}: loss={avg:.4f}", flush=True)

    if batch_timing:
        batch_timing.end_epoch()

    return {"loss/train": total_loss / max(num_batches, 1)}, global_step + epoch_step


@torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
def validate(
    model: RigPropagationModel,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        joint_features = batch["joint_features"].to(device, non_blocking=True)
        topology_features = batch["topology_features"].to(device, non_blocking=True)
        bone_name_tokens = batch["bone_name_tokens"].to(device, non_blocking=True)
        joint_mask = batch["joint_mask"].to(device, non_blocking=True)
        target_deltas = batch["target_deltas"].to(device, non_blocking=True)
        confidence_targets = batch["confidence_targets"].to(device, non_blocking=True)
        source_indices = batch["source_indices"][0].to(device, non_blocking=True)
        target_indices = batch["target_indices"][0].to(device, non_blocking=True)
        edge_direction = batch["edge_direction"][0].to(device, non_blocking=True)
        edge_mask = batch["edge_mask"][0].to(device, non_blocking=True).float()

        rotation_deltas, confidence = model(
            joint_features, topology_features, bone_name_tokens, joint_mask,
            source_indices, target_indices, edge_direction, edge_mask,
        )
        _, metrics = compute_loss(
            rotation_deltas, confidence, target_deltas,
            confidence_targets, config.training.loss_weights, joint_mask,
        )

        total_loss += metrics["loss/total"]
        num_batches += 1

    return {"loss/val": total_loss / max(num_batches, 1)}


def _build_model_config_dict(model: RigPropagationModel) -> dict[str, Any]:
    return {
        "max_joints": model.config.max_joints,
        "max_edges": model.config.max_edges,
        "node_feature_dim": model.config.node_feature_dim,
        "edge_feature_dim": model.config.edge_feature_dim,
        "hidden_dim": model.config.hidden_dim,
        "ffn_dim": model.config.ffn_dim,
        "num_message_passing_layers": model.config.num_message_passing_layers,
        "input_feature_dim": model.config.input_feature_dim,
        "dropout": model.config.dropout,
        "vocab_size": model.config.vocab_size,
        "token_length": model.config.token_length,
        "char_embed_dim": model.config.char_embed_dim,
        "conv_channels": model.config.conv_channels,
        "bone_context_dim": model.config.bone_context_dim,
        "topology_dim": model.config.topology_dim,
    }


def save_checkpoint(
    model: RigPropagationModel,
    optimizer: DmlAdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    state: CheckpointState,
    metrics: dict[str, float],
    path: str | Path,
) -> None:
    save_training_checkpoint(
        model, optimizer, scheduler, state, metrics,
        _build_model_config_dict(model), path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return load_training_checkpoint(path)


def train(
    config: TrainConfig,
    prep_log: PreparationLog,
    resume_path: str | Path | None = None,
    device_override: str | None = None,
) -> None:
    prep_log.log("train_func_entered")

    device = detect_training_device(device_override)
    print(f"Using device: {device}", flush=True)
    prep_log.log("device_detected", device=str(device))

    prep_log.log("train_start", device=str(device),
                 num_workers=config.data.num_workers,
                 batch_size=config.training.batch_size)

    train_paths = [resolve_data_path(p) for p in config.data.train_files]
    budget = create_memory_budget()

    val_dataset = RigPropagationDataset(
        train_paths, split=config.data.val_split, prep_log=prep_log,
        memory_budget=budget, budget_name="val",
    )
    train_dataset = RigPropagationDataset(
        train_paths, split="train", prep_log=prep_log,
        memory_budget=budget, budget_name="train",
    )

    val_dataset.evict_cache()
    train_dataset.reload_cache()
    gc.collect()

    train_status = "fully loaded" if train_dataset.is_fully_loaded else "chunked"
    print(f"Train: {train_dataset.total_count} samples ({train_status})", flush=True)
    prep_log.log("datasets_ready",
                 train_samples=len(train_dataset), val_samples=len(val_dataset))

    model = RigPropagationModel(config.model).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}", flush=True)
    prep_log.log("model_created", params=param_count)

    batch_result = resolve_batch_size(device, config.training.batch_size, param_count)
    config.training.batch_size = batch_result.batch_size
    print(f"Batch size: {batch_result.batch_size} ({batch_result.reason})", flush=True)

    pin_memory = supports_pin_memory(device)

    prep_log.log("dataloader_create_start", num_workers=0)
    val_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    prep_log.log("dataloader_create_done")

    if train_dataset.is_fully_loaded:
        steps_per_epoch = max(train_dataset.total_count // config.training.batch_size, 1)
    else:
        chunk_steps = len(train_dataset) // config.training.batch_size
        steps_per_epoch = max(chunk_steps * train_dataset.num_chunks, 1)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config.training, steps_per_epoch,
    )

    checkpoint_dir = resolve_data_path(config.output.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timing_log = TimingLog("rig_propagation")
    batch_timing = BatchTimingLog("rig_propagation")
    print(f"Timing log: {timing_log.path}", flush=True)
    print(f"Batch timing log: {batch_timing.step_path}", flush=True)
    prep_log.log("preparation_complete", steps_per_epoch=steps_per_epoch)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = config.training.early_stopping_patience
    all_metrics: dict[str, float] = {}
    start_epoch = 1
    global_step = 0

    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        ckpt_state = restore_checkpoint_state(ckpt)
        global_step = ckpt_state.global_step
        best_val_loss = ckpt_state.best_val_loss
        epochs_without_improvement = ckpt_state.epochs_without_improvement

        start_epoch = ckpt_state.epoch if ckpt_state.is_mid_epoch else ckpt_state.epoch + 1

        print(
            f"Resumed from epoch {ckpt_state.epoch}"
            f" (best_val_loss={best_val_loss:.4f}"
            f", mid_epoch={ckpt_state.is_mid_epoch})",
            flush=True,
        )

    epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, config.training.epochs + 1):
            print(f"Epoch {epoch}/{config.training.epochs}", flush=True)

            with TimingLog.measure() as train_time:
                train_metrics, global_step = train_one_epoch(
                    model, train_dataset, optimizer, scheduler,
                    config, device, epoch, pin_memory, batch_timing,
                    global_step,
                )

            train_dataset.evict_cache()
            val_dataset.reload_cache()

            with TimingLog.measure() as val_time:
                val_metrics = validate(model, val_loader, config, device)

            val_dataset.evict_cache()

            tier_change = budget.refresh()
            if tier_change:
                old_mb = tier_change[0] // (1024 * 1024)
                new_mb = tier_change[1] // (1024 * 1024)
                print(f"  Memory tier changed: {old_mb} MB -> {new_mb} MB", flush=True)

            train_dataset.reload_cache()

            all_metrics = {**train_metrics, **val_metrics}
            print(f"  train_loss={train_metrics['loss/train']:.4f}"
                  f"  val_loss={val_metrics['loss/val']:.4f}", flush=True)

            checkpoint_time_sec = 0.0

            if val_metrics["loss/val"] < best_val_loss:
                best_val_loss = val_metrics["loss/val"]
                epochs_without_improvement = 0
                with TimingLog.measure() as ckpt_time:
                    save_checkpoint(
                        model, optimizer, scheduler,
                        CheckpointState(epoch, global_step, False,
                                        best_val_loss, epochs_without_improvement),
                        all_metrics, checkpoint_dir / "best.pt",
                    )
                checkpoint_time_sec += ckpt_time["elapsed"]
            else:
                epochs_without_improvement += 1

            if epoch % config.output.save_every_epochs == 0:
                with TimingLog.measure() as ckpt_time:
                    save_checkpoint(
                        model, optimizer, scheduler,
                        CheckpointState(epoch, global_step, False,
                                        best_val_loss, epochs_without_improvement),
                        all_metrics, checkpoint_dir / f"epoch_{epoch}.pt",
                    )
                checkpoint_time_sec += ckpt_time["elapsed"]

            timing_log.write_epoch(epoch, {
                "train_sec": round(train_time["elapsed"], 3),
                "val_sec": round(val_time["elapsed"], 3),
                "checkpoint_sec": round(checkpoint_time_sec, 3),
                "total_sec": round(
                    train_time["elapsed"] + val_time["elapsed"] + checkpoint_time_sec, 3,
                ),
                "num_steps": steps_per_epoch,
                "train_loss": round(train_metrics["loss/train"], 6),
                "val_loss": round(val_metrics["loss/val"], 6),
            })

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if patience > 0 and epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}"
                    f" (no improvement for {patience} epochs)",
                    flush=True,
                )
                break

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch}.", flush=True)
        if epoch > 0:
            save_checkpoint(
                model, optimizer, scheduler,
                CheckpointState(epoch, global_step, True,
                                best_val_loss, epochs_without_improvement),
                all_metrics, checkpoint_dir / "last.pt",
            )
            print(f"Saved mid-epoch last.pt (epoch {epoch}).", flush=True)

    else:
        if epoch > 0:
            save_checkpoint(
                model, optimizer, scheduler,
                CheckpointState(epoch, global_step, False,
                                best_val_loss, epochs_without_improvement),
                all_metrics, checkpoint_dir / "last.pt",
            )
            print(f"Saved last.pt (epoch {epoch}).", flush=True)

    if best_val_loss < float("inf"):
        print(f"Best val loss: {best_val_loss:.4f} (best.pt)", flush=True)


def main() -> None:
    prep_log = PreparationLog("rig_propagation")
    prep_log.log("main_started")

    parser = argparse.ArgumentParser(description="Train Rig Propagation model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    prep_log.log("args_parsed", config=args.config, resume=args.resume, device=args.device)

    config = load_config(args.config)
    prep_log.log("config_loaded", num_train_files=len(config.data.train_files),
                 num_workers=config.data.num_workers, batch_size=config.training.batch_size)

    train(config, prep_log, resume_path=args.resume, device_override=args.device)


if __name__ == "__main__":
    main()
