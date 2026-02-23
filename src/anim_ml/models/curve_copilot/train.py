from __future__ import annotations

import argparse
import gc
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from anim_ml.data.curve_dataset import CurveCopilotDataset
from anim_ml.models.curve_copilot.model import CurveCopilotConfig, CurveCopilotModel
from anim_ml.paths import resolve_data_path
from anim_ml.utils.checkpoint import (
    CheckpointState,
    load_training_checkpoint,
    restore_checkpoint_state,
    save_training_checkpoint,
)
from anim_ml.utils.device import detect_training_device, supports_pin_memory
from anim_ml.utils.memory_budget import create_memory_budget
from anim_ml.utils.optimizer import DmlAdamW
from anim_ml.utils.preparation_log import PreparationLog
from anim_ml.utils.timing_log import BatchTimingLog, TimingLog


@dataclass
class LossWeights:
    value: float = 1.0
    tangent: float = 0.5
    interpolation: float = 0.3
    confidence: float = 0.2
    smoothness: float = 0.1
    frequency: float = 0.05


@dataclass
class TrainingConfig:
    batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    early_stopping_patience: int = 0
    loss_weights: LossWeights = field(default_factory=LossWeights)
    pae_pretrained: str = ""
    pae_learning_rate: float = 1e-5
    step_decay: list[float] = field(default_factory=lambda: [1.0])


@dataclass
class DataConfig:
    train_files: list[str] = field(default_factory=lambda: [])
    val_split: str = "val"
    num_workers: int = 4


@dataclass
class OutputConfig:
    checkpoint_dir: str = "runs/curve_copilot"
    save_every_epochs: int = 10
    log_every_steps: int = 100


@dataclass
class TrainConfig:
    model: CurveCopilotConfig = field(default_factory=CurveCopilotConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: str | Path) -> TrainConfig:
    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    model_cfg = CurveCopilotConfig(**raw.get("model", {}))

    training_raw = raw.get("training", {})
    loss_raw = training_raw.pop("loss_weights", {})
    loss_weights = LossWeights(**loss_raw)
    step_decay = training_raw.pop("step_decay", [1.0])
    training_cfg = TrainingConfig(**training_raw, loss_weights=loss_weights, step_decay=step_decay)

    data_cfg = DataConfig(**raw.get("data", {}))
    output_cfg = OutputConfig(**raw.get("output", {}))

    return TrainConfig(
        model=model_cfg,
        training=training_cfg,
        data=data_cfg,
        output=output_cfg,
    )


def _huber_loss(input: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    error = input - target
    abs_error = error.abs()
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    return (0.5 * quadratic.pow(2) + delta * linear).mean()


def compute_frequency_loss(
    prediction_value: torch.Tensor,
    target_value: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    context_values = context[:, :, 1]

    pred_sequence = torch.cat([context_values, prediction_value.unsqueeze(1)], dim=1)
    target_sequence = torch.cat([context_values, target_value.unsqueeze(1)], dim=1)

    pred_spectrum = cast(
        "torch.Tensor",
        torch.fft.rfft(pred_sequence.cpu(), dim=1).abs(),  # pyright: ignore[reportUnknownMemberType]
    )
    target_spectrum = cast(
        "torch.Tensor",
        torch.fft.rfft(target_sequence.cpu(), dim=1).abs(),  # pyright: ignore[reportUnknownMemberType]
    )

    num_freqs: int = pred_spectrum.shape[1]
    freq_weights = torch.exp(
        -0.5 * torch.arange(num_freqs, dtype=torch.float32),
    )

    loss = ((pred_spectrum - target_spectrum).pow(2) * freq_weights.unsqueeze(0)).mean()
    return loss.to(context.device)


def compute_loss(
    prediction: torch.Tensor,
    confidence: torch.Tensor,
    target: torch.Tensor,
    weights: LossWeights,
    confidence_targets: torch.Tensor,
    context: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = nn.functional.mse_loss

    value_loss = _huber_loss(prediction[:, 0], target[:, 1], delta=1.0)
    tangent_loss = mse(prediction[:, 1:5], target[:, 2:6])

    tangent_norms = torch.norm(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        target[:, 2:6], dim=1,
    )
    tangent_magnitude = cast("torch.Tensor", tangent_norms)
    interp_target = (tangent_magnitude > 0.01).float()
    logits = prediction[:, 5]
    interp_loss = (
        logits.clamp(min=0) - logits * interp_target + torch.log1p(torch.exp(-logits.abs()))
    ).mean()

    confidence_loss = mse(confidence, confidence_targets)

    smoothness_loss = torch.tensor(0.0, device=prediction.device)
    if context is not None:
        context_velocity = context[:, -1, 1] - context[:, -2, 1]
        predicted_velocity = prediction[:, 0] - context[:, -1, 1]
        implied_acceleration = predicted_velocity - context_velocity
        smoothness_loss = implied_acceleration.pow(2).mean()

    frequency_loss = torch.tensor(0.0, device=prediction.device)
    if context is not None and context.shape[1] >= 2:
        frequency_loss = compute_frequency_loss(
            prediction[:, 0], target[:, 1], context,
        )

    total = (
        weights.value * value_loss
        + weights.tangent * tangent_loss
        + weights.interpolation * interp_loss
        + weights.confidence * confidence_loss
        + weights.smoothness * smoothness_loss
        + weights.frequency * frequency_loss
    )

    metrics = {
        "loss/total": total.item(),
        "loss/value": value_loss.item(),
        "loss/tangent": tangent_loss.item(),
        "loss/interpolation": interp_loss.item(),
        "loss/confidence": confidence_loss.item(),
        "loss/smoothness": smoothness_loss.item(),
        "loss/frequency": frequency_loss.item(),
    }
    return total, metrics


def compute_multistep_loss(
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    targets: torch.Tensor,
    valid_steps: torch.Tensor,
    weights: LossWeights,
    context: torch.Tensor,
    step_decay: list[float],
) -> tuple[torch.Tensor, dict[str, float]]:
    num_steps = predictions.shape[1]
    total_loss = torch.tensor(0.0, device=predictions.device)
    merged_metrics: dict[str, float] = {}

    for step in range(num_steps):
        step_mask = valid_steps > step
        if not step_mask.any():
            break

        step_pred = predictions[step_mask, step, :]
        step_conf = confidences[step_mask, step, :]
        step_target = targets[step_mask, step, :]
        step_context = context[step_mask]

        with torch.no_grad():
            last_context_value = step_context[:, -1, 1]
            target_value = step_target[:, 1]
            value_distance = torch.abs(target_value - last_context_value)
            confidence_targets = torch.exp(-value_distance * 5.0) * (0.9 ** step)
            confidence_targets = confidence_targets.unsqueeze(-1)

        use_context = step_context if step == 0 else None
        step_loss, step_metrics = compute_loss(
            step_pred, step_conf, step_target, weights,
            confidence_targets, use_context,
        )

        decay = step_decay[step] if step < len(step_decay) else step_decay[-1]
        total_loss = total_loss + decay * step_loss

        for key, val in step_metrics.items():
            merged_metrics[f"step{step}/{key}"] = val

        if step == 0:
            for key, val in step_metrics.items():
                merged_metrics[key] = val

    merged_metrics["loss/total"] = total_loss.item()
    return total_loss, merged_metrics


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> tuple[DmlAdamW, torch.optim.lr_scheduler.LambdaLR]:
    pae_params = [p for n, p in model.named_parameters() if "pae_encoder" in n]

    if pae_params:
        other_params = [p for n, p in model.named_parameters() if "pae_encoder" not in n]
        param_groups: list[dict[str, object]] = [
            {"params": other_params, "lr": config.learning_rate},
            {"params": pae_params, "lr": config.pae_learning_rate},
        ]
        optimizer = DmlAdamW(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
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
    model: CurveCopilotModel,
    train_dataset: CurveCopilotDataset,
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
    accumulated: dict[str, float] = {}
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

            context = batch["context_keyframes"].to(device, non_blocking=True)
            prop_type = batch["property_type"].to(device, non_blocking=True)
            topo_features = batch["topology_features"].to(device, non_blocking=True)
            bone_name_tokens = batch["bone_name_tokens"].to(device, non_blocking=True)
            query_times = batch["query_times"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            valid_steps = batch["valid_steps"].to(device, non_blocking=True)

            curve_window = None
            if model.config.use_pae and "curve_window" in batch:
                curve_window = batch["curve_window"].to(device, non_blocking=True)

            compute_start = time.perf_counter()

            predictions, confidences = model(
                context, prop_type, topo_features, bone_name_tokens, query_times,
                curve_window=curve_window,
            )
            loss, metrics = compute_multistep_loss(
                predictions, confidences, target, valid_steps,
                config.training.loss_weights, context,
                config.training.step_decay,
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

            for key, val in metrics.items():
                accumulated[key] = accumulated.get(key, 0.0) + val
            num_batches += 1
            epoch_step += 1

            if epoch_step % config.output.log_every_steps == 0:
                avg = accumulated["loss/total"] / num_batches
                chunk_info = f" [chunk {chunk_idx + 1}/{num_chunks}]" if num_chunks > 1 else ""
                print(f"  epoch {epoch} step {epoch_step}{chunk_info}: loss={avg:.4f}", flush=True)

    if batch_timing:
        batch_timing.end_epoch()

    n = max(num_batches, 1)
    result = {f"train/{k.removeprefix('loss/')}": v / n for k, v in accumulated.items()}
    default_keys = ("total", "value", "tangent", "interpolation",
                    "confidence", "smoothness", "frequency")
    for key in default_keys:
        result.setdefault(f"train/{key}", 0.0)
    return result, global_step + epoch_step


@torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
def validate(
    model: CurveCopilotModel,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    accumulated: dict[str, float] = {}
    num_batches = 0

    for batch in dataloader:
        context = batch["context_keyframes"].to(device, non_blocking=True)
        prop_type = batch["property_type"].to(device, non_blocking=True)
        topo_features = batch["topology_features"].to(device, non_blocking=True)
        bone_name_tokens = batch["bone_name_tokens"].to(device, non_blocking=True)
        query_times = batch["query_times"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid_steps = batch["valid_steps"].to(device, non_blocking=True)

        curve_window = None
        if model.config.use_pae and "curve_window" in batch:
            curve_window = batch["curve_window"].to(device, non_blocking=True)

        predictions, confidences = model(
            context, prop_type, topo_features, bone_name_tokens, query_times,
            curve_window=curve_window,
        )
        _, metrics = compute_multistep_loss(
            predictions, confidences, target, valid_steps,
            config.training.loss_weights, context,
            config.training.step_decay,
        )

        for key, val in metrics.items():
            accumulated[key] = accumulated.get(key, 0.0) + val
        num_batches += 1

    n = max(num_batches, 1)
    result = {f"val/{k.removeprefix('loss/')}": v / n for k, v in accumulated.items()}
    default_keys = ("total", "value", "tangent", "interpolation",
                    "confidence", "smoothness", "frequency")
    for key in default_keys:
        result.setdefault(f"val/{key}", 0.0)
    return result


def save_checkpoint(
    model: CurveCopilotModel,
    optimizer: DmlAdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    state: CheckpointState,
    metrics: dict[str, float],
    path: str | Path,
) -> None:
    save_training_checkpoint(
        model, optimizer, scheduler, state, metrics,
        asdict(model.config), path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return load_training_checkpoint(path)


def _migrate_mha_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    remap = {
        ".attn.in_proj_weight": ".attn.qkv_proj.weight",
        ".attn.in_proj_bias": ".attn.qkv_proj.bias",
    }
    migrated: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        for old_suffix, new_suffix in remap.items():
            if old_suffix in key:
                new_key = key.replace(old_suffix, new_suffix)
                break
        migrated[new_key] = value
    return migrated


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

    model = CurveCopilotModel(config.model).to(device)

    if config.training.pae_pretrained and model.pae_encoder is not None:
        pae_path = resolve_data_path(config.training.pae_pretrained)
        pae_ckpt = torch.load(pae_path, map_location="cpu", weights_only=False)  # type: ignore[no-any-return]
        model.pae_encoder.load_state_dict(pae_ckpt["encoder_state_dict"])
        print(f"Loaded PAE pretrained weights from {pae_path}", flush=True)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    prep_log.log("model_created", params=sum(p.numel() for p in model.parameters()))

    train_paths = [resolve_data_path(p) for p in config.data.train_files]
    budget = create_memory_budget()

    val_dataset = CurveCopilotDataset(
        train_paths, split=config.data.val_split, prep_log=prep_log,
        memory_budget=budget, budget_name="val",
    )
    train_dataset = CurveCopilotDataset(
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

    timing_log = TimingLog("curve_copilot")
    batch_timing = BatchTimingLog("curve_copilot")
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
        ckpt = load_checkpoint(resolve_data_path(resume_path), device)
        model.load_state_dict(_migrate_mha_keys(ckpt["model_state_dict"]), strict=False)
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
            train_total = train_metrics["train/total"]
            val_total = val_metrics["val/total"]
            print(
                f"  train={train_total:.4f}  val={val_total:.4f}"
                f"  (v={val_metrics['val/value']:.4f}"
                f" t={val_metrics['val/tangent']:.4f}"
                f" i={val_metrics['val/interpolation']:.4f}"
                f" c={val_metrics['val/confidence']:.4f}"
                f" s={val_metrics['val/smoothness']:.4f}"
                f" f={val_metrics['val/frequency']:.4f})",
                flush=True,
            )

            checkpoint_time_sec = 0.0

            if val_total < best_val_loss:
                best_val_loss = val_total
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

            loss_detail = {
                f"{prefix}_{name}": round(all_metrics[key], 6)
                for prefix, phase in [("train", "train"), ("val", "val")]
                for name, key in [
                    ("loss", f"{phase}/total"),
                    ("value", f"{phase}/value"),
                    ("tangent", f"{phase}/tangent"),
                    ("interp", f"{phase}/interpolation"),
                    ("confidence", f"{phase}/confidence"),
                    ("smoothness", f"{phase}/smoothness"),
                    ("freq", f"{phase}/frequency"),
                ]
            }
            timing_log.write_epoch(epoch, {
                "train_sec": round(train_time["elapsed"], 3),
                "val_sec": round(val_time["elapsed"], 3),
                "checkpoint_sec": round(checkpoint_time_sec, 3),
                "total_sec": round(
                    train_time["elapsed"] + val_time["elapsed"] + checkpoint_time_sec, 3,
                ),
                "num_steps": steps_per_epoch,
                **loss_detail,
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
    prep_log = PreparationLog("curve_copilot")
    prep_log.log("main_started")

    parser = argparse.ArgumentParser(description="Train Curve Copilot model")
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
