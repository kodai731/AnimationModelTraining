from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from anim_ml.data.curve_dataset import CurveCopilotDataset
from anim_ml.models.curve_copilot.model import CurveCopilotConfig, CurveCopilotModel
from anim_ml.paths import resolve_data_path
from anim_ml.utils.timing_log import TimingLog


@dataclass
class LossWeights:
    value: float = 1.0
    tangent: float = 0.5
    interpolation: float = 0.3
    confidence: float = 0.2


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
    training_cfg = TrainingConfig(**training_raw, loss_weights=loss_weights)

    data_cfg = DataConfig(**raw.get("data", {}))
    output_cfg = OutputConfig(**raw.get("output", {}))

    return TrainConfig(
        model=model_cfg,
        training=training_cfg,
        data=data_cfg,
        output=output_cfg,
    )


def compute_loss(
    prediction: torch.Tensor,
    confidence: torch.Tensor,
    target: torch.Tensor,
    weights: LossWeights,
    confidence_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = nn.functional.mse_loss

    value_loss = mse(prediction[:, 0], target[:, 1])
    tangent_loss = mse(prediction[:, 1:5], target[:, 2:6])

    tangent_magnitude: torch.Tensor = torch.norm(target[:, 2:6], dim=1)  # type: ignore[assignment]
    interp_target: torch.Tensor = (tangent_magnitude > 0.01).float()  # type: ignore[assignment]
    interp_loss = nn.functional.binary_cross_entropy_with_logits(
        prediction[:, 5], interp_target,  # type: ignore[arg-type]
    )

    confidence_loss = mse(confidence, confidence_targets)

    total = (
        weights.value * value_loss
        + weights.tangent * tangent_loss
        + weights.interpolation * interp_loss
        + weights.confidence * confidence_loss
    )

    metrics = {
        "loss/total": total.item(),
        "loss/value": value_loss.item(),
        "loss/tangent": tangent_loss.item(),
        "loss/interpolation": interp_loss.item(),
        "loss/confidence": confidence_loss.item(),
    }
    return total, metrics


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(
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
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: TrainConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(dataloader):
        context = batch["context_keyframes"].to(device, non_blocking=True)
        prop_type = batch["property_type"].to(device, non_blocking=True)
        joint_cat = batch["joint_category"].to(device, non_blocking=True)
        query_time = batch["query_time"].to(device, non_blocking=True).squeeze(-1)
        target = batch["target"].to(device, non_blocking=True)

        with torch.no_grad():
            last_context_value = context[:, -1, 1]
            target_value = target[:, 1]
            value_distance = torch.abs(target_value - last_context_value)
            confidence_targets = torch.exp(-value_distance * 5.0).unsqueeze(-1)

        prediction, confidence = model(context, prop_type, joint_cat, query_time)
        loss, metrics = compute_loss(
            prediction, confidence, target, config.training.loss_weights, confidence_targets,
        )

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        optimizer.step()  # type: ignore[no-untyped-call]
        scheduler.step()

        total_loss += metrics["loss/total"]
        num_batches += 1

        if (step + 1) % config.output.log_every_steps == 0:
            avg = total_loss / num_batches
            print(f"  epoch {epoch} step {step + 1}: loss={avg:.4f}")

    return {"loss/train": total_loss / max(num_batches, 1)}


@torch.no_grad()
def validate(
    model: CurveCopilotModel,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        context = batch["context_keyframes"].to(device, non_blocking=True)
        prop_type = batch["property_type"].to(device, non_blocking=True)
        joint_cat = batch["joint_category"].to(device, non_blocking=True)
        query_time = batch["query_time"].to(device, non_blocking=True).squeeze(-1)
        target = batch["target"].to(device, non_blocking=True)

        last_context_value = context[:, -1, 1]
        target_value = target[:, 1]
        value_distance = torch.abs(target_value - last_context_value)
        confidence_targets = torch.exp(-value_distance * 5.0).unsqueeze(-1)

        prediction, confidence = model(context, prop_type, joint_cat, query_time)
        _, metrics = compute_loss(
            prediction, confidence, target, config.training.loss_weights, confidence_targets,
        )

        total_loss += metrics["loss/total"]
        num_batches += 1

    return {"loss/val": total_loss / max(num_batches, 1)}


def save_checkpoint(
    model: CurveCopilotModel,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    metrics: dict[str, float],
    path: str | Path,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "config": {
            "d_model": model.config.d_model,
            "n_heads": model.config.n_heads,
            "d_ff": model.config.d_ff,
            "n_layers": model.config.n_layers,
            "max_seq": model.config.max_seq,
            "dropout": model.config.dropout,
            "num_property_types": model.config.num_property_types,
            "num_joint_categories": model.config.num_joint_categories,
            "keyframe_dim": model.config.keyframe_dim,
        },
    }, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)  # type: ignore[no-any-return]


def train(config: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CurveCopilotModel(config.model).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_paths = [resolve_data_path(p) for p in config.data.train_files]
    train_dataset = CurveCopilotDataset(train_paths, split="train")
    val_dataset = CurveCopilotDataset(train_paths, split=config.data.val_split)

    use_workers = config.data.num_workers > 0

    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=use_workers,
    )
    val_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=use_workers,
    )

    steps_per_epoch = max(len(train_loader), 1)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config.training, steps_per_epoch,
    )

    checkpoint_dir = resolve_data_path(config.output.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timing_log = TimingLog("curve_copilot")
    print(f"Timing log: {timing_log.path}")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = config.training.early_stopping_patience
    all_metrics: dict[str, float] = {}
    epoch = 0

    try:
        for epoch in range(1, config.training.epochs + 1):
            print(f"Epoch {epoch}/{config.training.epochs}")

            with TimingLog.measure() as train_time:
                train_metrics = train_one_epoch(
                    model, train_loader, optimizer, scheduler, config, device, epoch,
                )

            with TimingLog.measure() as val_time:
                val_metrics = validate(model, val_loader, config, device)

            all_metrics = {**train_metrics, **val_metrics}
            print(f"  train_loss={train_metrics['loss/train']:.4f}"
                  f"  val_loss={val_metrics['loss/val']:.4f}")

            checkpoint_time_sec = 0.0

            if val_metrics["loss/val"] < best_val_loss:
                best_val_loss = val_metrics["loss/val"]
                epochs_without_improvement = 0
                with TimingLog.measure() as ckpt_time:
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, all_metrics,
                        checkpoint_dir / "best.pt",
                    )
                checkpoint_time_sec += ckpt_time["elapsed"]
            else:
                epochs_without_improvement += 1

            if epoch % config.output.save_every_epochs == 0:
                with TimingLog.measure() as ckpt_time:
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, all_metrics,
                        checkpoint_dir / f"epoch_{epoch}.pt",
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

            if patience > 0 and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch}.")

    if epoch > 0:
        save_checkpoint(
            model, optimizer, scheduler, epoch, all_metrics,
            checkpoint_dir / "last.pt",
        )
        print(f"Saved last.pt (epoch {epoch}).")

    if best_val_loss < float("inf"):
        print(f"Best val loss: {best_val_loss:.4f} (best.pt)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Curve Copilot model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
