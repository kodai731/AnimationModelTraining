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

from anim_ml.data.rig_dataset import RigPropagationDataset
from anim_ml.models.rig_propagation.model import RigPropagationConfig, RigPropagationModel


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
) -> torch.Tensor:
    dot = (pred * target).sum(dim=-1)
    angle = 2.0 * torch.acos(torch.clamp(dot.abs(), 0.0, 1.0))
    return angle.mean()


def compute_loss(
    rotation_deltas: torch.Tensor,
    confidence: torch.Tensor,
    target_deltas: torch.Tensor,
    confidence_targets: torch.Tensor,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    rotation_loss = compute_rotation_loss(rotation_deltas, target_deltas)

    confidence_loss = nn.functional.binary_cross_entropy(
        confidence.squeeze(-1), confidence_targets,
    )

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
    model: RigPropagationModel,
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
        joint_features = batch["joint_features"].to(device)
        joint_types = batch["joint_types"].to(device)
        target_deltas = batch["target_deltas"].to(device)
        confidence_targets = batch["confidence_targets"].to(device)

        rotation_deltas, confidence = model(joint_features, joint_types)
        loss, metrics = compute_loss(
            rotation_deltas, confidence, target_deltas,
            confidence_targets, config.training.loss_weights,
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
    model: RigPropagationModel,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        joint_features = batch["joint_features"].to(device)
        joint_types = batch["joint_types"].to(device)
        target_deltas = batch["target_deltas"].to(device)
        confidence_targets = batch["confidence_targets"].to(device)

        rotation_deltas, confidence = model(joint_features, joint_types)
        _, metrics = compute_loss(
            rotation_deltas, confidence, target_deltas,
            confidence_targets, config.training.loss_weights,
        )

        total_loss += metrics["loss/total"]
        num_batches += 1

    return {"loss/val": total_loss / max(num_batches, 1)}


def save_checkpoint(
    model: RigPropagationModel,
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
            "num_joints": model.config.num_joints,
            "node_feature_dim": model.config.node_feature_dim,
            "edge_feature_dim": model.config.edge_feature_dim,
            "hidden_dim": model.config.hidden_dim,
            "ffn_dim": model.config.ffn_dim,
            "num_message_passing_layers": model.config.num_message_passing_layers,
            "num_joint_types": model.config.num_joint_types,
            "joint_type_embed_dim": model.config.joint_type_embed_dim,
            "input_feature_dim": model.config.input_feature_dim,
            "dropout": model.config.dropout,
        },
    }, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)  # type: ignore[no-any-return]


def train(config: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_paths = [Path(p) for p in config.data.train_files]
    train_dataset = RigPropagationDataset(train_paths, split="train")
    val_dataset = RigPropagationDataset(train_paths, split=config.data.val_split)

    adjacency: torch.Tensor = torch.from_numpy(train_dataset.adjacency)  # type: ignore[no-any-return]

    model = RigPropagationModel(config.model, adjacency=adjacency).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )

    steps_per_epoch = max(len(train_loader), 1)
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config.training, steps_per_epoch,
    )

    checkpoint_dir = Path(config.output.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    all_metrics: dict[str, float] = {}

    for epoch in range(1, config.training.epochs + 1):
        print(f"Epoch {epoch}/{config.training.epochs}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, config, device, epoch,
        )
        val_metrics = validate(model, val_loader, config, device)

        all_metrics = {**train_metrics, **val_metrics}
        print(f"  train_loss={train_metrics['loss/train']:.4f}"
              f"  val_loss={val_metrics['loss/val']:.4f}")

        if val_metrics["loss/val"] < best_val_loss:
            best_val_loss = val_metrics["loss/val"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, all_metrics,
                checkpoint_dir / "best.pt",
            )

        if epoch % config.output.save_every_epochs == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, all_metrics,
                checkpoint_dir / f"epoch_{epoch}.pt",
            )

    save_checkpoint(
        model, optimizer, scheduler, config.training.epochs, all_metrics,
        checkpoint_dir / "last.pt",
    )
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Rig Propagation model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
