from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from anim_ml.utils.optimizer import DmlAdamW


@dataclass
class CheckpointState:
    epoch: int
    global_step: int
    is_mid_epoch: bool
    best_val_loss: float
    epochs_without_improvement: int


def save_training_checkpoint(
    model: nn.Module,
    optimizer: DmlAdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    state: CheckpointState,
    metrics: dict[str, float],
    model_config: dict[str, Any],
    path: str | Path,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({  # pyright: ignore[reportUnknownMemberType]
        "epoch": state.epoch,
        "global_step": state.global_step,
        "is_mid_epoch": state.is_mid_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "best_val_loss": state.best_val_loss,
        "epochs_without_improvement": state.epochs_without_improvement,
        "config": model_config,
    }, path)


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[no-any-return]


def restore_checkpoint_state(ckpt: dict[str, Any]) -> CheckpointState:
    return CheckpointState(
        epoch=ckpt["epoch"],
        global_step=ckpt.get("global_step", 0),
        is_mid_epoch=ckpt.get("is_mid_epoch", False),
        best_val_loss=ckpt.get("best_val_loss", float("inf")),
        epochs_without_improvement=ckpt.get("epochs_without_improvement", 0),
    )
