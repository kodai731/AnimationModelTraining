from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from anim_ml.models.periodic_autoencoder.model import PAEConfig, PeriodicAutoencoder
from anim_ml.paths import resolve_data_path
from anim_ml.utils.device import detect_training_device


@dataclass
class PAETrainingConfig:
    batch_size: int = 512
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.01


@dataclass
class PAETrainConfig:
    model: PAEConfig = field(default_factory=PAEConfig)
    training: PAETrainingConfig = field(default_factory=PAETrainingConfig)
    data_files: list[str] = field(default_factory=lambda: [])
    output_dir: str = "runs/periodic_autoencoder"


def load_pae_config(config_path: str | Path) -> PAETrainConfig:
    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    model_cfg = PAEConfig(**raw.get("model", {}))
    training_cfg = PAETrainingConfig(**raw.get("training", {}))
    data_files = raw.get("data_files", [])
    output_dir = raw.get("output_dir", "runs/periodic_autoencoder")

    return PAETrainConfig(
        model=model_cfg,
        training=training_cfg,
        data_files=data_files,
        output_dir=output_dir,
    )


def load_curve_windows(data_files: list[str], split: str = "train") -> torch.Tensor:
    all_windows: list[np.ndarray] = []

    for path_str in data_files:
        path = resolve_data_path(path_str)
        with h5py.File(path, "r") as f:
            if split not in f:
                continue
            grp = cast("h5py.Group", f[split])
            if "curve_window" not in grp:
                continue
            ds = cast("h5py.Dataset", grp["curve_window"])
            all_windows.append(np.asarray(ds[:], dtype=np.float32))

    if not all_windows:
        return torch.zeros(0, 64)

    return torch.as_tensor(np.concatenate(all_windows), dtype=torch.float32)


def train_pae(config: PAETrainConfig, device_override: str | None = None) -> None:
    device = detect_training_device(device_override)
    print(f"Using device: {device}", flush=True)

    model = PeriodicAutoencoder(config.model).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"PAE parameters: {param_count:,}", flush=True)

    train_data = load_curve_windows(config.data_files, "train")
    val_data = load_curve_windows(config.data_files, "val")
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}", flush=True)

    if len(train_data) == 0:
        print("No training data found. Exiting.", flush=True)
        return

    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=config.training.batch_size,
    ) if len(val_data) > 0 else None

    from anim_ml.utils.optimizer import DmlAdamW
    optimizer = DmlAdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    output_dir = Path(resolve_data_path(config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for (batch_data,) in train_loader:
            batch_data = batch_data.to(device)
            _, reconstructed = model(batch_data)
            loss = nn.functional.mse_loss(reconstructed, batch_data)

            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / max(num_batches, 1)

        val_loss_str = ""
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for (batch_data,) in val_loader:
                    batch_data = batch_data.to(device)
                    _, reconstructed = model(batch_data)
                    val_total += nn.functional.mse_loss(reconstructed, batch_data).item()
                    val_batches += 1
            val_loss = val_total / max(val_batches, 1)
            val_loss_str = f"  val_loss={val_loss:.6f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({  # pyright: ignore[reportUnknownMemberType]
                    "encoder_state_dict": model.encoder.state_dict(),
                    "decoder_state_dict": model.decoder.state_dict(),
                    "config": {
                        "window_size": config.model.window_size,
                        "latent_channels": config.model.latent_channels,
                        "feature_dim": config.model.feature_dim,
                    },
                }, output_dir / "best.pt")

        print(
            f"Epoch {epoch}/{config.training.epochs}"
            f"  train_loss={train_loss:.6f}{val_loss_str}",
            flush=True,
        )

    torch.save({  # pyright: ignore[reportUnknownMemberType]
        "encoder_state_dict": model.encoder.state_dict(),
        "decoder_state_dict": model.decoder.state_dict(),
        "config": {
            "window_size": config.model.window_size,
            "latent_channels": config.model.latent_channels,
            "feature_dim": config.model.feature_dim,
        },
    }, output_dir / "last.pt")
    print(f"Saved to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain Periodic Autoencoder")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_pae_config(args.config)
    train_pae(config, device_override=args.device)


if __name__ == "__main__":
    main()
