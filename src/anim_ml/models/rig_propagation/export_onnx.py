from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from anim_ml.models.rig_propagation.model import (
    RigPropagationConfig,
    RigPropagationModel,
)
from anim_ml.models.rig_propagation.train import load_checkpoint


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> RigPropagationModel:
    if device is None:
        device = torch.device("cpu")

    ckpt = load_checkpoint(checkpoint_path, device)
    config_dict: dict[str, Any] = ckpt["config"]
    config = RigPropagationConfig(**config_dict)

    model = RigPropagationModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def export_to_onnx(
    model: RigPropagationModel,
    output_path: str | Path,
    opset_version: int = 17,
) -> None:
    model.eval()
    device = next(model.parameters()).device
    num_joints = model.config.num_joints
    input_dim = model.config.input_feature_dim

    dummy_features = torch.randn(2, num_joints, input_dim, device=device)
    dummy_types = torch.zeros(2, num_joints, dtype=torch.long, device=device)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    batch = torch.export.Dim("batch", min=1, max=256)
    dynamic_shapes = {
        "joint_features": {0: batch},
        "joint_types": {0: batch},
    }

    torch.onnx.export(  # type: ignore[reportUnknownMemberType]
        model,
        (dummy_features, dummy_types),
        str(output_path),
        opset_version=opset_version,
        input_names=["joint_features", "joint_types"],
        output_names=["rotation_deltas", "confidence"],
        dynamic_shapes=dynamic_shapes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Rig Propagation to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    model = load_model_from_checkpoint(args.checkpoint)
    export_to_onnx(model, args.output, args.opset)
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
