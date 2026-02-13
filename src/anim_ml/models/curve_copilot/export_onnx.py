from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from anim_ml.models.curve_copilot.model import (
    CurveCopilotConfig,
    CurveCopilotModel,
)
from anim_ml.models.curve_copilot.train import load_checkpoint


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> CurveCopilotModel:
    if device is None:
        device = torch.device("cpu")

    ckpt = load_checkpoint(checkpoint_path, device)
    config_dict: dict[str, Any] = ckpt["config"]
    config = CurveCopilotConfig(**config_dict)

    model = CurveCopilotModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def export_to_onnx(
    model: CurveCopilotModel,
    output_path: str | Path,
    opset_version: int = 17,
) -> None:
    model.eval()
    device = next(model.parameters()).device
    max_seq = model.config.max_seq

    dummy_context = torch.randn(2, max_seq, 6, device=device)
    dummy_prop = torch.zeros(2, dtype=torch.long, device=device)
    dummy_joint = torch.zeros(2, dtype=torch.long, device=device)
    dummy_time = torch.tensor([0.5, 0.3], device=device)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    batch = torch.export.Dim("batch", min=1, max=256)
    dynamic_shapes = {
        "context_keyframes": {0: batch},
        "property_type": {0: batch},
        "joint_category": {0: batch},
        "query_time": {0: batch},
    }

    torch.onnx.export(  # type: ignore[reportUnknownMemberType]
        model,
        (dummy_context, dummy_prop, dummy_joint, dummy_time),
        str(output_path),
        opset_version=opset_version,
        input_names=["context_keyframes", "property_type", "joint_category", "query_time"],
        output_names=["prediction", "confidence"],
        dynamic_shapes=dynamic_shapes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Curve Copilot to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    model = load_model_from_checkpoint(args.checkpoint)
    export_to_onnx(model, args.output, args.opset)
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
