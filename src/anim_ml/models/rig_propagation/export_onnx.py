from __future__ import annotations

import argparse
from datetime import datetime
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
    max_joints = model.config.max_joints
    max_edges = model.config.max_edges
    input_dim = model.config.input_feature_dim

    dummy_features = torch.randn(2, max_joints, input_dim, device=device)
    dummy_topo = torch.randn(2, max_joints, 6, device=device)
    dummy_tokens = torch.zeros(2, max_joints, 32, dtype=torch.long, device=device)
    dummy_mask = torch.ones(2, max_joints, device=device)
    dummy_src = torch.zeros(max_edges, dtype=torch.long, device=device)
    dummy_tgt = torch.zeros(max_edges, dtype=torch.long, device=device)
    dummy_edge_dir = torch.zeros(max_edges, dtype=torch.long, device=device)
    dummy_edge_mask = torch.ones(max_edges, device=device)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    input_names = [
        "joint_features", "topology_features", "bone_name_tokens", "joint_mask",
        "source_indices", "target_indices", "edge_direction", "edge_mask",
    ]
    output_names = ["rotation_deltas", "confidence"]

    dynamic_axes = {
        "joint_features": {0: "batch"},
        "topology_features": {0: "batch"},
        "bone_name_tokens": {0: "batch"},
        "joint_mask": {0: "batch"},
        "rotation_deltas": {0: "batch"},
        "confidence": {0: "batch"},
    }

    torch.onnx.export(  # type: ignore[reportUnknownMemberType]
        model,
        (
            dummy_features, dummy_topo, dummy_tokens, dummy_mask,
            dummy_src, dummy_tgt, dummy_edge_dir, dummy_edge_mask,
        ),
        str(output_path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def main() -> None:
    from anim_ml.paths import get_exports_dir, get_runs_dir

    default_checkpoint = str(get_runs_dir() / "rig_propagation" / "best.pt")
    date_tag = datetime.now().strftime("%Y%m%d")
    default_output = str(get_exports_dir() / f"rig_propagation_{date_tag}.onnx")

    parser = argparse.ArgumentParser(description="Export Rig Propagation to ONNX")
    parser.add_argument(
        "--checkpoint", type=str, default=default_checkpoint, help="Model checkpoint path",
    )
    parser.add_argument("--output", type=str, default=default_output, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    model = load_model_from_checkpoint(args.checkpoint)
    export_to_onnx(model, args.output, args.opset)
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
