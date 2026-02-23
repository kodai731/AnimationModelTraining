from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from anim_ml.models.curve_copilot.model import (
    CurveCopilotConfig,
    CurveCopilotModel,
)
from anim_ml.models.curve_copilot.train import load_checkpoint


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

    state_dict = ckpt["model_state_dict"]
    state_dict = _migrate_mha_keys(state_dict)

    model.load_state_dict(state_dict, strict=False)
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

    dummy_context = torch.randn(2, max_seq, model.config.keyframe_dim, device=device)
    dummy_prop = torch.zeros(2, dtype=torch.long, device=device)
    dummy_topo = torch.randn(2, 6, device=device)
    dummy_tokens = torch.zeros(2, 32, dtype=torch.long, device=device)
    dummy_time = torch.tensor([0.5, 0.3], device=device)

    args: tuple[torch.Tensor, ...] = (
        dummy_context, dummy_prop, dummy_topo, dummy_tokens, dummy_time,
    )
    input_names = [
        "context_keyframes", "property_type",
        "topology_features", "bone_name_tokens", "query_time",
    ]

    if model.config.use_pae:
        dummy_curve_window = torch.randn(2, model.config.pae_window_size, device=device)
        args = args + (dummy_curve_window,)
        input_names.append("curve_window")

    output_names = ["prediction", "confidence"]
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(  # type: ignore[reportUnknownMemberType]
        model,
        args,
        str(output_path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def main() -> None:
    from anim_ml.paths import get_exports_dir, get_runs_dir

    default_checkpoint = str(get_runs_dir() / "curve_copilot" / "best.pt")
    date_tag = datetime.now().strftime("%Y%m%d")
    default_output = str(get_exports_dir() / f"curve_copilot_{date_tag}.onnx")

    parser = argparse.ArgumentParser(description="Export Curve Copilot to ONNX")
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
