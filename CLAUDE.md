# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Training pipeline for ML models used by the Rust rendering engine (`Rust_Rendering/src/ml/`).
Produces ONNX models for embedded CPU inference and hosts a gRPC server for Text-to-Motion.

Three models:
1. **Curve Copilot** — Predicts next keyframe value + tangent handles on animation curves (~4M params, ONNX)
2. **Rig Propagation** — Neural IK: predicts linked bone adjustments from a single bone edit (~4M params, ONNX)
3. **Text-to-Motion** — Generates humanoid animation from text prompts (Light-T2M/MoMask, gRPC server)

## Tech Stack

- Python, PyTorch, ONNX (opset 17+), gRPC
- Package manager: uv
- Linting: ruff
- Type checking: pyright

## Commands

```bash
uv sync                                    # Install dependencies
uv run ruff check src/                     # Lint
uv run pyright src/                        # Type check
uv run pytest tests/                       # All tests
uv run pytest tests/test_foo.py -k "name"  # Single test

uv run python -m anim_ml.models.curve_copilot.train --config configs/curve_copilot.yaml
uv run python -m anim_ml.models.curve_copilot.export_onnx --checkpoint runs/latest/model.pt --output exports/curve_copilot.onnx

uv run python -m anim_ml.server.service --port 50051
```

## Architecture

```
src/anim_ml/
├── data/          # BVH parsing, curve extraction, keyframe reduction, Bezier fitting
├── models/
│   ├── curve_copilot/      # Small causal Transformer (~4M params)
│   └── rig_propagation/    # GNN on skeleton hierarchy (~4M params)
├── server/        # gRPC Text-to-Motion server + HumanML3D→curve conversion
└── utils/         # Rotation math, skeleton utilities
```

## ONNX Export Constraints

Embedded models consumed by `ort` crate in Rust must satisfy:
- 1-5M parameters, float32, opset 17+
- CPU inference < 5ms (batch=1)
- Dynamic axes on batch dimension only

## Data

Training data from CMU MoCap (public domain) + 100STYLE (CC BY 4.0).
Mixamo is prohibited for ML training (Adobe ToS). AMASS is research-only.

## Design Documents

Detailed design in `.claude/local/Design/`:
- `Architecture.md` — Repository structure, tech stack, phased delivery
- `DataPipeline.md` — BVH→training sample conversion pipeline
- `CurveCopilotModel.md` — Model architecture, I/O spec, training, ONNX export
- `RigPropagationModel.md` — GNN architecture, skeleton mapping, training
- `TextToMotionServer.md` — gRPC proto, motion conversion pipeline, deployment

## Last Conservation
- read the last conversation before start session
- .claude/local/last-conversation