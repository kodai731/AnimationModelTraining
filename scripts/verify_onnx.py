from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def measure_latency(
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
    num_runs: int = 100,
) -> dict[str, float]:
    for _ in range(10):
        session.run(None, inputs)

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, inputs)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    sorted_latencies = sorted(latencies)
    p95_idx = int(num_runs * 0.95)
    p99_idx = int(num_runs * 0.99)

    return {
        "mean_ms": float(np.mean(latencies)),
        "p95_ms": sorted_latencies[min(p95_idx, num_runs - 1)],
        "p99_ms": sorted_latencies[min(p99_idx, num_runs - 1)],
        "min_ms": sorted_latencies[0],
        "max_ms": sorted_latencies[-1],
    }


def verify_onnx_model(model_path: str | Path, num_runs: int = 100) -> dict[str, object]:
    model_path = Path(model_path)
    results: dict[str, object] = {}

    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    results["checker"] = "passed"

    opset = model.opset_import[0].version
    results["opset"] = opset

    size_mb = model_path.stat().st_size / (1024 * 1024)
    results["size_mb"] = round(size_mb, 2)

    results["inputs"] = [
        {
            "name": inp.name,
            "shape": [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim],
            "dtype": inp.type.tensor_type.elem_type,
        }
        for inp in model.graph.input
    ]
    results["outputs"] = [
        {
            "name": out.name,
            "shape": [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim],
            "dtype": out.type.tensor_type.elem_type,
        }
        for out in model.graph.output
    ]

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    batch1_inputs: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        shape = [1 if isinstance(d, str) else d for d in inp.shape]
        if inp.type == "tensor(int64)":
            batch1_inputs[inp.name] = np.zeros(shape, dtype=np.int64)
        else:
            batch1_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

    latency = measure_latency(session, batch1_inputs, num_runs)
    results["latency"] = latency

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ONNX model")
    parser.add_argument("model", type=str, help="Path to ONNX model")
    parser.add_argument("--runs", type=int, default=100, help="Number of latency runs")
    args = parser.parse_args()

    results = verify_onnx_model(args.model, args.runs)

    print(f"ONNX Checker: {results['checker']}")
    print(f"Opset: {results['opset']}")
    print(f"Size: {results['size_mb']} MB")
    print()

    print("Inputs:")
    for inp in results["inputs"]:  # type: ignore[union-attr]
        print(f"  {inp['name']}: shape={inp['shape']} dtype={inp['dtype']}")  # type: ignore[index]

    print("Outputs:")
    for out in results["outputs"]:  # type: ignore[union-attr]
        print(f"  {out['name']}: shape={out['shape']} dtype={out['dtype']}")  # type: ignore[index]

    print()
    latency = results["latency"]
    print(f"Latency (batch=1, {args.runs} runs):")
    print(f"  mean: {latency['mean_ms']:.2f} ms")  # type: ignore[index]
    print(f"  p95:  {latency['p95_ms']:.2f} ms")  # type: ignore[index]
    print(f"  p99:  {latency['p99_ms']:.2f} ms")  # type: ignore[index]


if __name__ == "__main__":
    main()
