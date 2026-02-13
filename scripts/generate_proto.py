from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    msg = "Could not find project root (pyproject.toml not found)"
    raise RuntimeError(msg)


def generate_proto_stubs(project_root: Path) -> None:
    proto_dir = project_root / "src" / "anim_ml" / "server" / "proto"
    proto_file = proto_dir / "animation_ml.proto"

    if not proto_file.exists():
        msg = f"Proto file not found: {proto_file}"
        raise FileNotFoundError(msg)

    output_dir = proto_dir

    subprocess.check_call([
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        f"--pyi_out={output_dir}",
        str(proto_file),
    ])

    fix_generated_imports(output_dir)

    print(f"Generated proto stubs in {output_dir}")


def fix_generated_imports(output_dir: Path) -> None:
    grpc_file = output_dir / "animation_ml_pb2_grpc.py"
    if not grpc_file.exists():
        return

    content = grpc_file.read_text()
    fixed = re.sub(
        r"^import animation_ml_pb2 as animation__ml__pb2$",
        "from . import animation_ml_pb2 as animation__ml__pb2",
        content,
        flags=re.MULTILINE,
    )

    if fixed != content:
        grpc_file.write_text(fixed)
        print("Fixed import paths in animation_ml_pb2_grpc.py")


if __name__ == "__main__":
    root = find_project_root()
    generate_proto_stubs(root)
