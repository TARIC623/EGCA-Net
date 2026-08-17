"""Install the minimal EGCA-Net parser patch into an Ultralytics source checkout.

This script copies the two custom modules into ``ultralytics/nn/modules`` and
updates ``ultralytics/nn/tasks.py`` so YAML parsing recognizes ESFF and SCR.
It creates a one-time backup of tasks.py before changing it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def add_to_frozenset(text: str, block_name: str, symbol: str) -> str:
    """Add ``symbol`` to one parser frozenset, unless it is already present."""
    start = text.find(f"{block_name} = frozenset(")
    if start < 0:
        raise RuntimeError(f"Could not find {block_name} in tasks.py.")
    end = text.find("        }", start)
    if end < 0:
        raise RuntimeError(f"Could not find the end of {block_name} in tasks.py.")
    block = text[start:end]
    if symbol in block:
        return text
    return text[:end] + f"            {symbol},\n" + text[end:]


def add_to_inline_base_set(text: str, symbol: str) -> str:
    """Support older Ultralytics forks that use ``elif m in {...}`` inline."""
    start = text.find("elif m in {")
    if start < 0:
        raise RuntimeError("Could not find an inline parser base-module set in tasks.py.")
    end = text.find("        }:", start)
    if end < 0:
        raise RuntimeError("Could not find the end of the inline parser base-module set in tasks.py.")
    block = text[start:end]
    if symbol in block:
        return text
    return text[:end] + f"            {symbol},\n" + text[end:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ultralytics-root", type=Path, required=True, help="Ultralytics source checkout root")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    target = args.ultralytics_root.resolve()
    modules_dir = target / "ultralytics" / "nn" / "modules"
    tasks_path = target / "ultralytics" / "nn" / "tasks.py"
    if not modules_dir.is_dir() or not tasks_path.is_file():
        raise SystemExit("--ultralytics-root must contain ultralytics/nn/modules and ultralytics/nn/tasks.py")

    shutil.copy2(repo_root / "egca_modules" / "esff.py", modules_dir / "esff.py")
    shutil.copy2(repo_root / "egca_modules" / "scr.py", modules_dir / "scr.py")

    source = tasks_path.read_text(encoding="utf-8")
    imports = "from ultralytics.nn.modules.esff import ESFF\nfrom ultralytics.nn.modules.scr import SCR\n"
    if "ultralytics.nn.modules.esff import ESFF" not in source:
        marker = "from ultralytics.utils import"
        offset = source.find(marker)
        if offset < 0:
            raise RuntimeError("Could not find the Ultralytics utility import marker in tasks.py.")
        source = source[:offset] + imports + source[offset:]

    if "base_modules = frozenset(" in source:
        source = add_to_frozenset(source, "base_modules", "SCR")
        source = add_to_frozenset(source, "repeat_modules", "SCR")
    else:
        source = add_to_inline_base_set(source, "SCR")

    if "if m is ESFF:" not in source:
        if "base_modules = frozenset(" in source:
            marker = "        if m in base_modules:"
            special_case = """        if m is ESFF:\n            c2 = ch[f]\n            args = [c2, *args]\n        elif m in base_modules:"""
        else:
            marker = "        elif m in {"
            special_case = """        elif m is ESFF:\n            c2 = ch[f]\n            args = [c2, *args]\n        elif m in {"""
        if marker not in source:
            raise RuntimeError("Could not find the parser base-module branch in tasks.py.")
        source = source.replace(marker, special_case, 1)

    backup = tasks_path.with_suffix(".py.egca-net.bak")
    if not backup.exists():
        shutil.copy2(tasks_path, backup)
    tasks_path.write_text(source, encoding="utf-8")
    print(f"Patched {tasks_path}")
    print("Custom modules installed: ESFF, SCR")


if __name__ == "__main__":
    main()
