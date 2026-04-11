from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(value: str, base_dir: Path) -> str:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _normalize_device_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(',') if v.strip()]
    return [int(value)]


def load_options(default_opt_path: str) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--opt", type=str, default=default_opt_path)
    args, _ = parser.parse_known_args()

    base_dir = _repo_root()
    opt_path = Path(args.opt)
    if not opt_path.is_absolute():
        opt_path = (base_dir / opt_path).resolve()

    if not opt_path.exists():
        raise FileNotFoundError(f"Options file not found: {opt_path}")

    with opt_path.open("r", encoding="utf-8") as f:
        options = yaml.safe_load(f) or {}

    resolved: Dict[str, Any] = dict(options)
    resolved["opt_path"] = str(opt_path)
    resolved["root_dir"] = str(base_dir)

    for key, value in list(resolved.items()):
        if not isinstance(value, str):
            continue
        if key.endswith("_dir") or key.endswith("_path") or key in {
            "model_dir",
            "results_dir",
            "image_path",
            "image_dir",
            "checkpoint_dir",
            "IMAGE_PATH",
        }:
            resolved[key] = _resolve_path(value, base_dir)

    if "device_ids" in resolved:
        resolved["device_ids"] = _normalize_device_ids(resolved.get("device_ids"))
    if "cuda_visible_devices" in resolved and resolved["cuda_visible_devices"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(resolved["cuda_visible_devices"])

    for key, value in list(resolved.items()):
        if key.endswith("_device_ids"):
            resolved[key] = _normalize_device_ids(value)

    return resolved
