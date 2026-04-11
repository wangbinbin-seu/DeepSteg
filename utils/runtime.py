from __future__ import annotations

from typing import Iterable, Optional

import torch


def get_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wrap_model_for_cuda(model: torch.nn.Module, device: torch.device, device_ids: Optional[Iterable[int]] = None):
    """Move model to device and optionally wrap with DataParallel.

    - If cuda is available and device_ids has length>1, wraps with torch.nn.DataParallel.
    - Returns possibly-wrapped model.
    """
    model = model.to(device)
    if device.type == "cuda" and device_ids:
        ids = list(device_ids)
        if len(ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=ids)
    return model
