"""Portable best-checkpoint persistence with fitted train-only transforms."""
from pathlib import Path

import torch

from kangnn import GraphKAN, ModelConfig

from .grid import register_bases


def save_best(path, model, state, stats, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "model_config": model.config.to_dict(),
               "state_dict": state, "statistics": [x.detach().cpu() for x in stats],
               "metadata": metadata}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_best(path, device="cpu"):
    """Return (eval model, fitted-statistics tuple, metadata) on the requested device."""
    register_bases()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported experiment checkpoint version")
    model = GraphKAN(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    return (model.to(device).eval(), tuple(x.to(device) for x in payload["statistics"]),
            payload["metadata"])
