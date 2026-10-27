"""Save model config and weights without pickling executable model objects."""
import torch

from .model import GraphKAN, ModelConfig


def save_model(model, path):
    torch.save({"format_version": 1, "config": model.config.to_dict(),
                "state_dict": model.state_dict()}, path)


def load_model(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    # Register custom bases before loading; their Python code is never embedded.
    model = GraphKAN(ModelConfig(**payload["config"])).to(map_location)
    model.load_state_dict(payload["state_dict"])
    return model
