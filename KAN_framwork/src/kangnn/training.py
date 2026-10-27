"""Deterministic CPU training helpers and extensible supervised objectives."""
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

LOSSES = {"cross_entropy": nn.CrossEntropyLoss, "bce_logits": nn.BCEWithLogitsLoss,
          "mse": nn.MSELoss, "l1": nn.L1Loss, "smooth_l1": nn.SmoothL1Loss,
          "nll": nn.NLLLoss}


def register_loss(name, factory):
    """Register factory(**kwargs) -> callable(predictions, targets) -> scalar tensor."""
    if not name or name in LOSSES or not callable(factory):
        raise ValueError("Loss name must be new, nonempty, and factory callable")
    LOSSES[name] = factory


def make_loss(name="cross_entropy", **kwargs):
    if name not in LOSSES:
        raise ValueError(f"Unknown loss {name!r}; choose from {sorted(LOSSES)}")
    loss = LOSSES[name](**kwargs)
    if not callable(loss):
        raise TypeError("Loss factory must return a callable")
    return loss


def seed_everything(seed=0):
    """Call before model construction. Strict repeatability target: same CPU environment."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


@dataclass
class TrainingConfig:
    epochs: int = 100
    lr: float = 0.01
    weight_decay: float = 0.0
    kan_l1: float = 0.0

    def __post_init__(self):
        if self.epochs < 1 or self.lr <= 0 or self.weight_decay < 0 or self.kan_l1 < 0:
            raise ValueError("Require epochs >= 1, lr > 0, and nonnegative penalties")


def coefficient_penalty(model):
    from .transforms import KANLinear
    terms = [m.coefficients.abs().mean() for m in model.modules() if isinstance(m, KANLinear)]
    return sum(terms) if terms else next(model.parameters()).new_zeros(())


def select_targets(predictions, data, mask=None):
    targets = data.y
    if mask is not None:
        if mask.dtype != torch.bool or mask.ndim != 1 or mask.numel() != predictions.size(0):
            raise ValueError("mask must be boolean [num_predictions]")
        if not mask.any():
            raise ValueError("mask must select at least one example")
        predictions, targets = predictions[mask], targets[mask]
    if targets.numel() == 0:
        raise ValueError("No targets selected")
    return predictions, targets


def fit(model, data, loss="cross_entropy", config=None, mask=None):
    """Full-batch training; caller supplies splits, preprocessing, and data device.

    A custom callable may combine several supervised losses. Built-in NLL uses
    log_softmax explicitly; CE and BCEWithLogits consume raw logits.
    """
    config = config or TrainingConfig()
    objective = make_loss(loss) if isinstance(loss, str) else loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history = []
    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions, targets = select_targets(model(data), data, mask)
        if loss == "nll":
            predictions = predictions.log_softmax(-1)
        value = objective(predictions, targets)
        if value.ndim != 0:
            raise ValueError("Loss must return a scalar tensor")
        value = value + config.kan_l1 * coefficient_penalty(model)
        if not torch.isfinite(value):
            raise ValueError("Loss is not finite; check features, targets, and objective")
        value.backward()
        optimizer.step()
        history.append(value.detach().item())
    return history


@torch.no_grad()
def evaluate(model, data, loss="cross_entropy", mask=None):
    was_training = model.training
    model.eval()
    try:
        predictions, targets = select_targets(model(data), data, mask)
        objective = make_loss(loss) if isinstance(loss, str) else loss
        values = predictions.log_softmax(-1) if loss == "nll" else predictions
        result = {"loss": objective(values, targets).item()}
        if targets.dtype == torch.long and targets.ndim == 1:
            result["accuracy"] = (predictions.argmax(-1) == targets).float().mean().item()
        return result
    finally:
        model.train(was_training)
