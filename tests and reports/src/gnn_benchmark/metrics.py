"""Missing-label-aware losses and dependency-free, tie-correct ROC-AUC."""
import numpy as np
import torch
from torch.nn import functional as F


def masked_loss(predictions, targets, kind):
    observed = torch.isfinite(targets)
    if not observed.any():
        return None
    p, y = predictions[observed], targets[observed]
    return F.binary_cross_entropy_with_logits(p, y) if kind == "classification" else F.mse_loss(p, y)


def binary_auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    positive = labels == 1
    npos, nneg = int(positive.sum()), int((labels == 0).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[positive].sum() - npos*(npos+1)/2) / (npos*nneg))


def score_predictions(predictions, targets, kind):
    p = predictions.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite model predictions")
    scores = []
    for task in range(y.shape[1]):
        mask = np.isfinite(y[:, task])
        if not mask.any():
            scores.append(None)
        elif kind == "classification":
            scores.append(binary_auc(y[mask, task], p[mask, task]))
        else:
            scores.append(float(np.abs(p[mask, task] - y[mask, task]).mean()))
    valid = [x for x in scores if x is not None]
    return {"score": float(np.mean(valid)) if valid else None,
            "metric": "roc_auc" if kind == "classification" else "mae",
            "valid_tasks": len(valid), "per_task": scores}
