"""Seeded minibatch training with validation-only selection and raw-unit metrics."""
import time

import torch
from torch_geometric.loader import DataLoader

from kangnn import seed_everything

from .grid import build_model
from .checkpoints import save_best
from .metrics import masked_loss, score_predictions


def statistics(graphs, train_indices):
    x = torch.cat([graphs[i].x for i in train_indices])
    mean = x.mean(0)
    scale = x.std(0, correction=0).clamp_min(1e-6)
    y = torch.cat([graphs[i].y for i in train_indices])
    count = torch.isfinite(y).sum(0)
    ymean = torch.nan_to_num(y).sum(0) / count.clamp_min(1)
    variance = torch.where(torch.isfinite(y), (y-ymean).square(), 0).sum(0) / count.clamp_min(1)
    yscale = variance.sqrt().clamp_min(1e-6)
    yscale = torch.where(count > 1, yscale, torch.ones_like(yscale))
    return mean, scale, ymean, yscale, count > 0


@torch.no_grad()
def evaluate(model, loader, device, stats, kind):
    model.eval()
    mean, scale, ymean, yscale, supported = [value.to(device) for value in stats]
    outputs, labels, losses = [], [], []
    observed_count = 0
    for batch in loader:
        batch = batch.to(device)
        batch.x = (batch.x - mean) / scale
        y = batch.y.clone()
        y[:, ~supported] = float("nan")
        raw = model(batch)
        normalized = (y-ymean)/yscale if kind == "regression" else y
        value = masked_loss(raw, normalized, kind)
        if value is not None:
            count = torch.isfinite(normalized).sum().item()
            losses.append(value.item() * count)
            observed_count += count
        outputs.append((raw*yscale + ymean if kind == "regression" else raw).cpu())
        labels.append(y.cpu())
    result = score_predictions(torch.cat(outputs), torch.cat(labels), kind)
    result["loss"] = sum(losses)/observed_count if observed_count else None
    return result


def train_one(candidate, graphs, splits, kind, *, seed=0, epochs=20, width=32,
              graph_layers=2, batch_size=32, lr=0.001, device="cpu", patience=200, checkpoint_path=None,
              checkpoint_metadata=None, deterministic=True):
    if epochs < 1 or min(width, graph_layers, batch_size, patience) < 1 or lr <= 0:
        raise ValueError("Training budgets and learning rate must be positive")
    seed_everything(seed)
    if not deterministic:
        torch.use_deterministic_algorithms(False)
    stats = statistics(graphs, splits[0])
    model = build_model(candidate, graphs[0].x.size(1), graphs[0].y.size(1), width,
                        graph_layers).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader([graphs[i] for i in splits[0]], batch_size=batch_size,
                               shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader([graphs[i] for i in splits[1]], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader([graphs[i] for i in splits[2]], batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mean, scale, ymean, yscale, _ = [value.to(device) for value in stats]
    best, best_state, best_epoch, stale = None, None, None, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            batch.x = (batch.x-mean)/scale
            y = (batch.y-ymean)/yscale if kind == "regression" else batch.y
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(batch), y, kind)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
        validation = evaluate(model, val_loader, device, stats, kind)
        if validation["score"] is not None:
            objective = (-validation["score"] if kind == "classification" else validation["score"])
            selection_metric = validation["metric"]
        else:
            objective = validation["loss"]
            selection_metric = "validation_loss_fallback"
        if objective is None or not torch.isfinite(torch.tensor(objective)):
            raise ValueError("No finite validation objective")
        history.append({"epoch": epoch, "validation_loss": objective,
                        "validation_score": validation["score"]})
        if best is None or objective < best:
            best, best_epoch, stale = objective, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if checkpoint_path is not None:
                save_best(checkpoint_path, model, best_state, stats,
                          {**(checkpoint_metadata or {}), "seed": seed, "best_epoch": best_epoch,
                           "validation": validation, "selection_metric": selection_metric,
                           "candidate": candidate.to_dict(), "kind": kind})
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    validation = evaluate(model, val_loader, device, stats, kind)
    test = evaluate(model, test_loader, device, stats, kind)
    return {"parameters": parameters, "effective_hidden_channels": model.config.hidden_channels,
            "best_epoch": best_epoch, "epochs_executed": len(history),
            "validation": validation, "test": test, "selection_metric": selection_metric,
            "seconds": time.perf_counter()-started, "history": history,
            "train_observed_tasks": stats[-1].tolist(),
            "best_checkpoint": str(checkpoint_path) if checkpoint_path else None}
