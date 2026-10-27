"""Feature preparation and reproducible splits without fitting on held-out nodes."""
import torch
from torch import nn


class FeatureStandardizer(nn.Module):
    """Fit only on training rows; transform never changes the input tensor."""

    def __init__(self, num_features):
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("scale", torch.ones(num_features))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, x, mask=None):
        if x.ndim != 2 or x.size(1) != self.mean.numel() or not x.is_floating_point():
            raise ValueError("Expected floating features [num_nodes, num_features]")
        if mask is not None and (mask.dtype != torch.bool or mask.shape != (x.size(0),)):
            raise ValueError("Training mask must be boolean [num_nodes]")
        rows = x if mask is None else x[mask]
        if rows.size(0) == 0 or not torch.isfinite(rows).all():
            raise ValueError("Training features must be nonempty and finite")
        self.mean.copy_(rows.mean(0))
        self.scale.copy_(rows.std(0, correction=0).clamp_min(1e-6))
        self.fitted.fill_(True)
        return self

    def forward(self, x):
        if not self.fitted:
            raise RuntimeError("Call fit on training features first")
        if not torch.isfinite(x).all():
            raise ValueError("Features must be finite")
        return (x - self.mean) / self.scale


def node_split(num_nodes, train=0.6, validation=0.2, seed=0):
    """Return disjoint CPU train/validation/test masks using a local RNG."""
    if num_nodes < 3 or train <= 0 or validation <= 0 or train + validation >= 1:
        raise ValueError("Require >=3 nodes and positive train/validation/test fractions")
    ntrain, nval = int(num_nodes * train), int(num_nodes * validation)
    if min(ntrain, nval, num_nodes - ntrain - nval) < 1:
        raise ValueError("Each split must contain at least one node")
    order = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(seed))
    masks = []
    for indices in (order[:ntrain], order[ntrain:ntrain+nval], order[ntrain+nval:]):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[indices] = True
        masks.append(mask)
    return tuple(masks)


def toy_graph():
    """Fixed tiny offline dataset; designed for smoke tests, not benchmarking."""
    from torch_geometric.data import Data
    x = torch.tensor([[-1., 0.], [-0.8, 0.2], [-1.2, -0.1],
                      [1., 0.], [0.8, 0.2], [1.2, -0.1]])
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3],
                               [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 3, 5]])
    return Data(x=x, edge_index=edge_index, y=torch.tensor([0, 0, 0, 1, 1, 1]))
