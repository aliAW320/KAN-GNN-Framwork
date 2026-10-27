"""PyG MessagePassing operators with independently replaceable transforms."""
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from .transforms import TransformConfig, make_transform


def identity():
    return TransformConfig(kind="identity")


@dataclass
class LayerConfig:
    operator: str = "gcn"
    message: TransformConfig = field(default_factory=lambda: TransformConfig(bias=False))
    update: TransformConfig = field(default_factory=identity)
    attention: TransformConfig = field(default_factory=lambda: TransformConfig(bias=False))
    heads: int = 1
    dropout: float = 0.0
    self_loops: bool = True
    residual: bool = False
    edge_dim: int | None = None
    edge: TransformConfig = field(default_factory=TransformConfig)

    def __post_init__(self):
        if self.operator not in {"gcn", "sage", "gat", "gatv2"}:
            raise ValueError("operator must be gcn, sage, gat, or gatv2")
        if self.heads < 1 or not 0 <= self.dropout < 1:
            raise ValueError("heads must be positive and dropout in [0, 1)")
        if self.operator not in {"gat", "gatv2"} and self.heads != 1:
            raise ValueError("Multiple heads are only supported for attention operators")
        if self.edge_dim is not None and self.edge_dim < 1:
            raise ValueError("edge_dim must be positive")
        for key in ("message", "update", "attention", "edge"):
            if isinstance(getattr(self, key), dict):
                setattr(self, key, TransformConfig(**getattr(self, key)))


class GraphLayer(MessagePassing):
    """Homogeneous COO graphs, source-to-target edges, mean-combined attention heads.

    GCN normalizes weighted adjacency including self loops. SAGE takes neighbor
    means and adds a separately transformed root. GAT uses additive scalar
    source/target logits; GATv2 scores LeakyReLU(target + source) vectors.
    Optional edge embeddings are added to source messages (zero for new loops).
    """

    def __init__(self, in_channels, out_channels, config=None):
        cfg = config or LayerConfig()
        super().__init__(aggr="mean" if cfg.operator == "sage" else "add", node_dim=0)
        self.config, self.out_channels = cfg, out_channels
        heads = cfg.heads
        self.source = make_transform(in_channels, heads * out_channels, cfg.message)
        self.post = make_transform(out_channels, out_channels, cfg.update)
        self.root = (make_transform(in_channels, out_channels, cfg.message)
                     if cfg.operator == "sage" else None)
        self.target = (make_transform(in_channels, heads * out_channels, cfg.message)
                       if cfg.operator == "gatv2" else None)
        if cfg.operator in {"gat", "gatv2"}:
            self.att_source = nn.ModuleList(make_transform(out_channels, 1, cfg.attention)
                                            for _ in range(heads))
            self.att_target = (nn.ModuleList(make_transform(out_channels, 1, cfg.attention)
                                             for _ in range(heads))
                               if cfg.operator == "gat" else None)
        self.edge_encoder = (make_transform(cfg.edge_dim, heads * out_channels, cfg.edge)
                             if cfg.edge_dim is not None else None)
        self.skip = nn.Linear(in_channels, out_channels, bias=False) if cfg.residual else None
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        cfg, n = self.config, x.size(0)
        if x.ndim != 2 or not x.is_floating_point():
            raise ValueError("x must be floating point [num_nodes, num_features]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError("edge_index must be int64 [2, num_edges]")
        if edge_index.device != x.device:
            raise ValueError("Features and edge_index must be on the same device")
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= n):
            raise ValueError("edge_index references a nonexistent node")
        if edge_weight is not None:
            if cfg.operator != "gcn":
                raise ValueError("edge_weight is supported only by GCN")
            if edge_weight.shape != (edge_index.size(1),) or not torch.isfinite(edge_weight).all():
                raise ValueError("edge_weight must be finite [num_edges]")
            if (edge_weight < 0).any():
                raise ValueError("GCN requires nonnegative edge weights")
        if (edge_attr is None) != (self.edge_encoder is None):
            raise ValueError("Provide edge_attr exactly when edge_dim is configured")
        if edge_attr is not None:
            if edge_attr.shape != (edge_index.size(1), cfg.edge_dim):
                raise ValueError("edge_attr shape must be [num_edges, edge_dim]")
            if cfg.operator == "gcn" and cfg.self_loops:
                raise ValueError("GCN edge features require self_loops=False; add loops explicitly")
        weight = edge_weight
        if cfg.operator == "gcn":
            edge_index, weight = gcn_norm(edge_index, edge_weight, n, False,
                                          cfg.self_loops, "source_to_target", x.dtype)
        elif cfg.operator in {"gat", "gatv2"} and cfg.self_loops:
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0., num_nodes=n)
        # SAGE's root path already includes the central node: do not add loops.
        source = self.source(x).view(n, cfg.heads, self.out_channels)
        target = self.target(x).view_as(source) if self.target is not None else source
        edge_embedding = (self.edge_encoder(edge_attr).view(-1, cfg.heads, self.out_channels)
                          if self.edge_encoder is not None else None)
        out = self.propagate(edge_index, x=(source, target), weight=weight,
                             edge_embedding=edge_embedding, size=(n, n))
        out = out.mean(dim=1)
        if self.root is not None:
            out = out + self.root(x)
        out = self.post(out + self.bias)
        return out if self.skip is None else out + self.skip(x)

    def message(self, x_j, x_i, index, ptr, size_i, weight, edge_embedding):
        cfg = self.config
        values = x_j if edge_embedding is None else x_j + edge_embedding
        if cfg.operator in {"gat", "gatv2"}:
            if cfg.operator == "gat":
                logits = F.leaky_relu(self._score(self.att_source, values) + self._score(self.att_target, x_i), 0.2)
            else:
                logits = self._score(self.att_source, F.leaky_relu(values + x_i, 0.2))
            alpha = softmax(logits, index, ptr, size_i)
            return values * F.dropout(alpha, cfg.dropout, self.training)
        return values if weight is None else values * weight.view(-1, 1, 1)

    @staticmethod
    def _score(scorers, x):
        return torch.stack([scorer(x[:, h]) for h, scorer in enumerate(scorers)], dim=1)
