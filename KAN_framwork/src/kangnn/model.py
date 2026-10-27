"""End-to-end node, graph, and learned link prediction pipelines."""
from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from .graph import GraphLayer, LayerConfig, identity
from .transforms import TransformConfig, make_transform


@dataclass
class ModelConfig:
    in_channels: int
    hidden_channels: int
    out_channels: int
    layers: list[LayerConfig] = field(default_factory=lambda: [LayerConfig(), LayerConfig()])
    preprocessing: TransformConfig = field(default_factory=TransformConfig)
    readout: TransformConfig = field(default_factory=identity)
    decision: TransformConfig = field(default_factory=TransformConfig)
    task: str = "node"
    pooling: str = "mean"
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.in_channels, self.hidden_channels, self.out_channels) < 1:
            raise ValueError("Channel counts must be positive")
        if self.task not in {"node", "graph", "link"}:
            raise ValueError("task must be node, graph, or link")
        if self.pooling not in {"mean", "sum", "max"}:
            raise ValueError("pooling must be mean, sum, or max")
        if not self.layers or not 0 <= self.dropout < 1:
            raise ValueError("Require at least one layer and dropout in [0, 1)")
        self.layers = [LayerConfig(**c) if isinstance(c, dict) else c for c in self.layers]
        for key in ("preprocessing", "readout", "decision"):
            if isinstance(getattr(self, key), dict):
                setattr(self, key, TransformConfig(**getattr(self, key)))

    def to_dict(self):
        return asdict(self)


class GraphKAN(nn.Module):
    """Accepts PyG Data/Batch. Returns raw logits or regression scores.

    For link tasks, edge_label_index holds query pairs and is separate from
    message-passing edge_index. The ordered concatenation decoder is directed.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.hidden_channels
        self.preprocessing = make_transform(config.in_channels, width, config.preprocessing)
        self.layers = nn.ModuleList(GraphLayer(width, width, c) for c in config.layers)
        self.readout = make_transform(width, width, config.readout)
        self.decision = make_transform(width * (2 if config.task == "link" else 1),
                                       config.out_channels, config.decision)

    def encode(self, data):
        x = self.preprocessing(data.x)
        for layer in self.layers:
            x = layer(x, data.edge_index, getattr(data, "edge_weight", None),
                      getattr(data, "edge_attr", None))
            x = F.dropout(F.relu(x), self.config.dropout, self.training)
        return x

    def forward(self, data, edge_label_index=None):
        x = self.encode(data)
        if self.config.task == "graph":
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            pool = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}
            x = pool[self.config.pooling](x, batch)
        x = self.readout(x)
        if self.config.task == "link":
            pairs = edge_label_index if edge_label_index is not None else getattr(
                data, "edge_label_index", None)
            if pairs is None:
                raise ValueError("Link prediction requires edge_label_index")
            if pairs.ndim != 2 or pairs.shape[0] != 2 or pairs.dtype != torch.long:
                raise ValueError("edge_label_index must be int64 [2, num_queries]")
            if pairs.numel() and (pairs.min() < 0 or pairs.max() >= x.size(0)):
                raise ValueError("Query references a nonexistent node")
            x = torch.cat((x[pairs[0]], x[pairs[1]]), -1)
        return self.decision(x)
