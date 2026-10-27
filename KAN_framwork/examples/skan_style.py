"""SKAN-style post-aggregation update and optional KAN graph classifier.

This is a generic dense-feature example, not the upstream molecular pipeline.
"""
import torch
from torch_geometric.data import Batch

from kangnn import (GraphKAN, LayerConfig, ModelConfig, TrainingConfig, TransformConfig,
                    evaluate, fit, seed_everything, skan_transform)
from kangnn.data import toy_graph


def main():
    seed_everything(7)
    graphs = []
    for label in (0, 1, 0, 1):
        graph = toy_graph()
        graph.x = graph.x + float(label)
        graph.y = torch.tensor([label], dtype=torch.long)
        graphs.append(graph)
    batch = Batch.from_data_list(graphs)
    update = skan_transform(num_basis=8, domain=(-2., 2.), trainable_centers=True,
                            trainable_widths=True)
    classifier = skan_transform(num_basis=4, domain=(-4., 8.))
    model = GraphKAN(ModelConfig(
        2, 8, 2, task="graph",
        layers=[LayerConfig(operator="gcn", message=TransformConfig(kind="identity"),
                            update=update)],
        decision=classifier,
    ))
    history = fit(model, batch, config=TrainingConfig(epochs=30))
    print(f"SKAN-style graph objective: {history[0]:.6f} -> {history[-1]:.6f}")
    print("Toy training-set evaluation:", evaluate(model, batch))


if __name__ == "__main__":
    main()
