"""Run after installing the package: python examples/custom_extensions.py."""
import torch
from torch import nn

from kangnn import (GraphKAN, LayerConfig, ModelConfig, TrainingConfig, TransformConfig,
                    fit, register_basis, register_loss, seed_everything)
from kangnn.data import toy_graph


class PolynomialBasis(nn.Module):
    def __init__(self, num_basis=4):
        super().__init__()
        self.num_basis = num_basis
        self.register_buffer("powers", torch.arange(num_basis))

    def forward(self, x):
        return x.unsqueeze(-1) ** self.powers


class CustomLoss(nn.Module):
    def forward(self, predictions, targets):
        return nn.functional.cross_entropy(predictions, targets, label_smoothing=0.05)


def main():
    register_basis("polynomial", PolynomialBasis)
    register_loss("custom_ce", CustomLoss)
    seed_everything(7)
    kan = TransformConfig(kind="kan", basis="polynomial")
    model = GraphKAN(ModelConfig(2, 8, 2, preprocessing=kan, decision=kan,
                                layers=[LayerConfig(operator="sage", message=kan, update=kan)]))
    history = fit(model, toy_graph(), "custom_ce", TrainingConfig(epochs=20))
    print(f"Custom basis/loss: {history[0]:.6f} -> {history[-1]:.6f}")


if __name__ == "__main__":
    main()
