from kangnn import (GraphKAN, LayerConfig, ModelConfig, TrainingConfig, TransformConfig,
                    fit, seed_everything)
from kangnn.data import toy_graph

from kan_transforms import register_kan_bases


def main():
    register_kan_bases()
    seed_everything(7)
    laplace = TransformConfig(kind="kan", basis="laplace", basis_kwargs={"num_basis": 8})
    z = TransformConfig(kind="kan", basis="z_transform", basis_kwargs={"num_basis": 8})
    model = GraphKAN(ModelConfig(2, 8, 2, preprocessing=laplace, decision=z,
                                layers=[LayerConfig(operator="gatv2", message=z, update=laplace)]))
    history = fit(model, toy_graph(), config=TrainingConfig(epochs=30))
    print(f"Laplace/Z KAN training: {history[0]:.6f} -> {history[-1]:.6f}")


if __name__ == "__main__":
    main()
