"""JSON configuration runner, with an offline node-classification demo."""
import argparse
import json

from .checkpoint import save_model
from .data import toy_graph
from .graph import LayerConfig
from .model import GraphKAN, ModelConfig
from .training import TrainingConfig, evaluate, fit, seed_everything
from .transforms import TransformConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON file containing a ModelConfig")
    parser.add_argument("--operator", choices=["gcn", "sage", "gat", "gatv2"], default="gcn")
    parser.add_argument("--kind", choices=["mlp", "kan"], default="kan")
    parser.add_argument("--basis", choices=["bspline", "fourier", "relu", "rbf"], default="rbf")
    parser.add_argument("--loss", choices=["cross_entropy", "nll"], default="cross_entropy")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", help="Optional model checkpoint path")
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.config:
        with open(args.config, encoding="utf-8") as source:
            config = ModelConfig(**json.load(source))
    else:
        transform = TransformConfig(kind=args.kind, basis=args.basis,
                                    basis_kwargs={"num_basis": 9 if args.basis == "fourier" else 8})
        config = ModelConfig(2, 8, 2, preprocessing=transform, decision=transform,
                             layers=[LayerConfig(operator=args.operator, message=transform)])
    if config.task != "node" or config.in_channels != 2 or config.out_channels != 2:
        parser.error("Offline demo requires task=node, in_channels=2, out_channels=2; use Python for other data")
    model, data = GraphKAN(config), toy_graph()
    history = fit(model, data, args.loss, TrainingConfig(epochs=args.epochs))
    result = {"initial_loss": history[0], "final_training_loss": history[-1],
              "evaluation": evaluate(model, data, args.loss), "seed": args.seed}
    if args.save:
        save_model(model, args.save)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
