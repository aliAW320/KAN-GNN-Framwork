"""Plan, prepare, train/resume, and report the complete local model grid."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import torch

from .datasets import CLASSIFICATION, FEATURE_VERSION, discover, load_dataset, scaffold_split, schema
from .engine import train_one
from .grid import enumerate_grid
from .paper import extract_paper
from .reporting import create_reports

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def inventory(files):
    rows = []
    for name, path in files.items():
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            _, targets = schema(name, header)
            count = sum(bool(row) for row in reader)
        rows.append({"dataset": name, "rows": count, "tasks": len(targets),
                     "target_names": [header[i] for i in targets], "file": str(path),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows


def prepare_dataset(name, path, cache, limit):
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    from rdkit import rdBase
    from . import datasets as data_module
    loader_hash = hashlib.sha256(Path(data_module.__file__).read_bytes()).hexdigest()
    signature = hashlib.sha256(f"{source_hash}:{loader_hash}:{FEATURE_VERSION}:{limit}:{rdBase.rdkitVersion}".encode()).hexdigest()[:16]
    target = Path(cache) / f"{name}-{signature}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # Internal cache generated from CSV by this runner, never an external model file.
        return torch.load(target, weights_only=False)
    payload = load_dataset(name, path, limit)
    temporary = target.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "prepare", "run", "report"])
    parser.add_argument("--data", type=Path, default=REPO / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "gpu_200_epoch_plan")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--paper", type=Path, default=REPO / "3711896.3736984.pdf")
    parser.add_argument("--independent", dest="independent", action="store_true", default=True,
                        help="Default: independent message/update, 1536 candidates")
    parser.add_argument("--tied", dest="independent", action="store_false",
                        help="Alternative original 768-candidate tied grid")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--limit", type=int, help="First N valid molecules; diagnostics only, never a full result")
    parser.add_argument("--datasets", nargs="+", help="Subset to execute; plan still records all discovered datasets")
    parser.add_argument("--model-ids", nargs="+", help="Execute selected stable model IDs only")
    parser.add_argument("--max-runs", type=int, help="Bound work in this invocation; completed work is resumable")
    parser.add_argument("--device", default="cuda", help="Training device; default CUDA (no silent CPU fallback)")
    parser.add_argument("--allow-nondeterministic", action="store_true",
                        help="Explicitly permit nondeterministic GPU graph kernels")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.width, args.graph_layers, args.batch_size, args.patience) < 1 or args.lr <= 0:
        parser.error("Training dimensions and budgets must be positive")
    if args.limit is not None and args.limit < 3:
        parser.error("--limit must be at least 3")
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seed list must not contain duplicates")
    args.output.mkdir(parents=True, exist_ok=True)
    paper_path = ROOT / "reference" / "paper_results.csv"
    if args.command == "report":
        print(json.dumps(create_reports(args.output, paper_path), indent=2))
        return
    files = discover(args.data)
    if args.datasets and not set(args.datasets) <= set(files):
        parser.error("Unknown dataset name")
    candidates = enumerate_grid(args.independent)
    if args.model_ids and not set(args.model_ids) <= {c.id for c in candidates if c.has_kan}:
        parser.error("Unknown or all-MLP model ID")
    settings = {key: getattr(args, key) for key in ("independent", "epochs", "seeds", "width",
                "graph_layers", "batch_size", "lr", "patience", "limit", "device", "allow_nondeterministic")}
    settings["features"] = FEATURE_VERSION
    runtime_environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                           "cuda_runtime": torch.version.cuda}
    import torch_geometric
    from rdkit import rdBase
    runtime_environment.update(pyg=torch_geometric.__version__, rdkit=rdBase.rdkitVersion)
    from kangnn import model as core_module
    source_files = sorted(Path(__file__).parent.glob("*.py")) + sorted(Path(core_module.__file__).parent.glob("*.py"))
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in source_files)).hexdigest()
    if args.command == "run" and args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    if args.command == "run":
        runtime_environment["device"] = args.device
        if args.device.startswith("cuda"):
            runtime_environment["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
        env_path = args.output / "training_environment.json"
        if env_path.exists() and json.loads(env_path.read_text()) != runtime_environment:
            parser.error("Training environment changed; use a different --output")
        atomic_json(env_path, runtime_environment)
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if old["settings"] != settings or old["code_sha256"] != code_hash:
            parser.error("Output already belongs to another protocol; use a different --output")
        plan = old
        if {r["dataset"]: r["sha256"] for r in plan["inventory"]} != {
                name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}:
            parser.error("Dataset bytes changed; use a new output directory")
    else:
        plan = {"settings": settings, "seeds": args.seeds, "classification": sorted(CLASSIFICATION),
                "datasets": list(files), "inventory": inventory(files),
                "candidates": [c.to_dict() for c in candidates],
                "planning_environment": runtime_environment,
                "code_sha256": code_hash}
        atomic_json(plan_path, plan)
    if not paper_path.exists():
        extract_paper(args.paper, paper_path)
    create_reports(args.output, paper_path)
    if args.command == "plan":
        eligible = sum(c.has_kan for c in candidates)
        print(json.dumps({"candidates": len(candidates), "eligible": eligible,
                          "excluded": len(candidates)-eligible, "datasets": len(files),
                          "training_runs": eligible*len(files)*len(args.seeds)}, indent=2))
        return
    runs = 0
    for name, path in files.items():
        if args.datasets and name not in args.datasets:
            continue
        print(f"Preparing {name}", flush=True)
        graphs, keys, metadata = prepare_dataset(name, path, args.cache, args.limit)
        metadata["rdkit_version"] = rdBase.rdkitVersion
        atomic_json(args.output / "dataset_metadata" / f"{name}.json", metadata)
        for seed in args.seeds:
            splits = scaffold_split(keys, seed)
            atomic_json(args.output / "splits" / f"{name}-{seed}.json",
                        {"train": splits[0], "validation": splits[1], "test": splits[2],
                         "dataset_sha256": metadata["csv_sha256"], "limit": args.limit})
            if args.command == "prepare":
                continue
            for candidate in candidates:
                if not candidate.has_kan or (args.model_ids and candidate.id not in args.model_ids):
                    continue
                result_path = args.output / "results" / name / candidate.id / f"seed-{seed}.json"
                if result_path.exists():
                    previous = json.loads(result_path.read_text())
                    if previous["status"] == "completed" or not args.retry_failed:
                        continue
                record = {"dataset": name, "model_id": candidate.id, "seed": seed,
                          "candidate": candidate.to_dict(), "dataset_sha256": metadata["csv_sha256"],
                          "runtime_environment": runtime_environment}
                start = time.perf_counter()
                try:
                    result = train_one(candidate, graphs, splits, metadata["kind"], seed=seed,
                                       epochs=args.epochs, width=args.width, graph_layers=args.graph_layers,
                                       batch_size=args.batch_size, lr=args.lr, device=args.device,
                                       patience=args.patience,
                                       deterministic=not args.allow_nondeterministic,
                                       checkpoint_path=result_path.with_name(f"seed-{seed}-best.pt"),
                                       checkpoint_metadata={"dataset": name,
                                           "dataset_sha256": metadata["csv_sha256"],
                                           "target_columns": metadata["target_columns"],
                                           "features": FEATURE_VERSION,
                                           "protocol": settings, "code_sha256": code_hash,
                                           "runtime_environment": runtime_environment})
                    record.update(status="completed", **result)
                except Exception as exc:
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                                  traceback=traceback.format_exc(), seconds=time.perf_counter()-start)
                atomic_json(result_path, record)
                runs += 1
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                if runs % 25 == 0:
                    create_reports(args.output, paper_path)
                print(f"{name}: {candidate.id}: {record['status']}", flush=True)
                if args.max_runs is not None and runs >= args.max_runs:
                    create_reports(args.output, paper_path)
                    return
        del graphs, keys
    create_reports(args.output, paper_path)


if __name__ == "__main__":
    main()
