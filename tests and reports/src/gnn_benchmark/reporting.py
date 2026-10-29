"""Complete expected-result matrix and clearly labeled paper-context joins."""
import csv
import json
from pathlib import Path

import numpy as np


def read_paper(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_reports(output, paper_path):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    references = read_paper(paper_path)
    reference_by_dataset = {}
    for row in references:
        if row["status"] == "reported" and row["table"] in {"2", "3"}:
            reference_by_dataset.setdefault(row["dataset"], []).append(row)
    records = {}
    for path in (output / "results").glob("*/*/seed-*.json"):
        record = json.loads(path.read_text())
        records[(record["dataset"], record["model_id"], record["seed"])] = record
    rows, comparisons = [], []
    for dataset in plan["datasets"]:
        for candidate in plan["candidates"]:
            completed = []
            failures = []
            for seed in plan["seeds"]:
                record = records.get((dataset, candidate["model_id"], seed))
                if record and record["status"] == "completed":
                    completed.append(record)
                elif record:
                    failures.append(record)
            scores = [r["test"]["score"] for r in completed if r["test"]["score"] is not None]
            status = "excluded_no_kan" if not candidate["has_kan"] else "pending"
            if candidate["has_kan"]:
                if len(completed) == len(plan["seeds"]):
                    status = "completed" if len(scores) == len(completed) else "completed_undefined_metric"
                elif completed:
                    status = "partial"
                elif failures:
                    status = "failed"
            metric = "roc_auc" if dataset in plan["classification"] else "mae"
            mean = float(np.mean(scores)) if scores else None
            std = float(np.std(scores, ddof=1)) if len(scores) > 1 else None
            paper = reference_by_dataset.get(dataset, [])
            best = (max(paper, key=lambda r: float(r["mean"])) if metric == "roc_auc"
                    else min(paper, key=lambda r: float(r["mean"]))) if paper else None
            row = {"dataset": dataset, **candidate, "metric": metric, "status": status,
                   "runs_completed": len(completed), "runs_failed": len(failures),
                   "runs_expected": len(plan["seeds"]), "test_mean": mean, "test_std": std,
                   "parameters": completed[0]["parameters"] if completed else None,
                   "mean_seconds": float(np.mean([r["seconds"] for r in completed])) if completed else None,
                   "paper_best_model": best["paper_model"] if best else None,
                   "paper_best_mean": float(best["mean"]) if best else None,
                   "comparison_status": "context_only_protocol_not_matched" if best else "no_supervised_paper_reference",
                   "limited_dataset": plan["settings"]["limit"] is not None}
            rows.append(row)
            for ref in paper:
                comparisons.append({"dataset": dataset, "model_id": candidate["model_id"],
                                    "local_status": status, "metric": metric,
                                    "local_mean": mean, "local_std": std,
                                    "paper_model": ref["paper_model"], "paper_mean": ref["mean"],
                                    "paper_std": ref["std"], "paper_table": ref["table"],
                                    "numeric_delta": mean-float(ref["mean"]) if mean is not None else None,
                                    "comparison_status": "context_only_protocol_not_matched"})
    write_csv(output / "all_models_datasets.csv", rows, list(rows[0]))
    write_csv(output / "paper_comparisons.csv", comparisons, list(comparisons[0]) if comparisons
              else ["dataset", "model_id", "comparison_status"])
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {"model_dataset_status_counts": counts, "run_files": len(records),
               "candidate_models": len(plan["candidates"]),
               "eligible_models": sum(c["has_kan"] for c in plan["candidates"]),
               "datasets": len(plan["datasets"]), "seeds": plan["seeds"],
               "settings": plan["settings"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Experiment report", "", "## Completion", "",
             f"Candidates: {len(plan['candidates'])}; eligible: {summary['eligible_models']}; "
             f"datasets: {len(plan['datasets'])}; seeds: {plan['seeds']}.", "",
             "| Status | Model/dataset combinations |", "|---|---:|"]
    lines += [f"| {key} | {value} |" for key, value in sorted(counts.items())]
    lines += ["", "Pending rows are not measured results. Excluded rows were never trained.", "",
              "## Results and paper context", "",
              "All configurations, including pending/excluded ones: `all_models_datasets.csv`.",
              "Paper-model joins: `paper_comparisons.csv`; reference values are published means/stds, not local runs.",
              "The local dense atom features, graph edges, split IDs, budgets and target/unit aggregation "
              "are not established as identical to the paper. Deltas are contextual and do not prove reproduction "
              "or superiority. MUV has no comparable full-supervised result in the paper's tables 2/3.", "",
              "## Completed rows", "",
              "| Dataset | Model | Test score | Seeds completed |", "|---|---|---:|---:|"]
    for row in rows:
        if row["test_mean"] is not None:
            lines.append(f"| {row['dataset']} | {row['model_id']} | {row['test_mean']:.6g} | {row['runs_completed']} |")
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")
    return summary
