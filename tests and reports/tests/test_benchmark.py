import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from kangnn import seed_everything
from kangnn.transforms import KANLinear
from gnn_benchmark.datasets import load_dataset, molecule_graph, scaffold_split, schema
from gnn_benchmark.engine import statistics, train_one
from gnn_benchmark.grid import Candidate, build_model, enumerate_grid
from gnn_benchmark.metrics import binary_auc, masked_loss, score_predictions
from gnn_benchmark.paper import extract_paper


def graphs():
    result = []
    for i in range(12):
        x = torch.tensor([[float(i % 2), .2, .3], [.1, .4, .2]])
        result.append(Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]),
                           y=torch.tensor([[float(i % 2)]])))
    return result


def test_exact_grid_and_skips():
    grid = enumerate_grid()
    assert len(grid) == 768 and len({c.id for c in grid}) == 768
    assert sum(c.has_kan for c in grid) == 720
    assert sum(c.has_kan for c in enumerate_grid(True)) == 1488
    assert len(enumerate_grid(True)) == 1536
    with pytest.raises(ValueError):
        build_model(next(c for c in grid if not c.has_kan), 3, 1)


ELIGIBLE = [c for c in enumerate_grid(True) if c.has_kan]


@pytest.mark.parametrize("candidate", ELIGIBLE, ids=[c.id for c in ELIGIBLE])
def test_every_eligible_architecture_forward_backward(candidate):
    seed_everything(11)
    model = build_model(candidate, 3, 1, width=4, graph_layers=1)
    batch = Batch.from_data_list(graphs()[:2])
    output = model(batch)
    assert output.shape == (2, 1) and torch.isfinite(output).all()
    loss = masked_loss(output, batch.y, "classification")
    loss.backward()
    assert any(isinstance(m, KANLinear) for m in model.modules())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if candidate.preprocessing == "none":
        assert isinstance(model.preprocessing, torch.nn.Identity)
        assert model.config.hidden_channels == 3
    if candidate.preprocessing.startswith("kan"):
        assert sum(isinstance(m, KANLinear) for m in model.preprocessing.modules()) == int(candidate.preprocessing[3:])
    if candidate.decision.startswith("kan"):
        assert sum(isinstance(m, KANLinear) for m in model.decision.modules()) == int(candidate.decision[3:])


def test_metrics_ties_masks_and_raw_units():
    assert binary_auc([0, 1], [0., 1.]) == 1
    assert binary_auc([0, 1], [1., 1.]) == .5
    assert binary_auc([0, 1], [1., 0.]) == 0
    assert binary_auc([1, 1], [0., 1.]) is None
    p = torch.tensor([[0., 2.], [1., 4.]], requires_grad=True)
    y = torch.tensor([[float("nan"), 1.], [1., 5.]])
    loss = masked_loss(p, y, "regression")
    torch.testing.assert_close(loss, torch.tensor(2/3))
    loss.backward()
    assert p.grad[0, 0] == 0
    assert masked_loss(p, y*float("nan"), "regression") is None
    assert score_predictions(p, y, "regression")["score"] == .5


def test_scaffold_split_groups_disjoint_exhaustive():
    keys = [{"canonical": f"{i}", "scaffold": str(i//3)} for i in range(30)]
    first = scaffold_split(keys, 7)
    assert first == scaffold_split(keys, 7)
    assert sorted(sum((list(p) for p in first), [])) == list(range(30))
    groups = [{keys[i]["scaffold"] for i in part} for part in first]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    with pytest.raises(ValueError):
        scaffold_split(keys[:3])


def test_train_only_statistics_and_deterministic_training():
    dataset = graphs()
    split = (list(range(8)), [8, 9], [10, 11])
    stats = statistics(dataset, split[0])
    dataset[10].x += 100
    for a, b in zip(stats, statistics(dataset, split[0])):
        assert torch.equal(a, b)
    candidate = Candidate("mlp", "gcn", "kan", "kan", "mlp", "rbf")
    a = train_one(candidate, dataset, split, "classification", epochs=3, width=4, seed=7)
    b = train_one(candidate, dataset, split, "classification", epochs=3, width=4, seed=7)
    assert a["history"] == b["history"] and a["test"] == b["test"]
    assert a["best_epoch"] in (1, 2, 3) and a["parameters"] > 0
    for graph in dataset:
        graph.y = graph.y * 4 + 7
    assert train_one(candidate, dataset, split, "regression", epochs=2, width=4)["test"]["metric"] == "mae"


def test_csv_targets_duplicates_missing_and_invalid(tmp_path):
    path = tmp_path / "tiny.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "a", "b", "mol_id"])
        writer.writerows([["CC", "1", "", "a"], ["CCC", "0", "1", "b"],
                          ["c1ccccc1", "", "0", "c"], ["invalid", "1", "0", "d"]])
    dataset, keys, metadata = load_dataset("tox21", path)
    assert len(dataset) == 3 and len(keys) == 3 and len(metadata["invalid"]) == 1
    assert torch.isnan(dataset[0].y[0, 1])
    assert schema("qm8", ["smiles", "a", "b", "a"])[1] == [1, 2]
    graph, _ = molecule_graph("[He]")
    assert graph.edge_index.shape == (2, 0)
    assert molecule_graph("not-a-molecule")[0] is None


def test_paper_extraction_all_comparison_tables(tmp_path):
    paper = Path(__file__).parents[2] / "3711896.3736984.pdf"
    if not paper.exists():
        pytest.skip("Local paper PDF unavailable")
    rows = extract_paper(paper, tmp_path / "paper.csv")
    assert len(rows) == 192
    assert {r["table"] for r in rows} == {"1", "2", "3"}
    row = next(r for r in rows if r["table"] == "1" and r["paper_model"] == "GCN" and r["dataset"] == "bbbp")
    assert float(row["mean"]) == .622 and float(row["std"]) == .037


def test_report_keeps_pending_and_excluded(tmp_path):
    from gnn_benchmark.reporting import create_reports
    candidates = [enumerate_grid()[0].to_dict(), ELIGIBLE[0].to_dict()]
    (tmp_path / "plan.json").write_text(json.dumps({"datasets": ["bbbp"], "candidates": candidates,
        "seeds": [0], "classification": ["bbbp"], "settings": {"limit": None}}))
    summary = create_reports(tmp_path, tmp_path / "absent.csv")
    assert summary["model_dataset_status_counts"] == {"excluded_no_kan": 1, "pending": 1}
    assert "Pending rows are not measured" in (tmp_path / "REPORT.md").read_text()
    assert np.isfinite(torch.tensor(1.).numpy())


def test_best_checkpoint_includes_preprocessing_and_validation_selection(tmp_path):
    from gnn_benchmark.checkpoints import load_best
    from gnn_benchmark.engine import evaluate
    from torch_geometric.loader import DataLoader
    candidate = Candidate("kan2", "gcn", "mlp", "kan", "kan2", "tanh")
    dataset = graphs()
    split = (list(range(8)), [8, 9], [10, 11])
    path = tmp_path / "best.pt"
    result = train_one(candidate, dataset, split, "classification", epochs=3, width=4,
                       checkpoint_path=path, checkpoint_metadata={"dataset": "synthetic"})
    model, stats, metadata = load_best(path)
    assert metadata["dataset"] == "synthetic" and metadata["best_epoch"] == result["best_epoch"]
    assert metadata["selection_metric"] == "roc_auc"
    scores = [r["validation_score"] for r in result["history"]]
    assert metadata["validation"]["score"] == max(scores)
    observed = evaluate(model, DataLoader([dataset[i] for i in split[2]], batch_size=32),
                        "cpu", stats, "classification")
    assert observed == result["test"]
    # Test labels cannot select a different epoch or affect training.
    for i in split[2]:
        dataset[i].y = 1-dataset[i].y
    changed = train_one(candidate, dataset, split, "classification", epochs=3, width=4)
    assert changed["history"] == result["history"]
    assert changed["best_epoch"] == result["best_epoch"]


def test_reporting_completed_partial_failed_and_undefined(tmp_path):
    from gnn_benchmark.reporting import create_reports
    candidates = [c.to_dict() for c in ELIGIBLE[:4]]
    (tmp_path / "plan.json").write_text(json.dumps({"datasets": ["bbbp"], "candidates": candidates,
        "seeds": [0, 1], "classification": ["bbbp"], "settings": {"limit": None}}))
    for index, (scores, state) in enumerate([([.7, .9], "completed"), ([None, None], "completed"),
                                             ([.6], "completed"), ([None], "failed")]):
        for seed, score in enumerate(scores):
            folder = tmp_path / "results" / "bbbp" / candidates[index]["model_id"]
            folder.mkdir(parents=True, exist_ok=True)
            record = {"dataset": "bbbp", "model_id": candidates[index]["model_id"], "seed": seed,
                      "status": state, "test": {"score": score}, "parameters": 100, "seconds": 1.}
            (folder / f"seed-{seed}.json").write_text(json.dumps(record))
    paper = tmp_path / "paper.csv"
    paper.write_text("dataset,paper_model,table,metric,mean,std,status\nbbbp,GCN,2,roc_auc,0.6,0.01,reported\n")
    summary = create_reports(tmp_path, paper)
    assert summary["model_dataset_status_counts"] == {
        "completed": 1, "completed_undefined_metric": 1, "partial": 1, "failed": 1}
    with (tmp_path / "paper_comparisons.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["numeric_delta"]) == pytest.approx(.2)
    assert all(r["comparison_status"] == "context_only_protocol_not_matched" for r in rows)
