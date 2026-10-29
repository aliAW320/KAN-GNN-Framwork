import json
import sys

import pytest

from gnn_benchmark import cli


def tiny_data(tmp_path):
    folder = tmp_path / "data" / "bbbp" / "raw"
    folder.mkdir(parents=True)
    (folder / "small.csv").write_text(
        "smiles,p_np\nCC,1\nc1ccccc1,0\nc1ccncc1,1\nC1CCCCC1,0\nc1ccoc1,1\nC1CCNC1,0\n")
    return tmp_path / "data"


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["benchmark", *map(str, args)])
    cli.main()


def test_gpu_plan_defaults_and_report_without_training(tmp_path, monkeypatch, capsys):
    data = tiny_data(tmp_path)
    output = tmp_path / "report"
    invoke(monkeypatch, "plan", "--data", data, "--output", output)
    assert json.loads(capsys.readouterr().out)["eligible"] == 1488
    plan = json.loads((output / "plan.json").read_text())
    assert plan["settings"]["epochs"] == 200 and plan["settings"]["width"] == 256
    assert plan["settings"]["seeds"] == [0, 1, 2, 3, 4]
    assert plan["settings"]["device"] == "cuda"
    assert not (output / "results").exists()
    invoke(monkeypatch, "report", "--output", output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["model_dataset_status_counts"] == {"excluded_no_kan": 48, "pending": 1488}
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, "run", "--data", data, "--output", output)
    assert "CUDA requested but not available" in capsys.readouterr().err
    assert not (output / "results").exists()


def test_prepare_run_resume_fail_retry_and_provenance_guards(tmp_path, monkeypatch):
    data = tiny_data(tmp_path)
    output = tmp_path / "output"
    common = ["--data", data, "--output", output, "--cache", tmp_path / "cache",
              "--epochs", "1", "--seeds", "0", "--width", "4", "--batch-size", "2",
              "--graph-layers", "1", "--patience", "1", "--device", "cpu", "--datasets", "bbbp",
              "--model-ids", "pre-mlp__gcn__msg-kan__upd-kan__head-mlp__rbf", "--max-runs", "1"]
    invoke(monkeypatch, "prepare", *common)
    assert (output / "splits" / "bbbp-0.json").exists()
    original = cli.train_one
    def failing(*args, **kwargs):
        raise ValueError("injected test failure")
    monkeypatch.setattr(cli, "train_one", failing)
    invoke(monkeypatch, "run", *common)
    result_path = next((output / "results").glob("*/*/seed-0.json"))
    assert json.loads(result_path.read_text())["status"] == "failed"
    invoke(monkeypatch, "run", *common)
    assert json.loads(result_path.read_text())["status"] == "failed"
    monkeypatch.setattr(cli, "train_one", original)
    invoke(monkeypatch, "run", *common, "--retry-failed")
    completed = json.loads(result_path.read_text())
    assert completed["status"] == "completed"
    assert result_path.with_name("seed-0-best.pt").exists()
    invoke(monkeypatch, "run", *common)
    assert json.loads(result_path.read_text()) == completed
    with pytest.raises(SystemExit):
        invoke(monkeypatch, "plan", *common, "--epochs", "2")
    environment = output / "training_environment.json"
    previous = environment.read_text()
    environment.write_text(json.dumps({"python": "changed"}))
    with pytest.raises(SystemExit):
        invoke(monkeypatch, "run", *common)
    environment.write_text(previous)
    with (data / "bbbp" / "raw" / "small.csv").open("a") as handle:
        handle.write("CCCC,0\n")
    with pytest.raises(SystemExit):
        invoke(monkeypatch, "plan", *common)


@pytest.mark.parametrize("options", [["--epochs", "0"], ["--limit", "2"], ["--max-runs", "0"],
                                      ["--seeds", "0", "0"], ["--datasets", "missing"],
                                      ["--model-ids", "missing"]])
def test_invalid_cli_options(options, tmp_path, monkeypatch):
    data = tiny_data(tmp_path)
    with pytest.raises(SystemExit):
        invoke(monkeypatch, "plan", "--data", data, "--output", tmp_path / "out", *options)
