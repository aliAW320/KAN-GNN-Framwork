# Experiment report

## Completion

Candidates: 768; eligible: 720; datasets: 13; seeds: [0].

| Status | Model/dataset combinations |
|---|---:|
| completed | 12 |
| completed_undefined_metric | 1 |
| excluded_no_kan | 624 |
| pending | 9347 |

Pending rows are not measured results. Excluded rows were never trained.

## Results and paper context

All configurations, including pending/excluded ones: `all_models_datasets.csv`.
Paper-model joins: `paper_comparisons.csv`; reference values are published means/stds, not local runs.
The local dense atom features, graph edges, split IDs, budgets and target/unit aggregation are not established as identical to the paper. Deltas are contextual and do not prove reproduction or superiority. MUV has no comparable full-supervised result in the paper's tables 2/3.

## Completed rows

| Dataset | Model | Test score | Seeds completed |
|---|---|---:|---:|
| bace | pre-mlp__gcn__msg-kan__upd-kan__head-mlp__rbf | 0.585185 | 1 |
| bbbp | pre-mlp__gcn__msg-kan__upd-kan__head-mlp__rbf | 0.804498 | 1 |
| esol | pre-mlp__gcn__msg-kan__upd-kan__head-mlp__rbf | 2.05963 | 1 |
| freesolv | pre-mlp__gcn__msg-kan__upd-kan__head-mlp__rbf | 2.99731 | 1 |
| hiv | pre-mlp__gcn__msg-kan__upd-kan__head-mlp__rbf | 0.5 | 1 |
| lipophilicity | pre-mlp__gcn__msg-kan__upd-kan__head-mlp__rbf | 1.0697 | 1 |
| qm7 | pre-mlp__gcn__msg-kan__upd-kan__head-mlp__rbf | 194.384 | 1 |
| qm8 | pre-mlp__gcn__msg-kan__upd-kan__head-mlp__rbf | 0.0299056 | 1 |
| qm9 | pre-mlp__gcn__msg-kan__upd-kan__head-mlp__rbf | 19.3317 | 1 |
| sider | pre-mlp__gcn__msg-kan__upd-kan__head-mlp__rbf | 0.51676 | 1 |
| tox21 | pre-mlp__gcn__msg-kan__upd-kan__head-mlp__rbf | 0.43104 | 1 |
| toxcast | pre-mlp__gcn__msg-kan__upd-kan__head-mlp__rbf | 0.488889 | 1 |
