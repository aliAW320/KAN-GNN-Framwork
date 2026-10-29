# GNN architecture sweep: tests and reports

A separate experiment project for the requested KAN/MLP ablations on **every local
molecular dataset**. All new source, tests, reference data and reports live here;
it imports the existing `kangnn` framework without changing it.

The main experiment is configured for your **RTX 4060 Ti**, **200 epochs** and
**validation-selected best checkpoints**. The full benchmark has **not** been
trained on this CPU machine. The local measured results are explicitly labeled
small pipeline diagnostics, not final paper comparisons.

## Main outputs

- [GPU experiment matrix and report](reports/gpu_200_epoch_plan/REPORT.md)
- [All model/dataset rows](reports/gpu_200_epoch_plan/all_models_datasets.csv)
- [Merged paper comparisons](reports/gpu_200_epoch_plan/paper_comparisons.csv)
- [Machine-readable summary](reports/gpu_200_epoch_plan/summary.json)
- [Extracted paper results](reference/paper_results.csv)
- [Dataset audit and implementation details](docs/IMPLEMENTATION.md)
- [Future steps](docs/FUTURE_STEPS.md)

Reports include **pending**, **partial**, **completed**, **undefined metric**,
**failed**, and **excluded-no-KAN** states. Pending scores remain empty: published
paper scores are never substituted for local model measurements.

## 1. Exact architecture grid

You clarified that message and update types are **independent**:

| Factor | Choices |
|---|---|
| Preprocessing: 4 | None; one linear MLP layer; 2 KAN layers; 4 KAN layers |
| Graph operator: 4 | GCN, GAT, GATv2, mean GraphSAGE |
| Message transform: 2 | One MLP layer or one KAN layer |
| Post-aggregation update: 2 | One MLP layer or one KAN layer |
| Decision head: 4 | One MLP layer; 1, 2, or 4 KAN layers |
| KAN family: 6 | Fourier, ReLU ramps, B-spline, RBF, sigmoid, tanh |

`4 × 4 × 2 × 2 × 4 × 6 = 1,536` candidates.

**48 all-MLP candidates are excluded**, leaving **1,488 models per dataset**.
Across 13 datasets that is **19,344 training runs per seed**. The default five
seeds produce **96,720 runs**, following the paper's use of five independent
runs for mean/std reporting. Use `--seeds 0` for a single-seed initial sweep.

The old tied interpretation is also available with `--tied`: 768 candidates,
720 eligible. This is an alternative experiment, not the main requested grid.

### How stages are interpreted

- “Aggregation with KAN/MLP” means a configurable transform **after** the
  operator's invariant reduction. The reduction itself stays normalized sum,
  attention-weighted sum or mean; replacing it with an order-dependent dense map
  would change the graph semantics.
- All KAN stages in one candidate use the same selected basis family.
- Each candidate uses 2 graph layers by default, configurable separately from
  the number of KAN layers inside preprocessing/decision stages.
- Depth counts are exact KANLinear counts. `kan4` means four KAN layers and three
  intermediate widths, not four extra layers after a projection.
- MLP depth was not specified in the request; its stage depth is one linear layer.
  The enclosing model still applies ReLU after each graph layer.
- No preprocessing is **true identity**. Encoder width then remains the 9 input
  features; no hidden learned projection is inserted. Other preprocessing modes
  project to `--width`, default 256. Parameter counts consequently differ.
- Attention score functions remain ordinary bias-free linear maps. The grid's
  KAN/MLP choices apply to message and update, not another uncounted attention axis.
- Sigmoid and tanh are learned shifted/scaled basis expansions, not a replacement
  of the entire KAN by a single sigmoid/tanh activation.

## 2. Install on the GPU machine

Copy the repository with `data/`, `KAN_framwork/`, and this directory. Install a
CUDA-enabled PyTorch build appropriate to that machine. Then, from the repository
root and with its environment activated:

```bash
python -m pip install -e './KAN_framwork[test]'
python -m pip install -e './tests and reports[test]'
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Do not install the CPU-only PyTorch wheel on the GPU machine. The new package
installs RDKit for graph construction. No pandas or scikit-learn dependency is
required. The `pdftotext` system executable is needed only if regenerating paper
reference values; the extracted CSV is already provided.

## 3. Plan, prepare, run, and resume

Run commands from the repository root. All paths containing spaces are quoted.

```bash
# Inspect the full grid without training or requiring a CUDA device.
python -m gnn_benchmark.cli plan

# Convert all supplied CSVs to graph caches and write scaffold split IDs.
python -m gnn_benchmark.cli prepare

# Train on CUDA: independent choices, 200 epochs, five seeds, best checkpoints.
python -m gnn_benchmark.cli run --device cuda

# Run one complete model/seed job to check VRAM usage and estimate throughput.
python -m gnn_benchmark.cli run --device cuda --max-runs 1

# Later, repeat the same command without a bound: completed runs are skipped.
python -m gnn_benchmark.cli run --device cuda

# Refresh reports at any point, without loading datasets or training.
python -m gnn_benchmark.cli report
```

The optional Linux helper is `bash 'tests and reports/run_gpu.sh'` after activating
the target environment. Python commands also work on other supported platforms.
CUDA is the training default and an unavailable CUDA device is an explicit error;
there is no silent CPU training fallback.

**Epoch budget:** default `epochs=200` and `patience=200` run all 200 epochs unless
a run fails. Smaller explicit patience enables early stopping. A best checkpoint
is replaced atomically whenever validation improves, so it survives interruption.
Ties keep the earlier best epoch. Test scores never select a checkpoint.

Best selection uses maximum validation macro ROC-AUC for classification and
minimum validation raw-unit macro MAE for regression. If validation AUC cannot be
defined, selection falls back to validation loss and records that fallback.

### Smaller, distinct experiments

Use another output directory when changing epochs, seeds, width, batch size,
determinism or other protocol settings. This prevents results being mixed.

```bash
# One-seed sweep on the GPU.
python -m gnn_benchmark.cli run --device cuda --seeds 0 \
  --output 'tests and reports/reports/gpu_one_seed'

# Smaller batches if your particular RTX 4060 Ti configuration needs them.
python -m gnn_benchmark.cli run --device cuda --batch-size 16 \
  --output 'tests and reports/reports/gpu_batch16'

# Run only selected datasets; the matrix still records all discovered datasets.
python -m gnn_benchmark.cli run --device cuda --datasets bbbp esol --max-runs 2
```

`--model-ids` selects stable IDs from `plan.json`, for example:
`pre-mlp__gcn__msg-kan__upd-kan__head-mlp__rbf`.
`--max-runs` bounds newly attempted jobs in that invocation, including failures;
it does not change the experiment definition.

A completed job is resumable at job boundaries. An interrupted unfinished job
restarts training from epoch one on the next invocation; best-so-far weights are
preserved, but optimizer/RNG mid-epoch resume is not implemented. Failed jobs are
skipped unless `--retry-failed` is supplied. Source/data/protocol changes require
a new output directory; training-library/device changes are also checked against
`training_environment.json` once actual training has begun.

## 4. Determinism and GPU behavior

Tests are deterministic CPU checks. Real training seeds Python, NumPy, PyTorch,
the training-loader shuffle generator and scaffold-group ordering. Strict
PyTorch deterministic algorithms are on by default. No cross-version/device
bitwise guarantee is made.

If a CUDA operation rejects strict deterministic mode, explicitly opt out in a
separate output directory:

```bash
python -m gnn_benchmark.cli run --device cuda --allow-nondeterministic \
  --output 'tests and reports/reports/gpu_fast'
```

This allows the operation but relaxes reproducibility; it is not enabled silently.
The RTX GPU path has not been executed on this CPU-only machine. Default width
256, batch size 32 and two graph layers are starting settings, not a VRAM-fit
promise for every molecule/model. CUDA out-of-memory failures are recorded, not
silently retried with different batch sizes or reduced models.

Best checkpoints for tens of thousands of runs can consume substantial storage.
Results include per-run trainable parameter counts, actual durations, executed
epochs and effective hidden widths so storage/runtime can be planned from a real
GPU pilot. No GPU runtime estimate is inferred from CPU diagnostics.

## 5. Datasets and leakage controls

All 13 discovered datasets are included:

| Classification: ROC-AUC ↑ | Regression: MAE ↓ |
|---|---|
| BBBP | Lipophilicity (paper alias: Lipo) |
| Tox21 | FreeSolv |
| ToxCast | ESOL |
| SIDER | QM7 |
| HIV | QM8 |
| BACE | QM9 |
| MUV | |

Raw inputs are read only. Graph features are nine dense atom descriptors:
scaled atomic number, degree, formal charge, hydrogen count, hybridization code,
aromatic/ring flags, mass and chirality code. Bonds become bidirectional edges;
**bond attributes are not used in this experiment**. Molecular categorical
embeddings from the paper/source are not reproduced.

- Use seeded 80/10/10 **scaffold-group splits**, with exact group membership IDs
  saved. Entire acyclic/empty-scaffold groups are kept together; large groups can
  make actual fractions differ from their targets. There is no silent random-split
  fallback. Exact split IDs are not asserted identical to the paper's IDs.
- Fit atom-feature statistics only on training graphs; no learned preprocessing
  still uses the same training-fitted statistical scaling as other candidates.
- Regression target means/scales are fitted only on training labels. Optimize
  normalized targets and report MAE after reversing the transform.
- Classification uses masked BCEWithLogits, including multilabel targets. Missing
  labels stay NaN and never enter the objective. Missing is not treated as class 0.
- Macro ROC-AUC averages only tasks with both classes observed in the evaluation
  split. Report valid task counts; undefined AUC remains empty, not zero.
- Tasks with no observed training labels are excluded from evaluation.
- Invalid molecules, malformed rows and no-label rows are audited, not silently
  converted into arbitrary graphs.
- QM8 raw data repeats PBE0 column names. The explicit policy selects the first
  occurrence of each label, producing 12 unique targets. Positional indices are
  saved, so duplicate CSV headers cannot silently overwrite values.
- QM9 selects the 12 target columns mu, alpha, homo, lumo, gap, r2, zpve, u0,
  u298, h298, g298 and cv. Mixed-unit macro MAE is reported as this explicit local
  metric; equality to the paper's unit/aggregation convention is unverified.

## 6. Best checkpoint and inference

Each successful run writes:

```text
reports/gpu_200_epoch_plan/
  results/<dataset>/<model_id>/
    seed-0.json              Metrics, history, epoch, parameters, provenance
    seed-0-best.pt           Best weights, model config, fitted statistics
```

Checkpoints are saved on validation improvement **before** any test evaluation.
The final test score evaluates those saved best weights exactly once per job.
JSON also records CSV hash, code hash, seed, target columns, feature protocol and
training environment. No best model is selected using the test set.

```python
# Replace the path below with an actual completed checkpoint from your GPU run.
# This snippet illustrates the checkpoint API; it requires that checkpoint file.
from gnn_benchmark.checkpoints import load_best

model, stats, metadata = load_best("path/to/seed-0-best.pt", device="cuda")
feature_mean, feature_scale, target_mean, target_scale, supported_tasks = stats
# For a PyG batch using the same nine feature definitions:
# batch = batch.to("cuda")
# batch.x = (batch.x - feature_mean) / feature_scale
# with torch.no_grad():
#     output = model(batch)
#     if metadata["kind"] == "regression":
#         predictions = output * target_scale + target_mean
#     else:
#         probabilities = output.sigmoid()
```

Sigmoid/tanh factories are automatically registered by `load_best`. The checkpoint
uses tensor-only-safe `weights_only=True` loading; graph caches are internally
generated trusted PyG object caches and must not be replaced with untrusted files.

## 7. Reports merged with the paper

`reference/paper_results.csv` contains **192 entries** extracted from tables 1–3
of `3711896.3736984.pdf`: 12 backbone variants in table 1, 11 comparison models
in table 2, and 9 regression variants in table 3, each across six datasets.
The source PDF hash, table, model, mean, std and reported/OOM status are retained.
Few-shot table 4 is not merged as if it were ordinary supervised training.

`all_models_datasets.csv` has **19,968 candidate/dataset rows**: 19,344 eligible
and 624 excluded. It includes stage choices, basis, run counts, mean/std when
measured, trainable parameters and the published best reference for context.
Sample std is empty with one seed rather than misrepresented as a five-seed std.

`paper_comparisons.csv` joins every local candidate to relevant published models
from tables 2/3 on the same dataset. It preserves paper and local scores separately,
reports numeric deltas only for measured local values, and labels comparisons
`context_only_protocol_not_matched`.

**These are contextual comparisons, not claims of exact reproduction.** Dense
atom features, omitted bond features, scaffold IDs, model internals, validation
selection, quantum-target units and MAE aggregation may differ from the paper.
MUV has no full-supervised reference in these tables. Do not interpret a numerical
delta as statistically established superiority under a matched protocol.

Reports refresh after every 25 newly attempted jobs and at bounded/normal exit.
For current results before that interval, run the separate `report` command.
Individual result/checkpoint files are written immediately as available.

## 8. Verification and local diagnostics

```bash
pytest 'tests and reports/tests' -q
ruff check 'tests and reports'
```

**Local verification: 1,505 tests passed, 94.29% new-package line coverage; lint passed.**

Every one of the **1,488 eligible independent architectures** has a deterministic
forward/backward check. Additional checks cover depth counts, exclusion counts,
finite gradients, all bases, losses/missing labels, tie-correct AUC, train-only
statistics, scaffold isolation, repeated training, best checkpoint roundtrip,
validation-only selection and paper table extraction. If the PDF is unavailable,
the extraction test explicitly skips; the reference CSV remains usable.

An initial pipeline diagnostic, run before the GPU-only clarification, used one
model, at most 512 molecules per dataset and two epochs. It is in
[diagnostic_512/REPORT.md](reports/diagnostic_512/REPORT.md). All 13 dataset jobs
completed; MUV's small evaluation subset had no valid two-class AUC task, which is
reported as undefined. These values are **not full-data experiment results**.

`reports/full_sweep/` is the earlier CPU preparation audit/initial tied-grid plan;
it contains no full benchmark training results. The current intended experiment
is `reports/gpu_200_epoch_plan/`. Graph caches have been prepared locally for all
13 full datasets and can be transferred, but rebuilding them on the GPU machine
under its RDKit version may be necessary. Caches are excluded by this directory's
`.gitignore`; source CSVs are not duplicated here.

A [GitHub Actions template](ci/benchmark-ci.yml) is supplied inside this directory.
It is intentionally not active until copied to `.github/workflows/`, because all
new code was requested here. It runs architecture tests, not the 200-epoch sweep.
