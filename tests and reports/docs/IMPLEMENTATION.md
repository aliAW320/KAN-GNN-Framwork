# Benchmark implementation and work log

Date: 28 October 2026. This is a separate experiment project; the existing graph
framework's runtime source has not been modified for this task.

## Request and accepted interpretation

Initial proposal: four preprocessing options, eight tied graph options, four
heads and six basis families, giving 768 candidates. The user clarified that
message and update transforms must be independent, and that training will occur
on a different RTX 4060 Ti machine with 200 epochs and saved best results.

The final grid is 4 preprocessing x 4 backbones x 2 messages x 2 updates x 4
heads x 6 families = 1,536. Of these, 48 contain no KAN and are excluded before
model construction. The 1,488 eligible models across 13 datasets yield 19,344
runs per seed. Five seeds default to 96,720 jobs for paper-style mean/std;
single-seed or other explicit seed sets are supported.

No full experiment is launched locally. CUDA is the run default, and missing
CUDA produces an explicit error. A small diagnostic performed before the GPU
clarification is separately labeled and never used as a full-data result.

## Inspection and dataset decisions

Inspected local data headers with Python's CSV reader. There are 13 raw molecular
CSV files. BACE has many additional descriptor columns; only its `Class` target
and `mol` SMILES are selected. Regression targets and classification fields are
explicit rather than inferred from arbitrary numeric columns. QM8 duplicate names
are read positionally and deduplicated with an explicit first-occurrence policy.
QM9 selects 12 listed targets instead of accidentally treating identifiers or
rotational constants as labels.

Installed RDKit into the existing local environment to validate CSV-to-graph
processing. The first automatic approval review timed out; its permitted retry
succeeded. No upstream training repository or downloaded training script was run.

Atom featurization: nine scaled numeric descriptors. Bonds supply both directions
of COO topology, with no bond-feature tensor. This is a deliberately documented
experimental feature schema, not the molecular categorical embeddings from
GNN-SKAN. It remains fixed across all candidates and basis choices.

Entire Murcko scaffold groups, ignoring scaffold chirality, are assigned to
seeded 80/10/10 splits. Group ordering is deterministic by size and seeded hash.
Acyclic molecules share an empty-scaffold group; large groups can skew fractions.
The last groups reserve nonempty validation/test parts rather than falling back
to row-level random splits. The same CSV order/group split is reused across every
model for each seed. Exact canonical duplicates cannot cross scaffold groups.

Feature mean/scale uses training atoms only. Regression target mean/scale uses
training labels only, and predictions are transformed back for evaluation.
Training label availability gates evaluation tasks. Missing labels are masked,
not zero-filled for the loss. The rank-based AUC implementation handles score ties
and returns undefined when one class is absent.

## Architecture construction

`grid.py` enumerates stable, readable IDs. No learned projection is inserted in
the no-preprocessing variant: the encoder then operates at its nine input features.
Other preprocessing modes project to width 256 by default. This affects parameter
counts and is explicitly recorded rather than calling the modes equal capacity.

Each KAN depth maps directly to the framework's KANLinear stack. Message and
update each have one configurable feature transform. Every graph layer is followed
by the framework's fixed ReLU, and the operator performs its own permutation-safe
aggregation. Attention scorers remain linear, avoiding another uncounted grid axis.

Sigmoid/tanh basis classes implement shifted, scaled real univariate functions;
their centers and positive scales are learned. They obey the original basis shape
contract and are registered only by this package. Fourier uses nine bases so the
constant/sine/cosine structure is valid; other families use eight bases.

## Training, validation selection and persistence

Seeded minibatch Adam, default lr 0.001, batch size 32, two graph layers,
200 epochs, patience 200. Default patience does not stop a 200-epoch run early.
Smaller explicit patience enables early stopping. Classification uses masked
BCEWithLogits; regression uses masked MSE on normalized targets.

Validation macro ROC-AUC (maximize) or raw-unit macro MAE (minimize) selects the
best weights. Undefined validation AUC falls back to validation loss, explicitly
recorded. The test set never selects weights and is evaluated once at the end on
the selected best state. Every improved checkpoint is written atomically with
config, weights, train-fitted transformations, epoch and provenance. Mid-epoch
optimizer resume is not implemented; unfinished jobs restart from the seed.

`load_best` registers extension bases, loads tensor-safe checkpoint contents and
returns an eval model plus fitted transformations and metadata. Checkpoints are
CPU-stored and move to the requested inference device. The separate experiment
checkpoint format avoids changing the core framework's checkpoint contract.

Per-job result JSON includes trainable parameters, effective hidden width, best
and executed epochs, durations, target/metric task counts, validation history,
seed and file/code hashes. No dataset, result or model is silently modified to
fit GPU memory. Failures include tracebacks and can be retried explicitly.

## Provenance and resume

Plans capture settings, model IDs, dataset file hashes, target schemas and planning
environment. Actual training records its own Python/PyTorch/PyG/RDKit/CUDA/GPU
environment. A CPU-created plan is therefore not represented as a measured GPU
run. Changed source, CSV bytes or protocol settings require a different output;
training-environment changes are also rejected once a training environment exists.

Internal caches are keyed by CSV hash, loader source, featurization version,
row limit and RDKit version. Their trusted local PyG serialization is separate
from the tensor-safe model checkpoint loader. Completed jobs skip on repeated
invocation; failed jobs skip unless retry is requested. Selection filters and
per-invocation bounds do not alter the underlying experiment grid.

## Paper extraction and reporting

Used `pdftotext -layout` on the supplied GNN-SKAN PDF. Parsed tables 1, 2 and 3
into 192 numeric/OOM entries, preserving model/table/dataset/mean/std and PDF hash.
Tests verify table coverage and a known published value. The provided reference
CSV avoids requiring the paper or the text-extraction executable on the GPU host.

The main matrix has 19,968 candidate/dataset rows, with 624 exclusions and 19,344
eligible combinations. All expected rows remain visible before training. Completed
seed results are aggregated with sample std only when multiple defined scores
exist. Partial/failed/undefined/pending statuses remain distinguishable.

The second CSV joins every candidate to relevant published models from table 2
(classification) or 3 (regression). Local and paper values are distinct fields;
missing local scores remain empty. Deltas are labeled context-only, because
features, split IDs, quantum-target units and aggregation conventions are not
established as identical. MUV is evaluated locally but not compared against a
few-shot paper result as if it were full supervised training.

## Files added

- `src/gnn_benchmark/grid.py`: enumeration, new bases and model construction.
- `datasets.py`: schemas, featurization, missing labels, hashing and scaffold splits.
- `metrics.py`: masked losses and tie-correct task-wise/macro evaluation.
- `engine.py`: seeded minibatch training, validation selection and final evaluation.
- `checkpoints.py`: atomic best weights plus fitted preprocessing and safe loading.
- `paper.py`: extracted table references and provenance.
- `reporting.py`: complete matrix, paper joins, summary JSON and Markdown report.
- `cli.py`: plan, prepare, run, report; selection, bounded work, guards and resume.
- `tests/`: every eligible architecture, numerical/statistical/provenance checks,
  best checkpoints, CLI and reporting states.
- `reference/paper_results.csv`: 192 published comparison entries.
- `reports/`: plans, audits, small labeled diagnostics and comparison exports.
- `run_gpu.sh`: optional quoted-path Linux helper for the target environment.
- `ci/benchmark-ci.yml`: optional CI template kept within the requested directory.
- `pyproject.toml`, `.gitignore`, README and documentation.

## Work sequence and verification

1. Inspected core APIs, data files, local GPU availability and missing dependencies.
2. Asked about ambiguous model enumeration and training budget while implementing
   independent dataset/reporting functionality.
3. Added the initial tied grid plus optional independent enumeration. Added six
   basis choices and true identity preprocessing.
4. Implemented and tested schemas, graph loading, scaffold splits, masked training,
   metrics, resume and paper extraction.
5. Ran all 720 initial eligible architectures and a bounded diagnostic on all
   datasets (up to 512 molecules, two epochs, one model). All 13 jobs executed;
   MUV's small test subset had undefined AUC, correctly retained as undefined.
6. The user selected independent modes and a separate GPU host with 200 epochs.
   Changed the main defaults to 1,536 candidates, width 256, CUDA and 200 epochs,
   with five configurable seeds. No full local training was launched.
7. Added validation-metric selection, atomically saved best checkpoints and
   inference loading, along with actual-training environment guards.
8. Prepared full graph caches for all 13 datasets and saved all 65 dataset/seed
   splits for the intended five-seed experiment, without training those models.
9. Expanded architecture tests to all 1,488 eligible independent configurations;
   added CLI failure/retry/resume and completion-state tests.
10. Wrote GPU launch instructions, data/metric contracts, contextual comparison
    limits and a roadmap. Kept all new code and generated reports in this directory.

Final test results and dataset audit are appended below.

## Full-data preparation audit

| Dataset | Valid molecules | Rejected rows | Targets | Train/val/test (seed 0) |
|---|---:|---:|---:|---|
| bace | 1513 | 0 | 1 | 1209/152/152 |
| bbbp | 2039 | 11 | 1 | 1631/204/204 |
| esol | 1128 | 0 | 1 | 777/254/97 |
| freesolv | 642 | 0 | 1 | 435/152/55 |
| hiv | 41120 | 7 | 1 | 32896/4112/4112 |
| lipophilicity | 4200 | 0 | 1 | 3360/420/420 |
| muv | 93087 | 0 | 17 | 74469/9309/9309 |
| qm7 | 6834 | 0 | 1 | 5466/684/684 |
| qm8 | 21786 | 0 | 12 | 17428/2179/2179 |
| qm9 | 133885 | 0 | 12 | 107107/13389/13389 |
| sider | 1427 | 0 | 27 | 1141/143/143 |
| tox21 | 7823 | 8 | 12 | 5643/1474/706 |
| toxcast | 8579 | 18 | 617 | 6223/1578/778 |

Full per-row rejection reasons, exact target columns and all five seeds are in dataset metadata/split JSON files.

## Final verification

- **1,505 tests passed**; every one of the 1,488 eligible independent models was
  checked for a finite forward pass and backward gradients.
- New-package line coverage: **94.29%** (561/595 statements), with XML at
  `reports/benchmark_coverage.xml`.
- Ruff passed; Linux helper shell syntax passed; README file links resolved.
- All 13 complete CSV datasets were prepared and all 65 dataset/seed split files
  were generated for the main planned experiment.
- Main plan: 19,344 pending eligible model/dataset rows, 624 excluded rows, zero
  full benchmark result files. No 200-epoch run was launched here.
- The earlier separately labeled diagnostic has 13 completed short jobs; its MUV
  AUC is undefined, not replaced by zero or a paper result.
- GPU execution, VRAM fit, remote CI and full-sweep scientific results remain
  unverified until execution on the target machine.

The optional CI template has a 90% coverage gate. It remains inactive within this
new directory until explicitly installed in the repository workflow directory.
