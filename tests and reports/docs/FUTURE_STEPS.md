# Future experiment work

## Execute on the target GPU

1. Install CUDA PyTorch and the two local packages on the RTX 4060 Ti machine.
2. Inspect `plan` and run one bounded job to measure real throughput, peak VRAM and
   checkpoint storage. Do not extrapolate quality from the local diagnostic.
3. Choose a consistent batch size before the main sweep; use a new output if
   settings differ. Default width is 256, but no universal VRAM-fit claim is made.
4. Execute/resume all intended seeds, retaining failure logs and source/data hashes.
5. Regenerate reports after completion and inspect undefined metrics and failures
   before making comparisons.

## Improve exact comparability with the paper

- Recover exact scaffold split IDs, target selections, regression units and
  feature pipelines. The current contextual deltas do not establish reproduction.
- Resolve duplicate QM8 headers against the source dataset documentation.
- Match QM9 target/unit averaging before treating its macro MAE as comparable.
- Add categorical atom/bond embeddings and exact backbone variants if reproducing
  GNN-SKAN rather than conducting this explicitly different dense-feature grid.
- Keep MUV few-shot/episodic protocols separate from the full-supervised sweep.

## Engineering extensions

- Mid-epoch/optimizer/RNG training resume while retaining best-so-far checkpoints.
- Optional mixed precision and memory profiling, after numerical/gradient checks.
- Explicit multi-GPU or process sharding with atomic job ownership; do not launch
  concurrent writers to the same result/checkpoint directories in the current runner.
- More configurable MLP depth, message depth, attention stages and additional
  backbones. Each new axis must update candidate counts and deterministic tests.
- Parameter-budget-matched comparisons for identity-preprocessing versus projected
  variants. Current true identity variants retain a different hidden width.
- Confidence intervals and paired-seed comparisons under fully matched protocols.
- Richer plots and dashboards after real sweep results exist.
- Select a checkpoint retention/compression policy for large grids with substantial
  storage requirements; current default saves best weights for every job as requested.
- Activate the supplied CI template by copying it to `.github/workflows/` when
  ready; no workflow outside this new directory was changed automatically.
