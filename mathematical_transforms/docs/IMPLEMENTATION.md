# Transform package implementation log

Date: 27 October 2026.

## Scope and decisions

The user requested a separate directory for transforms such as Z and Laplace and
confirmed that both numerical transforms and transform-inspired KAN bases were
wanted. Created `mathematical_transforms/` as an independently installable package
named `kan-transforms`, imported as `kan_transforms`.

Numerical tools depend only on PyTorch. The optional KAN adapter imports the
sibling `kangnn` package only when registration is explicitly requested. The
original framework's source and default basis choices are unchanged. The new
package registers `laplace` and `z_transform`; ordinary KAN config objects can
then select them in every learned stage.

## Mathematical conventions

Numerical Z evaluation is the finite sum
`X(z)=sum_{k=0}^{N-1} x[k]*z**(-(k+start_index))`. `start_index=0` is a truncated
unilateral sequence, consistent with the
[UBC transform notes](https://www.math.ubc.ca/~feldman/m267/genfnl.pdf).
An integer negative start permits finite sequences with negative indices.
Zero queries are deliberately rejected even for special sequences whose limit
might exist. No symbolic simplification or infinite-sequence convergence inference
is performed. On the unit circle with N equally spaced points, this formula
matches the forward DFT; an automated comparison checks it against PyTorch FFT.

Numerical Laplace evaluation integrates `x(t)*exp(-s*t)` by PyTorch trapezoidal
quadrature on the supplied real, nonnegative, strictly increasing grid. Only the
provided interval is integrated. A sample at time zero is permitted but not
required. Accuracy depends on grid resolution and omitted portions of the
signal. Numerical tests compare to the *finite-interval* analytical exponential
integral, avoiding an accidental claim of exact integration to infinity.

Both transforms accept batched signals and a configurable sample dimension.
Queries are scalar or 1D, returning the batch shape or batch shape plus query
count respectively. Real and complex inputs use PyTorch promotion and complex
autograd; real scalar objectives are expected for training, following
[PyTorch's complex-gradient convention](https://docs.pytorch.org/docs/stable/notes/autograd).
Inputs must share a device and use float32/64 or complex64/128. Half-precision
and integer numerical inputs are explicitly unsupported. Finite inputs can still
overflow for extreme query values; results are not silently clipped.

## KAN kernel design

The KAN modules are kernel feature expansions, not transforms of whole signals:

- `LaplaceBasis`: `B_k(x)=exp(-softplus(x)*t_k)` with positive time-node buffers.
  The default 8 time nodes span `[0.1,4]`.
- `ZTransformBasis`: `B_k(x)=exp(-k*log1p(softplus(x)))`, equivalent to
  `(1+softplus(x))**(-k)` for k starting at zero. The zero-order basis is constant.

Positive feature maps let ordinary real node features use these kernels without
zero singularities or exponentially growing activations. They change the query
coordinate; these modules do not evaluate a transform directly at `s=x` or `z=x`.
Activations stay in `[0,1]` for finite real inputs, although derivatives can
saturate. Fixed nodes are buffers that move with `.to(device)` and serialize in
state dicts. Outer KAN coefficients and the existing optional SiLU branch remain
trainable. These are new framework design choices, not claims about any supplied
paper's architecture.

`register_kan_bases()` is idempotent for these same class factories. It validates
both names for collisions before inserting either, so conflicting user extensions
cannot cause a partial registration. Checkpoint/config users must register before
reconstruction. Fully custom basis classes still use the existing registry.

## Files and steps completed

1. Inspected the sibling package's basis shape contract, custom extension example,
   build configuration and GitHub workflow.
2. Confirmed the requested scope: standalone numerical tools plus KAN bases.
3. Added a separate setuptools project, test extras and source manifest.
4. Implemented `src/kan_transforms/numerical.py`: tensor/shape/device validation,
   finite Z sums, complex kernels and nonuniform finite-interval Laplace quadrature.
5. Implemented `bases.py`: bounded real kernel families, buffer persistence and
   explicit optional registration with conflict protection.
6. Exported the public API from `__init__.py`.
7. Added standalone numerical and mixed Laplace/Z KAN training examples.
8. Added numerical reference, DFT, gradcheck, validation, device, shape and batch
   tests, plus KAN basis/registration/training/checkpoint tests for all backbones.
9. The initial Laplace gradcheck perturbed a time-zero boundary to negative time.
   Corrected the numerical-derivative fixture to an interior positive time grid;
   boundary rejection remains separately tested. Corrected a lint issue in the
   test fixture's local callable.
10. Added a dedicated GitHub workflow for Python 3.10/3.12 CPU testing with pinned
    torch/PyG versions, a 90% coverage gate, examples, original-framework regression
    tests, package builds and artifact delivery. Remote execution is not claimed.
11. Wrote README examples, conventions, GPU instructions and integration limits,
    this work log, and a separate future-work plan. Linked the new package from
    the repository README.
12. Installed the package locally in editable mode without network downloads and
    ran lint, tests, examples, combined regression checks and distribution builds.

## Test strategy and verification

No tests download data. Numerical fixtures use fixed tensors; graph training
repeats seeded CPU runs and compares histories exactly. Gradient checks use double
precision and finite differences. Complex cases include both signal and query
derivatives. Tests cover scalar/vector queries, batched sample axes, Z indexing,
nonuniform Laplace grids, exact DFT equivalence, invalid inputs, basis saturation,
state dictionaries and registration collisions.

Every new basis is exercised through GCN, GraphSAGE, GAT and GATv2, with KAN
preprocessing, message, update, attention, readout and decision configs. Model
checkpoints roundtrip exactly in evaluation mode. Existing edge-encoder support
accepts these bases through the same transform factory.

Initial new-package results: **34 passed, 3 CUDA tests skipped, 97.56% line
coverage**; lint passed. Skips mean GPU execution is unverified on this machine,
not a successful GPU test. The implementation uses device-preserving tensor
operations and includes GPU parity/backward tests for execution on CUDA machines.
Existing Python 3.14/PyG upstream deprecation warnings remain visible.

Final combined regression and distribution checks are recorded below after
completion. GitHub Actions has been configured but has not been run remotely.

### Final local verification

- New package: **34 passed, 3 skipped**, **97.56% coverage**, 90% gate satisfied.
- Combined suites: **180 passed, 3 skipped**; all skips require unavailable CUDA.
- `ruff check mathematical_transforms`: passed.
- Numerical example: Z outputs `[6, 2.75, -2-2j]`; sampled Laplace values agree
  with the displayed exponential reference within the documented discretization
  and finite-tail approximation.
- KAN example: fixed seed 7, 30 epochs; objective decreased from approximately
  `0.838587` to `0.043325` on the offline toy training data.
- `python -m build mathematical_transforms --no-isolation`: built source archive
  and universal wheel successfully. Backend: existing local Python 3.14.7,
  PyTorch 2.14.1+cpu, PyG 2.8.1.
- Dedicated CI workflow is present but remote execution and CUDA performance
  remain unverified. No packages were published.
