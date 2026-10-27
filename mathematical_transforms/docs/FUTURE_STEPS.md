# Future transform work

## Numerical transforms

1. Add explicit quadrature selection (trapezoidal, Simpson where applicable) with
   convergence-rate tests on known finite-domain integrals.
2. Add chunked query evaluation to avoid materializing large `[queries,samples]`
   kernel tensors; verify output and gradient equivalence.
3. Add continuous Fourier, Mellin and other transforms as separate, documented
   APIs with domain and normalization conventions. Do not label feature bases as
   complete numerical transforms.
4. Consider numerical inverse Z and inverse Laplace algorithms only with documented
   convergence, contour and conditioning assumptions, and independent reference
   tests. No inverse transforms are implemented currently.
5. Support explicit tail models and error estimates for Laplace integration;
   avoid implying that an arbitrary finite grid determines an infinite signal.
6. Add overflow diagnostics and scaling options for extreme transform queries.

## KAN basis extensions

1. Add optional trainable positive time nodes and Z-query scaling, with stable
   parameter constraints, initialization policies and gradient/state tests.
2. Compare fixed versus trainable kernels and alternative positive query maps.
3. Consider damped oscillatory Laplace kernels with real/imaginary feature
   splitting for real-valued graph networks.
4. Benchmark learned coefficients and base residuals separately; test predictive
   quality on held-out graphs rather than only a tiny training fixture.
5. Add standalone KAN numerical examples for signals, distinct from graph examples.

## Interface and reproducibility

1. Add a unified optional-extension discovery mechanism if more independent
   packages are introduced. Keep registration explicit and collision-safe.
2. Add a transform-specific CLI if users need file-based signal conversion;
   currently the public interface is Python plus runnable examples.
3. Add plots of transform magnitude/phase, convergence and learned KAN functions.
4. Add a GPU CI runner when available, retaining explicit skips on CPU runners.
5. Run the configured GitHub matrix and record supported backend combinations.
6. Test built wheels in clean dependency environments and add dependency lockfiles
   for stable experiment reproduction.
7. Add broader graph-task and edge-encoder integration matrices if extending
   task-specific transform presets; current new-basis training tests target node
   prediction while the sibling framework tests its shared task machinery.
