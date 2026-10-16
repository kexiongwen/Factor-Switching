# Factor switching

Rotation-Sign-Permutation (RSP) alignment of posterior factor loadings using JAX.
Use it with existing MCMC draws from factor models with rotational, sign, or
permutation indeterminacy before computing summaries. Model assumptions are
described in the [algorithm notes](Factor_Switching.md).

The package is imported as `fs`.

## Installation

Requires Python 3.12 or newer. From a local checkout:

```sh
python -m pip install -e .
```

## Quick start

```python
import numpy as np
from fs import rsp_exact

# Synthetic draws for demonstration; replace with your posterior samples.
rng = np.random.default_rng(0)
base = np.array([[1.0, 0.0], [0.8, 0.0], [0.0, 1.0], [0.0, 0.6]])
loadings = base + 0.05 * rng.normal(size=(100, 4, 2))
loadings *= rng.choice([-1.0, 1.0], size=(100, 1, 2))

result = rsp_exact(loadings)
aligned = result.aligned_loadings          # (draws, variables, factors)
print(result.mean_loadings)                 # (variables, factors)
print(result.diagnostics["stop_reason"])
```

See [the full example](examples/quickstart.py) for alignment and diagnostics.

## Main functions

| Function | Purpose |
|---|---|
| `rsp_exact()` | Varimax rotation followed by sign and permutation alignment |
| `transform_scores()` | Apply the corresponding transformations to factor scores |

## Input and precision

- Pass finite NumPy or JAX arrays shaped `(T, p, q)`: draws, variables, factors.
- Flat arrays shaped `(T, p*q)` use variable-major (C) order: zero-based entry
  `(r, j)` is in column `r*q + j`. Supply either `p` and `q`, or `colnames`
  following `LambdaV1_1, ..., LambdaV1_q, ..., LambdaVp_q`. These extra shape
  arguments are only for flat input.
- Alignment defaults to `precision="float32"`, casting inputs without changing
  JAX's global x64 setting. Use `precision="float64"` for double precision;
  this enables global x64 and leaves it enabled after the call.
- RSP supports `sp_path="elim"` (default) and `sp_path="enum"`. "Exact" refers
  to the sign/permutation subproblem for a fixed reference; the full alternating
  procedure is not guaranteed to reach a global optimum.

## Documentation

- [Algorithm and model assumptions](Factor_Switching.md)

Licensed under [GPL-2.0-only](LICENSE).
