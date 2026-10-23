# Factor switching

`Python 3.12+` · `JAX` · `NumPy` · `Optax` · `SciPy` · `GPL-2.0-only`

**Rotation–Sign–Permutation (RSP) alignment of posterior factor loadings, in JAX.**

Posterior draws from a factor model are identified only up to rotation, column
signs, and column order. `fs` rotates each draw to a sparse varimax orientation,
then aligns all draws to a common reference so that they can be summarised
coherently. Model assumptions and the algorithm are described in the
[algorithm notes](Factor_Switching.md).

```python
import numpy as np
from fs import rsp_exact

rng = np.random.default_rng(0)
base = np.array([[1.0, 0.0], [0.8, 0.0], [0.0, 1.0], [0.0, 0.6]])
draws = base + 0.05 * rng.normal(size=(100, 4, 2))
draws *= rng.choice([-1.0, 1.0], size=(100, 1, 2))   # simulate label switching

result = rsp_exact(draws)
print(result.mean_loadings)                  # (p, q) aligned mean
print(result.diagnostics["stop_reason"])     # converged_threshold
```

## Installation

Requires Python 3.12 or newer. From a checkout of this repository:

```sh
python -m pip install -e .
```

The package is imported as `fs`.

## API

| Function | Purpose |
|---|---|
| `rsp_exact()` | Varimax rotation followed by sign and permutation alignment |
| `transform_scores()` | Apply the estimated transforms to factor scores |
| `from_flat()` / `to_flat()` / `lambda_colnames()` | R-style flat `(T, p*q)` helpers |

`rsp_exact()` returns an `RSPResult` with `aligned_loadings`, `mean_loadings`,
per-draw `signs` and zero-based `permutations`, loss histories, convergence
diagnostics (`n_iter`, `converged`, `status_code`, `diagnostics`), and optional
per-draw orthogonal `transforms` (`return_transforms=True`). Pass those
transforms to `transform_scores()` to co-transform matching factor scores.

## Input and options

- Inputs are finite, real-valued arrays shaped `(T, p, q)`, or flat
  `(T, p*q)` in variable-major order with explicit `p`/`q` or strict
  `LambdaV1_1 … LambdaVp_q` column names. Complex input is rejected.
- `precision="float32"` (default) casts explicitly and leaves JAX's global x64
  setting untouched. `precision="float64"` enables global x64 and leaves it
  enabled after the call (even if the call then fails validation).
- `rotate=False` skips Step 1 varimax (useful for aligning across chains).
  `sp_path="elim"` is the fast default; `sp_path="enum"` is a validation path.
  "Exact" refers to the sign/permutation subproblem at fixed reference — the
  full alternating procedure is not guaranteed to reach a global optimum.
- `assignment_backend` selects the assignment solver (`"auto"` default,
  `"optax"`, or host-side `"scipy"` for validation).
- `metadata={...}` attaches hashable scalars or scalar sequences to the result;
  it never affects the kernels.
- `enum` and `assignment_backend="scipy"` run eagerly rather than under `jax.jit`,
  so loss scalars may differ in their last few digits while aligned results agree.

## Documentation

- [Algorithm and model assumptions](Factor_Switching.md)
- [Runnable example](examples/quickstart.py)

## License

[GPL-2.0-only](LICENSE)
