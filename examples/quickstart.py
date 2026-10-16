"""Quickstart: synthetic draws -> rsp_exact -> diagnostics and mean loadings.

Run: python examples/quickstart.py
"""
import numpy as np

from fs import rsp_exact


def make_draws(seed, t=500, p=8, q=2):
    rng = np.random.default_rng(seed)
    true = np.zeros((p, q))
    true[:5, 0] = 1.0   # unequal block sizes / scales avoid exact objective ties
    true[5:, 1] = 0.8
    draws = true[None] + 0.05 * rng.normal(size=(t, p, q))
    for i in range(t):  # simulate raw-MCMC switching
        draws[i] = draws[i][:, rng.permutation(q)] * rng.choice([-1.0, 1.0], q)[None]
    return draws


def main():
    draws = make_draws(seed=0)
    res = rsp_exact(draws)  # default precision: float32
    print(f"converged={bool(res.converged)}, "
          f"n_iter={int(res.n_iter)}, final loss={float(res.loss_final):.4f}, "
          f"dtype={np.asarray(res.aligned_loadings).dtype}")
    print("stop reason:", res.diagnostics["stop_reason"])
    print(np.round(np.asarray(res.mean_loadings), 2))


if __name__ == "__main__":
    main()
