"""Batched varimax using the matrix iteration of R `stats::varimax` with
`normalize = FALSE`, default `eps = 1e-5`, and at most 1000 iterations by
default. Returns rotated loadings and rotation matrices for composition.

The default `polar="qdwh"` obtains the polar factor W of B through QR-based
dynamically weighted Halley iteration. Its stopping quantity is
d = <W, B>_F = tr(WᵀB), equal to the sum of singular values for a polar factor.
The `polar="svd"` path obtains this factor directly from an SVD.

QDWH results must have converged inner solves and finite orthogonality error
||WᵀW - I||_F <= orth_tol. All varimax matrix products (both the SVD and the
QDWH iteration paths, and the orthogonality check) are pinned to JAX's
`highest` matmul precision, so a float32 GPU run is not limited by the
backend's default float32 matmul precision (e.g. TF32); CPU results are
unchanged. Failed draws are recomputed with SVD outside the jitted batch;
accepted draws retain their QDWH results.

`return_n_iter=True` also returns per-draw iteration counts. The batched while
loop runs until its slowest active draw stops, while completed draws retain
their states; the counts describe individual draws, not batch execution cost.
"""
from __future__ import annotations

import numbers

import jax
import jax.numpy as jnp
import numpy as np

from .._numeric import (any_nonzero, binary_exponent, max_abs, scale_pow2)


def _staged_scale(x: jax.Array):
    """Return loadings scaled in two power-of-two stages, plus the exponent.

    Uniform positive scaling preserves the varimax maximizers in exact
    arithmetic, while the intermediate B grows with the fourth power of
    input magnitude. The binary exponent e = floor(log2(max|x|)) is read
    exactly from the IEEE exponent bits (binary_exponent); a floating log2
    is not used because it can round down-and-out by one for exact powers of
    two and is undefined for subnormals. The scale is applied as two
    exponent-bit shifts (scale_pow2), e = e1 + e2, rather than as floating
    divisions: a division would flush a subnormal operand to zero on
    flush-to-zero backends before the power-of-two factor could compensate.
    The split keeps each shift small and mirrors the specification's
    two-factor construction.

    Returns (scaled loadings, e) with e = 0 when the draw is zero or
    non-finite. Floating-point rounding can change the iteration path,
    particularly for degenerate or non-unique optima.
    """
    maxabs = max_abs(x)
    e = binary_exponent(maxabs)
    finite = jnp.all(jnp.isfinite(x))
    ok = finite & any_nonzero(x) & (e < jnp.finfo(x.dtype).maxexp)
    e = jnp.where(ok, e, 0).astype(jnp.int32)
    e1 = jnp.floor_divide(e, 2)
    e2 = e - e1
    xs = scale_pow2(scale_pow2(x, -e1), -e2)
    return xs, e


def _varimax_body(xs, p, tt):
    """Return the matrix B whose polar factor defines the next rotation."""
    z = xs @ tt
    colsums_sq = jnp.sum(z**2, axis=0)  # colSums(z^2)
    # B = crossprod(x, z^3 - z %*% diag(colsums_sq) / p)
    b = xs.T @ (z**3 - z * (colsums_sq / p)[None, :])
    return b


def _varimax_single(x: jax.Array, eps: float, max_iter: int):
    """x: (p, q) with q >= 2. Returns (rotated loadings, rotmat TT, n_iter).

    A zero draw keeps the identity rotation without entering the loop:
    its d sequence stays zero, so the strict stopping rule would not fire.
    """
    p, q = x.shape
    xs, e = _staged_scale(x)
    # Zero and non-finite draws keep the identity rotation without iterating
    # (any_nonzero inspects magnitude bits, so subnormal draws are active).
    active = jnp.all(jnp.isfinite(x)) & any_nonzero(x)

    def cond(state):
        _tt, d, dpast, i = state
        # R: break when d < dpast * (1 + eps); never true on the first sweep
        # (dpast = 0 and d >= 0 -> 0 < 0 is FALSE), matching R exactly.
        return active & (d >= dpast * (1.0 + eps)) & (i < max_iter)

    def body(state):
        tt, _d, _dpast, i = state
        b = _varimax_body(xs, p, tt)
        u, s, vh = jnp.linalg.svd(b, full_matrices=False)
        tt_new = u @ vh
        return tt_new, jnp.sum(s), _d, i + 1

    init = (jnp.eye(q, dtype=x.dtype), jnp.zeros((), x.dtype), jnp.zeros((), x.dtype), 0)
    tt, _d, _dpast, i = jax.lax.while_loop(cond, body, init)
    # Restore the original scale from the exact power-of-two prefix. This is
    # bitwise equivalent to x @ tt for ordinary ranges and keeps subnormal
    # outputs representable instead of flushing the matmul operands to zero.
    return scale_pow2(xs @ tt, e), tt, i


def _varimax_single_qdwh(x: jax.Array, eps: float, max_iter: int):
    """QDWH variant with the same stopping rule and scale normalization.
    x: (p, q) with q >= 2.

    A zero or non-finite computed norm of B keeps the current rotation.
    A zero B has no unique polar factor, and QDWH's zero result would not be
    an orthogonal rotation. No absolute small-norm cutoff is used: a small
    full-rank matrix can still require a rotation update.

    Per-draw QDWH convergence flags accumulate into `ok_conv`; a draw with any
    non-converged inner solve is marked for the host-side fallback."""
    from jax.lax.linalg import qdwh

    p, q = x.shape
    xs, e = _staged_scale(x)
    # Zero and non-finite draws keep I without iterating (subnormal-safe).
    active = jnp.all(jnp.isfinite(x)) & any_nonzero(x)

    def cond(state):
        _tt, d, dpast, i, _okc = state
        return active & (d >= dpast * (1.0 + eps)) & (i < max_iter)

    def body(state):
        tt, _d, _dpast, i, okc = state
        b = _varimax_body(xs, p, tt)
        u, _h, _n, conv = qdwh(b)
        b_norm = jnp.linalg.norm(b)
        good = jnp.isfinite(b_norm) & (b_norm > 0.0)
        tt_new = jnp.where(good, u, tt)
        d_new = jnp.sum(tt_new * b)  # d = <W, B>_F = tr(WᵀB) on the SCALED B
        okc = okc & (conv | (~good))  # skipped (degenerate) updates don't count
        return tt_new, d_new, _d, i + 1, okc

    init = (
        jnp.eye(q, dtype=x.dtype),
        jnp.zeros((), x.dtype),
        jnp.zeros((), x.dtype),
        0,
        jnp.asarray(True),
    )
    tt, _d, _dpast, i, okc = jax.lax.while_loop(cond, body, init)
    return scale_pow2(xs @ tt, e), tt, i, okc


def _default_orth_tol(dtype):
    return 1e-3 if dtype == jnp.float32 else 1e-9


def _varimax_impl(loadings, eps, max_iter, return_n_iter):
    t, p, q = loadings.shape
    if q < 2 or p < 2:
        # q < 2: identity rotation by spec 4.1. p < 2: for a single row the
        # varimax gradient B in (18) is identically zero (colSums(z^2)/p = z^2
        # for p = 1), so every update keeps the identity while d = 0 never
        # triggers the strict stopping rule: without this shortcut each draw
        # would burn the full max_iter (1000) budget.
        eye = jnp.broadcast_to(jnp.eye(q, dtype=loadings.dtype), (t, q, q))
        if return_n_iter:
            return loadings, eye, jnp.zeros((t,), dtype=jnp.int32)
        return loadings, eye
    with jax.default_matmul_precision("highest"):
        rotated, rotmats, n_iters = jax.vmap(
            lambda x: _varimax_single(x, eps, max_iter)
        )(loadings)
    if return_n_iter:
        return rotated, rotmats, n_iters.astype(jnp.int32)
    return rotated, rotmats


def _varimax_qdwh_impl(loadings, eps, max_iter, orth_tol):
    """Batched QDWH varimax core, returning per-draw acceptance flags.

    Returns (rotated, rotmats, n_iters, bad): bad[t] iff the draw's rotation
    is non-finite, or non-orthogonal beyond orth_tol (checked at HIGHEST
    matmul precision), or any of its inner QDWH solves did not converge.
    Inner convergence alone is insufficient: a zero B can yield W=0,
    which is not orthogonal.

    q < 2 or p < 2 returns the input with an identity rotation and zero
    counts (same shortcut as the SVD path; p = 1 makes B identically zero).
    """
    t, p, q = loadings.shape
    if q < 2 or p < 2:  # same identity shortcut as the SVD path
        eye = jnp.broadcast_to(jnp.eye(q, dtype=loadings.dtype), (t, q, q))
        return loadings, eye, jnp.zeros((t,), dtype=jnp.int32), jnp.zeros((t,), bool)
    with jax.default_matmul_precision("highest"):
        rotated, rotmats, n_iters, ok_conv = jax.vmap(
            lambda x: _varimax_single_qdwh(x, eps, max_iter)
        )(loadings)
    eye = jnp.eye(q, dtype=loadings.dtype)
    with jax.default_matmul_precision("highest"):
        orth_err = jnp.linalg.norm(
            jnp.einsum("tji,tjk->tik", rotmats, rotmats) - eye, axis=(1, 2)
        )
    bad = (~jnp.isfinite(orth_err)) | (orth_err > orth_tol) | (~ok_conv)
    return rotated, rotmats, n_iters.astype(jnp.int32), bad


_VARIMAX_JIT = None
_VARIMAX_QDWH_JIT = None


def _varimax_svd_cached(loadings, eps, max_iter, return_n_iter):
    global _VARIMAX_JIT
    if _VARIMAX_JIT is None:
        _VARIMAX_JIT = jax.jit(
            _varimax_impl, static_argnames=("eps", "max_iter", "return_n_iter")
        )
    return _VARIMAX_JIT(loadings, eps=eps, max_iter=max_iter, return_n_iter=return_n_iter)


def _varimax_qdwh_cached(loadings, eps, max_iter, orth_tol):
    global _VARIMAX_QDWH_JIT
    if _VARIMAX_QDWH_JIT is None:
        _VARIMAX_QDWH_JIT = jax.jit(
            _varimax_qdwh_impl, static_argnames=("eps", "max_iter", "orth_tol")
        )
    return _VARIMAX_QDWH_JIT(loadings, eps=eps, max_iter=max_iter, orth_tol=orth_tol)


def _varimax_qdwh_host(loadings, eps, max_iter, return_n_iter, orth_tol):
    """Run QDWH, then replace failed draws with SVD results.

    Fallback runs outside the jitted batch so its input size can depend on
    the failed-draw mask.
    """
    rotated, rotmats, n_iters, bad = _varimax_qdwh_cached(loadings, eps, max_iter, orth_tol)
    if bool(jnp.any(bad)):
        idx = np.nonzero(np.asarray(bad))[0]
        sub = jnp.asarray(loadings)[idx]  # host gather; dynamic size is fine here
        r_svd, t_svd, n_svd = _varimax_svd_cached(sub, eps, max_iter, True)
        rotated = rotated.at[idx].set(r_svd)
        rotmats = rotmats.at[idx].set(t_svd)
        n_iters = n_iters.at[idx].set(n_svd)
    if return_n_iter:
        return rotated, rotmats, n_iters
    return rotated, rotmats


def _require_positive_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _require_nonnegative_real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite non-negative scalar, got {value!r}")
    value_f = float(value)
    if not np.isfinite(value_f) or value_f < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return value_f


def varimax_batch(loadings: jax.Array, eps: float = 1e-5, max_iter: int = 1000,
                  return_n_iter: bool = False, polar: str = "qdwh",
                  orth_tol: float | None = None):
    """(T, p, q) -> ((T, p, q) rotated, (T, q, q) rotation matrices); with
    return_n_iter=True also the (T,) int32 per-draw iteration counts.

    q < 2 or p < 2 returns the input and identity rotations with zero counts
    (as R does for ncol < 2; p = 1 makes the varimax gradient identically
    zero, so the identity count is exact rather than an iteration limit).
    q > p makes the gradient B rank-deficient (rank <= p < q), so the polar
    factor is not unique and the returned rotation is one of many; the
    factor-analysis model itself assumes q < p.

    polar: "qdwh" (default) uses jax.lax.linalg.qdwh with a degenerate-matrix
    guard and host-side SVD fallback; orth_tol defaults to 1e-3 in float32
    and 1e-9 in float64. "svd" uses the SVD iteration from R's varimax.

    Loadings must be float32 or float64. Zeros and non-finite draws keep an
    identity rotation; non-finite draws also return a count of zero. Scaling
    uses exact IEEE exponent shifts, so subnormal (and extreme normal-range)
    draws are processed at a normal scale and restored afterwards instead of
    being flushed to zero.

    Raises ValueError for non-positive (or bool) max_iter, negative or
    non-finite eps, non-finite/negative orth_tol, non-float32/float64 input,
    invalid polar, or non-finite return_n_iter.

    Each polar path reuses a cached jitted wrapper, with iteration settings
    captured as static arguments. QDWH fallback selection runs on the host.
    """
    loadings = jnp.asarray(loadings)
    if loadings.ndim != 3:
        raise ValueError(f"expected (T, p, q), got {loadings.shape}")
    if any(size < 1 for size in loadings.shape):
        raise ValueError(f"empty loadings shape {loadings.shape} is not supported")
    if loadings.dtype not in (jnp.float32, jnp.float64):
        raise ValueError(
            f"varimax_batch requires float32 or float64 loadings, got {loadings.dtype}"
        )
    max_iter = _require_positive_int("max_iter", max_iter)
    eps = _require_nonnegative_real("eps", eps)
    if orth_tol is not None:
        orth_tol = _require_nonnegative_real("orth_tol", orth_tol)
    if not isinstance(return_n_iter, (bool, np.bool_)):
        raise ValueError(f"return_n_iter must be a bool, got {return_n_iter!r}")
    if polar not in ("svd", "qdwh"):
        raise ValueError(f"unknown polar {polar!r}")
    if polar == "svd":
        return _varimax_svd_cached(loadings, eps, max_iter, return_n_iter)
    if orth_tol is None:
        orth_tol = _default_orth_tol(loadings.dtype)
    return _varimax_qdwh_host(loadings, eps, max_iter, return_n_iter, orth_tol)
