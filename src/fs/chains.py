"""Transform factor scores using accumulated orthogonal transforms."""
from __future__ import annotations

import jax.numpy as jnp


def transform_scores(scores, transforms):
    """Apply the accumulated orthogonal transforms to factor scores.

    scores: (T, n, q) or (n, q) row-stored factor vectors f (i.e. rows of the
    score matrix). transforms: (T, q, q) or (q, q) from RSPResult.transforms.

    Valid combinations: (n,q)+(q,q) -> (n,q); (T,n,q)+(T,q,q) or
    (T,n,q)+(q,q) -> (T,n,q); (n,q)+(T,q,q) broadcasts the single score
    matrix under every draw's transform -> (T,n,q).

    Row-stored convention: f' = f @ O. For column-vector scores F this is
    F' = Oᵀ F — the same operation on transposed storage. Prediction is
    invariant: (Λ O)(f O)ᵀ = Λ fᵀ.

    Raises ValueError for invalid transform shapes, mismatched q dimensions,
    or mismatched leading draw counts.
    """
    scores = jnp.asarray(scores)
    transforms = jnp.asarray(transforms)
    if transforms.ndim not in (2, 3) or transforms.shape[-1] != transforms.shape[-2]:
        raise ValueError(
            f"transforms must be (q, q) or (T, q, q), got {transforms.shape}"
        )
    q = transforms.shape[-1]
    if scores.ndim not in (2, 3) or scores.shape[-1] != q:
        raise ValueError(
            f"scores must be (n, q) or (T, n, q) with q={q}, got {scores.shape}"
        )
    if scores.ndim == 3 and transforms.ndim == 3:
        if scores.shape[0] != transforms.shape[0]:
            raise ValueError(
                "per-draw T mismatch: scores T="
                f"{scores.shape[0]}, transforms T={transforms.shape[0]}"
            )
    if scores.ndim == 2 and transforms.ndim == 2:
        return scores @ transforms
    if scores.ndim == 3 and transforms.ndim == 3:
        return scores @ transforms
    if scores.ndim == 3 and transforms.ndim == 2:
        return scores @ transforms[None]
    if scores.ndim == 2 and transforms.ndim == 3:
        # one shared score matrix under every draw's transform -> (T, n, q)
        return scores[None, :, :] @ transforms
    raise ValueError(
        f"scores {scores.shape} and transforms {transforms.shape}: valid combos are "
        "(n,q)+(q,q), (T,n,q)+(T,q,q), (T,n,q)+(q,q), (n,q)+(T,q,q)"
    )
