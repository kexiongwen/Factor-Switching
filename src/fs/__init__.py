"""RSP alignment of posterior loadings in Bayesian factor models using JAX.

Default precision is float32; precision="float64" enables double precision.
"""
from .api import rsp_exact
from .chains import transform_scores
from .io import from_flat, lambda_colnames, to_flat
from .types import RSPResult

__version__ = "0.1.0.dev0"

__all__ = [
    "rsp_exact",
    "transform_scores",
    "from_flat",
    "to_flat",
    "lambda_colnames",
    "RSPResult",
]
