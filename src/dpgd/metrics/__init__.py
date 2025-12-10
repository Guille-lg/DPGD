"""Evaluation metrics for DPGD."""

from .bertscore import compute_bertscore
from .profile_hit_rate import compute_phr, compute_phr_details
from .readability_es import compute_szigriszt_pazos
from .sari import compute_sari

__all__ = [
    "compute_sari",
    "compute_bertscore",
    "compute_szigriszt_pazos",
    "compute_phr",
    "compute_phr_details",
]

