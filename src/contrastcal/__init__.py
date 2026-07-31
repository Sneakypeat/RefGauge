"""ContrastCal public API."""

from .calibration import calibrate_loso, calibration_summary
from .meta import MetaAnalysisResult, random_effects
from .reference import audit_references

__all__ = [
    "MetaAnalysisResult",
    "audit_references",
    "calibrate_loso",
    "calibration_summary",
    "random_effects",
]
__version__ = "0.1.0"
