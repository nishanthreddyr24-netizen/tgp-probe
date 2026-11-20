"""Trajectory Geometry Probe (TGP): groundedness verification from hidden-state geometry.

    from tgp_probe import TGPProbe

    probe = TGPProbe.from_pretrained("Qwen/Qwen2.5-3B-Instruct", calibrator="calib.json")
    result = probe.score(context="...", response="...")
    print(result.score, result.is_hallucination)
"""

__version__ = "0.1.0"

from .calibrator import Calibrator, list_bundled  # noqa: E402
from .features import FEATURE_SETS, FULL_FEATURES, PAPER_FEATURES, compute_features  # noqa: E402
from .probe import TGPProbe, TGPResult, TokenTrace  # noqa: E402

__all__ = [
    "TGPProbe",
    "TGPResult",
    "TokenTrace",
    "Calibrator",
    "compute_features",
    "list_bundled",
    "FEATURE_SETS",
    "PAPER_FEATURES",
    "FULL_FEATURES",
    "__version__",
]
