"""Logistic calibrator that maps TGP features to a hallucination probability.

Inference is pure numpy: a fitted calibrator is a small JSON file, so serving
does not need scikit-learn. Fitting does need it (``pip install tgp-probe[train]``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

FORMAT_VERSION = 1


def _clean(X: np.ndarray) -> np.ndarray:
    # Same sanitisation as the research scripts.
    return np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=10.0, neginf=-10.0)


@dataclass
class Calibrator:
    feature_names: List[str]
    mean: List[float]
    scale: List[float]
    coef: List[float]
    intercept: float
    threshold: float = 0.5
    metadata: Dict[str, object] = field(default_factory=dict)

    # ── inference ──────────────────────────────────────────────────────────
    def _matrix(self, feats) -> np.ndarray:
        if isinstance(feats, Mapping):
            feats = [feats]
        if len(feats) and isinstance(feats[0], Mapping):
            missing = [n for n in self.feature_names if n not in feats[0]]
            if missing:
                raise KeyError(f"features missing for calibrator: {missing}")
            feats = [[f[n] for n in self.feature_names] for f in feats]
        X = _clean(np.atleast_2d(np.asarray(feats, dtype=np.float64)))
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {X.shape[1]}")
        return X

    def predict_proba(self, feats) -> np.ndarray:
        """P(hallucinated) for one feature dict, a list of dicts, or an (N, F) array."""
        X = self._matrix(feats)
        z = (X - np.asarray(self.mean)) / np.asarray(self.scale)
        logits = z @ np.asarray(self.coef) + self.intercept
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, feats) -> np.ndarray:
        return self.predict_proba(feats) >= self.threshold

    # ── fitting ────────────────────────────────────────────────────────────
    @classmethod
    def fit(
        cls,
        X: Union[np.ndarray, Sequence[Sequence[float]]],
        y: Sequence[int],
        feature_names: Sequence[str],
        C: float = 1.0,
        threshold: Union[float, str] = "youden",
        metadata: Optional[Dict[str, object]] = None,
    ) -> "Calibrator":
        """Fit standardisation and a class-balanced L2 logistic regression.

        ``threshold="youden"`` picks the cut-off that maximises TPR - FPR on the
        training data. Pass a float to fix it instead.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_curve
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:  # pragma: no cover
            raise ImportError("Fitting needs scikit-learn: pip install 'tgp-probe[train]'") from e

        X = _clean(X)
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) != 2:
            raise ValueError("labels must contain both classes (0 = grounded, 1 = hallucinated)")

        scaler = StandardScaler().fit(X)
        scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
        clf = LogisticRegression(class_weight="balanced", C=C, max_iter=1000, random_state=42)
        clf.fit((X - scaler.mean_) / scale, y)

        cal = cls(
            feature_names=list(feature_names),
            mean=scaler.mean_.tolist(),
            scale=scale.tolist(),
            coef=clf.coef_[0].tolist(),
            intercept=float(clf.intercept_[0]),
            metadata=dict(metadata or {}),
        )
        if threshold == "youden":
            fpr, tpr, thr = roc_curve(y, cal.predict_proba(X))
            cal.threshold = float(np.clip(thr[int(np.argmax(tpr - fpr))], 0.0, 1.0))
        else:
            cal.threshold = float(threshold)
        return cal

    # ── persistence ────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, object]:
        return {"format_version": FORMAT_VERSION, **asdict(self)}

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "Calibrator":
        d = dict(d)
        version = d.pop("format_version", FORMAT_VERSION)
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported calibrator format_version {version}")
        return cls(**d)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Calibrator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ── bundled calibrators ─────────────────────────────────────────────────────

BUNDLED_DIR = Path(__file__).parent / "calibrators"


def bundled_name(model_name: str) -> str:
    """'Qwen/Qwen2.5-3B-Instruct' -> 'qwen--qwen2.5-3b-instruct'"""
    return model_name.strip("/").replace("/", "--").lower()


def find_bundled(model_name: str) -> Optional[Path]:
    p = BUNDLED_DIR / f"{bundled_name(model_name)}.json"
    return p if p.is_file() else None


def list_bundled() -> List[str]:
    return sorted(p.stem for p in BUNDLED_DIR.glob("*.json"))
