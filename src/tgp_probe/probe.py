"""High-level API: load an evaluator model once, then score (context, response) pairs."""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .calibrator import Calibrator, find_bundled
from .features import compute_features, features_to_vector, resolve_feature_set

DEFAULT_MAX_CONTEXT_TOKENS = 2500
DEFAULT_MAX_RESPONSE_TOKENS = 150

CalibratorLike = Union[None, str, Path, Calibrator]


@dataclass
class TokenTrace:
    token: str
    curvature: float
    ncn: float


@dataclass
class TGPResult:
    """Outcome of scoring one (context, response) pair."""

    #: P(response is NOT grounded in the context); ``None`` without a calibrator.
    score: Optional[float]
    #: ``score >= calibrator.threshold``; ``None`` without a calibrator.
    is_hallucination: Optional[bool]
    #: Raw geometric features (always present, training-free).
    features: Dict[str, float]
    latency_ms: float
    n_context_tokens: int
    n_response_tokens: int
    context_truncated: bool
    response_truncated: bool
    #: Per-response-token traces, if requested with ``return_tokens=True``.
    tokens: Optional[List[TokenTrace]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _Encoded:
    ids: List[int]
    ctx_len: int
    ctx_truncated: bool
    resp_truncated: bool


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_kwarg() -> str:
    """transformers>=4.56 renamed ``torch_dtype`` to ``dtype``."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


class TGPProbe:
    """Trajectory Geometry Probe: an offline groundedness verifier.

    A small evaluator LM is *teacher-forced* over ``context + response``, and
    TGP measures how the response tokens' hidden-state trajectories bend
    (curvature) and drift from the context manifold (NCN).

    The response should come from a *different* system (a RAG pipeline, an
    API model, a human). The paper shows that TGP cannot reliably flag the
    evaluator model's own autoregressive generations (the "generation gap").

    Example::

        probe = TGPProbe.from_pretrained("Qwen/Qwen2.5-3B-Instruct", calibrator="calib.json")
        r = probe.score(context=docs, response=answer)
        r.score, r.is_hallucination, r.features
    """

    def __init__(
        self,
        model,
        tokenizer,
        calibrator: CalibratorLike = None,
        *,
        model_name: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        max_response_tokens: Optional[int] = None,
        alpha: float = 1.0,
    ):
        self.model = model.eval()
        # Run the backbone only: the LM head's (seq x vocab) logits are not needed.
        self.backbone = getattr(model, "base_model", model)
        self.tokenizer = tokenizer
        self.model_name = model_name or getattr(model.config, "_name_or_path", None) or "unknown"
        self.alpha = alpha
        self.input_device = model.get_input_embeddings().weight.device

        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        self.pad_token_id = int(pad)

        self.calibrator = self._resolve_calibrator(calibrator)
        meta = self.calibrator.metadata if self.calibrator else {}
        self.max_context_tokens = int(
            max_context_tokens or meta.get("max_context_tokens") or DEFAULT_MAX_CONTEXT_TOKENS
        )
        self.max_response_tokens = int(
            max_response_tokens or meta.get("max_response_tokens") or DEFAULT_MAX_RESPONSE_TOKENS
        )
        if self.max_context_tokens < 2:
            raise ValueError("max_context_tokens must be >= 2")

    # ── construction ───────────────────────────────────────────────────────
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        calibrator: CalibratorLike = "auto",
        *,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        max_context_tokens: Optional[int] = None,
        max_response_tokens: Optional[int] = None,
        **model_kwargs,
    ) -> "TGPProbe":
        """Load a Hugging Face causal LM as the evaluator.

        Args:
            calibrator: path to a calibrator JSON, a :class:`Calibrator`,
                ``"auto"`` (use a calibrator bundled for this model, if any), or
                ``None`` (features only).
            device: ``"cuda"``, ``"mps"`` or ``"cpu"``; auto-detected by default.
                Pass ``device_map="auto"`` instead to shard a large model.
            dtype: defaults to fp16 on GPU/MPS (as in the paper) and fp32 on CPU.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        use_device_map = "device_map" in model_kwargs
        device = device or _default_device()
        if dtype is None:
            dtype = torch.float32 if device == "cpu" and not use_device_map else torch.float16

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model_kwargs[_dtype_kwarg()] = dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code, **model_kwargs
        )
        if not use_device_map:
            model = model.to(device)
        return cls(
            model,
            tokenizer,
            calibrator,
            model_name=model_name_or_path,
            max_context_tokens=max_context_tokens,
            max_response_tokens=max_response_tokens,
        )

    def _resolve_calibrator(self, calibrator: CalibratorLike) -> Optional[Calibrator]:
        if calibrator is None:
            return None
        if isinstance(calibrator, Calibrator):
            cal = calibrator
        elif isinstance(calibrator, str) and calibrator == "auto":
            path = find_bundled(self.model_name)
            if path is None:
                return None
            cal = Calibrator.load(path)
        else:
            cal = Calibrator.load(calibrator)
        trained_on = cal.metadata.get("model")
        if trained_on and trained_on != self.model_name:
            warnings.warn(
                f"Calibrator was fitted on {trained_on!r} but the evaluator is {self.model_name!r}; "
                "feature scales differ between models, so scores will be unreliable.",
                stacklevel=3,
            )
        return cal

    # ── encoding & forward ─────────────────────────────────────────────────
    def _encode(self, context: str, response: str) -> _Encoded:
        if not context or not context.strip():
            raise ValueError("context is empty; TGP measures grounding against a context")
        if not response or not response.strip():
            raise ValueError("response is empty")
        ctx = list(self.tokenizer(context, add_special_tokens=True)["input_ids"])
        resp = list(self.tokenizer(response, add_special_tokens=False)["input_ids"])
        if not resp:
            raise ValueError("response produced no tokens")

        ctx_trunc = len(ctx) > self.max_context_tokens
        if ctx_trunc:  # keep the first token (BOS / sink) + the most recent context
            ctx = ctx[:1] + ctx[-(self.max_context_tokens - 1):]
        resp_trunc = len(resp) > self.max_response_tokens
        resp = resp[: self.max_response_tokens]
        return _Encoded(ctx + resp, len(ctx), ctx_trunc, resp_trunc)

    @torch.inference_mode()
    def _hidden_states(self, batch: Sequence[_Encoded]) -> List[torch.Tensor]:
        """One padded forward pass -> per-sample ``(L+1, len_i, D)`` hidden states."""
        lengths = [len(e.ids) for e in batch]
        max_len = max(lengths)
        ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, e in enumerate(batch):  # right padding keeps real-token positions unchanged
            ids[i, : lengths[i]] = torch.tensor(e.ids)
            mask[i, : lengths[i]] = 1
        out = self.backbone(
            input_ids=ids.to(self.input_device),
            attention_mask=mask.to(self.input_device),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hs = out.hidden_states
        return [torch.stack([h[i, : lengths[i]] for h in hs]) for i in range(len(batch))]

    # ── public API ─────────────────────────────────────────────────────────
    def extract(self, context: str, response: str) -> Dict[str, float]:
        """Training-free geometric features for one pair."""
        return self.extract_batch([context], [response], batch_size=1)[0]

    def extract_batch(
        self,
        contexts: Sequence[str],
        responses: Sequence[str],
        batch_size: int = 8,
        show_progress: bool = False,
    ) -> List[Dict[str, float]]:
        return [r.features for r in self._run(contexts, responses, batch_size, False, show_progress)]

    def score(self, context: str, response: str, return_tokens: bool = False) -> TGPResult:
        """Score one pair. Higher ``score`` means the response is less grounded."""
        return self._run([context], [response], 1, return_tokens, False)[0]

    __call__ = score

    def score_batch(
        self,
        contexts: Sequence[str],
        responses: Sequence[str],
        batch_size: int = 8,
        return_tokens: bool = False,
        show_progress: bool = False,
    ) -> List[TGPResult]:
        return self._run(contexts, responses, batch_size, return_tokens, show_progress)

    def fit(
        self,
        contexts: Sequence[str],
        responses: Sequence[str],
        labels: Sequence[int],
        feature_set: Union[str, Sequence[str]] = "paper",
        C: float = 1.0,
        batch_size: int = 8,
        show_progress: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Calibrator:
        """Fit (and attach) a calibrator. ``labels``: 1 = hallucinated, 0 = grounded."""
        names = resolve_feature_set(feature_set)
        feats = self.extract_batch(contexts, responses, batch_size, show_progress)
        X = np.array([features_to_vector(f, names) for f in feats])
        meta = {
            "model": self.model_name,
            "dtype": str(next(self.model.parameters()).dtype),
            "feature_set": feature_set if isinstance(feature_set, str) else "custom",
            "n_train": int(len(labels)),
            "max_context_tokens": self.max_context_tokens,
            "max_response_tokens": self.max_response_tokens,
            **(metadata or {}),
        }
        self.calibrator = Calibrator.fit(X, labels, names, C=C, metadata=meta)
        return self.calibrator

    # ── internals ──────────────────────────────────────────────────────────
    def _run(
        self,
        contexts: Sequence[str],
        responses: Sequence[str],
        batch_size: int,
        return_tokens: bool,
        show_progress: bool,
    ) -> List[TGPResult]:
        if len(contexts) != len(responses):
            raise ValueError("contexts and responses must have the same length")
        encoded = [self._encode(c, r) for c, r in zip(contexts, responses)]
        # Length-sorted batches minimise padding.
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i].ids), reverse=True)
        chunks = [order[i : i + batch_size] for i in range(0, len(order), max(1, batch_size))]
        if show_progress:
            try:
                from tqdm.auto import tqdm

                chunks = tqdm(chunks, desc="TGP", unit="batch")
            except ImportError:
                pass

        results: List[Optional[TGPResult]] = [None] * len(encoded)
        for chunk in chunks:
            t0 = time.perf_counter()
            batch = [encoded[i] for i in chunk]
            hidden = self._hidden_states(batch)
            partial: List[Tuple[int, Dict[str, Any]]] = []
            for i, e, h in zip(chunk, batch, hidden):
                partial.append((i, compute_features(h, e.ctx_len, self.alpha, return_tokens)))
            ms = (time.perf_counter() - t0) * 1000 / len(chunk)
            for i, feats in partial:
                results[i] = self._make_result(encoded[i], feats, ms)
        return results  # type: ignore[return-value]

    def _make_result(self, e: _Encoded, feats: Dict[str, Any], ms: float) -> TGPResult:
        traces = feats.pop("_tokens", None)
        score = flag = None
        if self.calibrator is not None:
            score = float(self.calibrator.predict_proba(feats)[0])
            flag = bool(score >= self.calibrator.threshold)
        tokens = None
        if traces is not None:
            resp_ids = e.ids[e.ctx_len :]
            tokens = [
                TokenTrace(self.tokenizer.decode([tid]), float(k), float(n))
                for tid, k, n in zip(resp_ids, traces["curvature"], traces["ncn"])
            ]
        return TGPResult(
            score=score,
            is_hallucination=flag,
            features={k: float(v) for k, v in feats.items()},
            latency_ms=round(ms, 3),
            n_context_tokens=e.ctx_len,
            n_response_tokens=len(e.ids) - e.ctx_len,
            context_truncated=e.ctx_truncated,
            response_truncated=e.resp_truncated,
            tokens=tokens,
        )
