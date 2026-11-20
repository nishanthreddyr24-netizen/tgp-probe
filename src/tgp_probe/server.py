"""HTTP service for real-time verification (``pip install tgp-probe[server]``).

    tgp-probe serve --model Qwen/Qwen2.5-3B-Instruct --calibrator calib.json

    curl -X POST localhost:8000/score -H 'content-type: application/json' \
         -d '{"context": "...", "response": "..."}'
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover
    raise ImportError("The server needs FastAPI: pip install 'tgp-probe[server]'") from e

from . import __version__
from .probe import TGPProbe


class ScoreRequest(BaseModel):
    context: str = Field(..., description="Source text the response must be grounded in.")
    response: str = Field(..., description="Text to verify, e.g. a RAG answer or summary.")
    return_tokens: bool = False


class BatchScoreRequest(BaseModel):
    items: List[ScoreRequest] = Field(..., max_length=256)
    batch_size: int = Field(8, ge=1, le=64)


class TokenOut(BaseModel):
    token: str
    curvature: float
    ncn: float


class ScoreResponse(BaseModel):
    score: Optional[float]
    is_hallucination: Optional[bool]
    features: Dict[str, float]
    latency_ms: float
    n_context_tokens: int
    n_response_tokens: int
    context_truncated: bool
    response_truncated: bool
    tokens: Optional[List[TokenOut]] = None


def create_app(probe: TGPProbe) -> FastAPI:
    app = FastAPI(title="TGP Probe", version=__version__,
                  description="Groundedness verification from hidden-state trajectory geometry.")
    # One GPU model shared by all requests: serialise forward passes.
    lock = threading.Lock()

    def _run(contexts, responses, batch_size, return_tokens):
        try:
            with lock:
                return probe.score_batch(contexts, responses, batch_size=batch_size,
                                         return_tokens=return_tokens)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.get("/health")
    def health():
        cal = probe.calibrator
        return {
            "status": "ok",
            "model": probe.model_name,
            "calibrated": cal is not None,
            "threshold": cal.threshold if cal else None,
            "features": cal.feature_names if cal else None,
            "max_context_tokens": probe.max_context_tokens,
            "max_response_tokens": probe.max_response_tokens,
        }

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest):
        return _run([req.context], [req.response], 1, req.return_tokens)[0].to_dict()

    @app.post("/score/batch", response_model=List[ScoreResponse])
    def score_batch(req: BatchScoreRequest):
        tokens = any(i.return_tokens for i in req.items)
        results = _run([i.context for i in req.items], [i.response for i in req.items],
                       req.batch_size, tokens)
        return [r.to_dict() for r in results]

    return app
