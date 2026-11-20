import math

import pytest
import torch

from tgp_probe.features import (
    FULL_FEATURES,
    compute_features,
    resolve_feature_set,
    token_curvature,
    token_ncn_per_layer,
)


def test_straight_line_has_zero_curvature():
    direction = torch.randn(16)
    traj = torch.stack([direction * (i + 1) for i in range(8)])[:, None, :]  # (8, 1, 16)
    assert token_curvature(traj).item() == pytest.approx(0.0, abs=1e-4)


def test_bent_trajectory_has_positive_curvature():
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])[:, None, :]
    assert token_curvature(pts).item() > 0.1


def test_ncn_is_one_when_response_copies_context():
    ctx = torch.randn(3, 5, 8)
    gen = ctx[:, 2:4, :].clone()
    ncn = token_ncn_per_layer(gen, ctx)
    assert torch.allclose(ncn, torch.ones_like(ncn), atol=1e-5)


def test_compute_features_keys_and_finiteness():
    hidden = torch.randn(13, 20, 32)  # L=12 layers + embedding
    feats = compute_features(hidden, ctx_len=12)
    assert list(feats) == FULL_FEATURES
    assert all(math.isfinite(v) for v in feats.values())


def test_compute_features_accepts_half_precision_and_token_traces():
    hidden = torch.randn(7, 10, 16).half()
    feats = compute_features(hidden, ctx_len=6, return_token_traces=True)
    assert len(feats["_tokens"]["curvature"]) == 4
    assert len(feats["_tokens"]["ncn"]) == 4


@pytest.mark.parametrize("ctx_len", [0, 10])
def test_compute_features_rejects_bad_split(ctx_len):
    with pytest.raises(ValueError):
        compute_features(torch.randn(7, 10, 16), ctx_len=ctx_len)


def test_resolve_feature_set():
    assert resolve_feature_set("paper") == ["Full_Curv", "Middle_Curv", "NCN"]
    assert resolve_feature_set(["NCN"]) == ["NCN"]
    with pytest.raises(ValueError):
        resolve_feature_set("nope")
    with pytest.raises(ValueError):
        resolve_feature_set(["NotAFeature"])
