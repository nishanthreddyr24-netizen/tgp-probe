import random

import pytest

from tgp_probe import PAPER_FEATURES, FULL_FEATURES, TGPProbe
from tgp_probe.calibrator import Calibrator

from .conftest import WORDS

CTX = "the company amazon was founded by bezos in the city"
GOOD = "amazon was founded by bezos"
BAD = "music video from japan with dj equipment"


def _sentence(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def test_extract_returns_all_features(probe):
    feats = probe.extract(CTX, GOOD)
    assert sorted(feats) == sorted(FULL_FEATURES)


def test_score_without_calibrator_returns_features_only(probe):
    r = probe.score(CTX, GOOD)
    assert r.score is None and r.is_hallucination is None
    assert r.n_response_tokens == 5
    assert r.n_context_tokens == len(CTX.split()) + 1  # + BOS
    assert r.latency_ms > 0


def test_batch_matches_single(probe):
    """Right padding must not change any sample's features."""
    rng = random.Random(0)
    ctxs = [_sentence(rng, rng.randint(3, 30)) for _ in range(7)]
    resps = [_sentence(rng, rng.randint(1, 12)) for _ in range(7)]
    batched = probe.extract_batch(ctxs, resps, batch_size=4)
    for c, r, b in zip(ctxs, resps, batched):
        single = probe.extract(c, r)
        for k in single:
            assert b[k] == pytest.approx(single[k], rel=1e-3, abs=1e-4), k


def test_truncation(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    probe = TGPProbe(model, tok, max_context_tokens=5, max_response_tokens=3)
    r = probe.score(CTX, BAD)
    assert r.context_truncated and r.response_truncated
    assert r.n_context_tokens == 5 and r.n_response_tokens == 3


def test_return_tokens(probe):
    r = probe.score(CTX, GOOD, return_tokens=True)
    assert [t.token for t in r.tokens] == GOOD.split()


@pytest.mark.parametrize("ctx,resp", [("", GOOD), ("   ", GOOD), (CTX, "")])
def test_empty_inputs_raise(probe, ctx, resp):
    with pytest.raises(ValueError):
        probe.score(ctx, resp)


def test_fit_then_score_and_reload(probe, tmp_path, tiny_model_and_tokenizer):
    rng = random.Random(1)
    ctxs, resps, labels = [], [], []
    for i in range(40):
        c = _sentence(rng, 15)
        ctxs.append(c)
        # grounded = copied span of the context; hallucinated = random words
        resps.append(" ".join(c.split()[3:9]) if i % 2 == 0 else _sentence(rng, 6))
        labels.append(i % 2)
    cal = probe.fit(ctxs, resps, labels, show_progress=False)
    assert cal.feature_names == PAPER_FEATURES
    assert cal.metadata["model"] == "tiny-llama"

    r = probe.score(CTX, GOOD)
    assert 0.0 <= r.score <= 1.0 and isinstance(r.is_hallucination, bool)

    path = cal.save(tmp_path / "cal.json")
    model, tok = tiny_model_and_tokenizer
    reloaded = TGPProbe(model, tok, calibrator=str(path), model_name="tiny-llama")
    assert reloaded.score(CTX, GOOD).score == pytest.approx(r.score)


def test_calibrator_model_mismatch_warns(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    cal = Calibrator(["NCN"], [0.0], [1.0], [1.0], 0.0, metadata={"model": "other/model"})
    with pytest.warns(UserWarning, match="fitted on"):
        TGPProbe(model, tok, calibrator=cal, model_name="tiny-llama")


def test_auto_calibrator_without_bundle_is_none(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    assert TGPProbe(model, tok, calibrator="auto", model_name="no/such-model").calibrator is None
