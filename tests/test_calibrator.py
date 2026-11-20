import numpy as np
import pytest

from tgp_probe.calibrator import Calibrator, bundled_name


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, 3)) + y[:, None] * np.array([2.0, 1.0, -1.5])
    return X, y


def test_fit_matches_sklearn_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y = _data()
    cal = Calibrator.fit(X, y, ["a", "b", "c"])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42).fit(sc.transform(X), y)
    np.testing.assert_allclose(cal.predict_proba(X), clf.predict_proba(sc.transform(X))[:, 1], atol=1e-6)
    assert 0.0 <= cal.threshold <= 1.0


def test_dict_input_and_round_trip(tmp_path):
    X, y = _data()
    cal = Calibrator.fit(X, y, ["a", "b", "c"], metadata={"model": "m"})
    path = cal.save(tmp_path / "cal.json")
    loaded = Calibrator.load(path)
    p1 = cal.predict_proba({"a": 1.0, "b": 0.5, "c": -1.0, "extra": 9.0})
    p2 = loaded.predict_proba([[1.0, 0.5, -1.0]])
    np.testing.assert_allclose(p1, p2)
    assert loaded.metadata == {"model": "m"}


def test_errors():
    X, y = _data()
    cal = Calibrator.fit(X, y, ["a", "b", "c"])
    with pytest.raises(KeyError):
        cal.predict_proba({"a": 1.0})
    with pytest.raises(ValueError):
        Calibrator.fit(X, np.zeros(len(y)), ["a", "b", "c"])


def test_nan_and_inf_are_sanitised():
    X, y = _data()
    cal = Calibrator.fit(X, y, ["a", "b", "c"])
    p = cal.predict_proba([[np.nan, np.inf, -np.inf]])
    assert np.isfinite(p).all()


def test_bundled_name():
    assert bundled_name("Qwen/Qwen2.5-3B-Instruct") == "qwen--qwen2.5-3b-instruct"
