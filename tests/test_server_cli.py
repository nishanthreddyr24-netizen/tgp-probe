import pytest

from tgp_probe.calibrator import Calibrator
from tgp_probe.cli import build_parser, main

CTX = "the company amazon was founded by bezos in the city"


@pytest.fixture()
def client(probe):
    from fastapi.testclient import TestClient

    from tgp_probe.server import create_app

    probe.calibrator = Calibrator(
        ["Full_Curv", "Middle_Curv", "NCN"], [0.0] * 3, [1.0] * 3, [1.0, 1.0, -1.0], 0.0
    )
    return TestClient(create_app(probe))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["calibrated"] is True


def test_score(client):
    body = client.post("/score", json={"context": CTX, "response": "amazon", "return_tokens": True}).json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["tokens"][0]["token"] == "amazon"


def test_score_batch(client):
    items = [{"context": CTX, "response": r} for r in ["amazon", "music video", "bezos founded"]]
    body = client.post("/score/batch", json={"items": items}).json()
    assert [b["n_response_tokens"] for b in body] == [1, 2, 2]


def test_empty_context_is_422(client):
    assert client.post("/score", json={"context": "", "response": "x"}).status_code == 422


def test_cli_parser_and_calibrators(capsys):
    args = build_parser().parse_args(["score", "--context", "a", "--response", "b"])
    assert args.calibrator == "auto"
    args = build_parser().parse_args(["train", "--out", "x.json"])
    assert args.calibrator == "none" and args.features == "paper"
    assert main(["calibrators"]) == 0
    assert "qwen--qwen2.5-0.5b-instruct" in capsys.readouterr().out
