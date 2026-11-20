"""Labelled benchmark loaders used to fit calibrators (``pip install tgp-probe[train]``).

Each loader returns a class-balanced list of ``(context, response, label)``
tuples, where ``label = 1`` means hallucinated / unfaithful. Prompt formats
match the research scripts in ``experiments/``.
"""

from __future__ import annotations

import json
import random
from typing import Callable, Dict, List, Tuple

Pair = Tuple[str, str, int]


def _balance(pairs: List[Pair], n: int, seed: int = 42) -> List[Pair]:
    random.Random(seed).shuffle(pairs)
    fac = [p for p in pairs if p[2] == 0][: n // 2]
    hal = [p for p in pairs if p[2] == 1][: n // 2]
    k = min(len(fac), len(hal))
    if k == 0:
        raise RuntimeError("no usable pairs for one of the classes")
    return fac[:k] + hal[:k]


def _hf(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("Training data needs `datasets`: pip install 'tgp-probe[train]'") from e
    return load_dataset(*args, **kwargs)


def halueval_qa(n: int = 500) -> List[Pair]:
    pairs: List[Pair] = []
    for row in _hf("pminervini/HaluEval", "qa", split="data"):
        ctx = f"Question: {row['question']}\nPassage: {row['knowledge']}"
        pairs += [(ctx, row["right_answer"], 0), (ctx, row["hallucinated_answer"], 1)]
    return _balance(pairs, n)


def halueval_summarization(n: int = 500) -> List[Pair]:
    pairs: List[Pair] = []
    for row in _hf("pminervini/HaluEval", "summarization", split="data"):
        ctx = f"Document: {row['document']}\nSummary:\n"
        pairs += [(ctx, row["right_summary"], 0), (ctx, row["hallucinated_summary"], 1)]
    return _balance(pairs, n)


def halueval_dialogue(n: int = 500) -> List[Pair]:
    pairs: List[Pair] = []
    for row in _hf("pminervini/HaluEval", "dialogue", split="data"):
        ctx = (f"Knowledge: {row['knowledge']}\n"
               f"Dialogue History: {row['dialogue_history']}\nResponse:\n")
        pairs += [(ctx, row["right_response"], 0), (ctx, row["hallucinated_response"], 1)]
    return _balance(pairs, n)


def frank(n: int = 500) -> List[Pair]:
    import urllib.request

    url = "https://raw.githubusercontent.com/artidoro/frank/main/data/human_annotations_sentence.json"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    pairs: List[Pair] = []
    for ex in data:
        if ex.get("split") != "valid":
            continue
        label = 0
        for sent in ex.get("summary_sentences_annotations", []):
            votes = [1 if a[0] == "NoE" else 0 for a in sent.values()]
            if not round(sum(votes) / max(1, len(votes))):
                label = 1  # majority of annotators found an error in this sentence
        if ex.get("article") and ex.get("summary"):
            pairs.append((f"Article: {ex['article']}\nSummary:\n", ex["summary"], label))
    return _balance(pairs, n)


def xsum(n: int = 500) -> List[Pair]:
    pairs: List[Pair] = []
    for row in _hf("orange-share/xsum-factuality", split="test"):
        faithful = str(row.get("label", "faithful")).lower() in {"faithful", "0", "consistent"}
        pairs.append((f"Article: {row['document']}\nSummary:\n", row["summary"], 0 if faithful else 1))
    return _balance(pairs, n)


LOADERS: Dict[str, Callable[[int], List[Pair]]] = {
    "halueval_qa": halueval_qa,
    "halueval_summarization": halueval_summarization,
    "halueval_dialogue": halueval_dialogue,
    "frank": frank,
    "xsum": xsum,
}
