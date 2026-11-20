"""Run TGP on hand-written test cases and print what it decides.

    python examples/custom_cases.py                       # bundled Qwen2.5-0.5B calibrator
    python examples/custom_cases.py --model Qwen/Qwen2.5-3B-Instruct --calibrator calib.json
    python examples/custom_cases.py --markdown results.md  # also save a Markdown table

Every case is a (context, response) pair with the answer we expect:
"grounded" (supported by the context) or "hallucinated" (contradicts it, or
adds things the context does not support). Add your own to CASES.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

from tgp_probe import TGPProbe

AMAZON = ("Amazon was founded by Jeff Bezos in July 1994 in Bellevue, Washington. "
          "It started as an online bookstore before expanding into electronics, software and cloud computing.")
EIFFEL = ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
          "It was designed by Gustave Eiffel's company and completed in 1889 for the World's Fair. "
          "It is 330 metres tall.")
DRUG = ("Paracetamol: the usual adult dose is 500 mg to 1 g every 4 to 6 hours. "
        "Do not take more than 4 g in 24 hours. Overdose can cause serious liver damage.")
POLICY = ("Refund policy: customers can return unused items within 30 days of delivery for a full refund. "
          "Opened software and gift cards cannot be returned. Refunds are issued to the original payment method "
          "within 5 business days.")
ARTICLE = ("Heavy rain caused the River Severn to burst its banks on Tuesday, flooding more than 200 homes in "
           "Shrewsbury. The Environment Agency issued three severe flood warnings, and about 400 residents were "
           "moved to temporary shelters. No injuries were reported. Officials said water levels were expected "
           "to fall by the weekend.")


@dataclass
class Case:
    name: str
    kind: str
    context: str
    response: str
    expected: str  # "grounded" | "hallucinated"


def qa(question: str, passage: str) -> str:
    return f"Question: {question}\nPassage: {passage}"


def summ(document: str) -> str:
    return f"Document: {document}\nSummary:\n"


def dial(knowledge: str, history: str) -> str:
    return f"Knowledge: {knowledge}\nDialogue History: {history}\nResponse:\n"


CASES: List[Case] = [
    # ── question answering ────────────────────────────────────────────────
    Case("QA correct", "grounded", qa("Who founded Amazon?", AMAZON),
         "Jeff Bezos", "grounded"),
    Case("QA wrong person", "entity swap", qa("Who founded Amazon?", AMAZON),
         "Elon Musk", "hallucinated"),
    Case("QA correct year", "grounded", qa("When was the Eiffel Tower completed?", EIFFEL),
         "1889", "grounded"),
    Case("QA wrong year", "number change", qa("When was the Eiffel Tower completed?", EIFFEL),
         "1925", "hallucinated"),
    Case("QA wrong city", "entity swap", qa("Where is the Eiffel Tower?", EIFFEL),
         "London, England", "hallucinated"),
    # ── RAG-style answers ─────────────────────────────────────────────────
    Case("Dose correct", "grounded", qa("What is the maximum daily dose of paracetamol?", DRUG),
         "No more than 4 g in 24 hours.", "grounded"),
    Case("Dose wrong", "number change", qa("What is the maximum daily dose of paracetamol?", DRUG),
         "Up to 10 g in 24 hours is safe.", "hallucinated"),
    Case("Refund correct", "grounded", qa("How long do I have to return an item?", POLICY),
         "You can return unused items within 30 days of delivery.", "grounded"),
    Case("Refund made-up rule", "unsupported claim", qa("Can I return a gift card?", POLICY),
         "Yes, gift cards can be returned for store credit within 90 days.", "hallucinated"),
    # ── summarization ─────────────────────────────────────────────────────
    Case("Summary faithful", "grounded", summ(ARTICLE),
         "The River Severn flooded over 200 homes in Shrewsbury and about 400 residents were moved to shelters.",
         "grounded"),
    Case("Summary negated", "negation", summ(ARTICLE),
         "The River Severn flooded over 200 homes in Shrewsbury and several people were seriously injured.",
         "hallucinated"),
    Case("Summary off-topic", "off-topic drift", summ(ARTICLE),
         "The new smartphone features a faster processor and an improved camera system.",
         "hallucinated"),
    # ── dialogue ──────────────────────────────────────────────────────────
    Case("Dialogue grounded", "grounded",
         dial("Inception is a 2010 film directed by Christopher Nolan, starring Leonardo DiCaprio.",
              "[Human]: Do you know who directed Inception?"),
         "Yes, Inception was directed by Christopher Nolan.", "grounded"),
    Case("Dialogue wrong", "entity swap",
         dial("Inception is a 2010 film directed by Christopher Nolan, starring Leonardo DiCaprio.",
              "[Human]: Do you know who directed Inception?"),
         "Yes, Inception was directed by Steven Spielberg.", "hallucinated"),
]


def run(probe: TGPProbe, cases: List[Case]):
    rows = []
    for c in cases:
        r = probe.score(c.context, c.response, return_tokens=True)
        verdict = "hallucinated" if r.is_hallucination else "grounded"
        weakest = min(r.tokens, key=lambda t: t.ncn).token.strip() if r.tokens else ""
        rows.append({
            "case": c.name, "kind": c.kind, "response": c.response, "expected": c.expected,
            "score": r.score, "verdict": verdict, "ok": verdict == c.expected,
            "curv": r.features["Full_Curv"], "mid_curv": r.features["Middle_Curv"],
            "ncn": r.features["NCN"], "weakest": weakest, "ms": r.latency_ms,
        })
    return rows


def print_table(rows, threshold):
    print(f"\nthreshold = {threshold:.3f}   (score >= threshold -> hallucinated)\n")
    print(f"{'case':<22}{'expected':<14}{'score':>7}  {'verdict':<14}{'':3}{'curv':>7}{'ncn':>7}  weakest token")
    print("-" * 92)
    for r in rows:
        mark = "OK" if r["ok"] else "XX"
        print(f"{r['case']:<22}{r['expected']:<14}{r['score']:>7.3f}  {r['verdict']:<14}{mark:<3}"
              f"{r['curv']:>7.3f}{r['ncn']:>7.3f}  {r['weakest']!r}")
    n_ok = sum(r["ok"] for r in rows)
    print(f"\n{n_ok}/{len(rows)} correct")


def to_markdown(rows, model, threshold) -> str:
    lines = [
        f"Model: `{model}` · threshold {threshold:.3f} · "
        f"{sum(r['ok'] for r in rows)}/{len(rows)} correct\n",
        "| Case | Response | Expected | Score | TGP says | | Curv | NCN | Weakest token |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} ({r['kind']}) | {r['response']} | {r['expected']} | {r['score']:.3f} | "
            f"{r['verdict']} | {'✅' if r['ok'] else '❌'} | {r['curv']:.3f} | {r['ncn']:.3f} | `{r['weakest']}` |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--calibrator", default="auto")
    ap.add_argument("--device", default=None)
    ap.add_argument("--markdown", default=None, help="write results as a Markdown table")
    args = ap.parse_args()

    probe = TGPProbe.from_pretrained(args.model, calibrator=args.calibrator, device=args.device)
    if probe.calibrator is None:
        raise SystemExit("No calibrator: pass --calibrator path.json (train one with `tgp-probe train`).")
    rows = run(probe, CASES)
    print_table(rows, probe.calibrator.threshold)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(to_markdown(rows, args.model, probe.calibrator.threshold))
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
