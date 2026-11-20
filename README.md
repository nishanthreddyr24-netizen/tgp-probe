# Trajectory Geometry Probe (TGP)

Fast, training-free **groundedness verification** for RAG and summarization outputs, built on the geometry of LLM hidden-state trajectories.

TGP teacher-forces a small evaluator model over `context + response` and measures how the response tokens' hidden states move through the layers. One forward pass, no text generation, no LLM-as-a-judge API calls.

```python
from tgp_probe import TGPProbe

probe = TGPProbe.from_pretrained("Qwen/Qwen2.5-3B-Instruct", calibrator="calib.json")

r = probe.score(
    context="Amazon was founded by Jeff Bezos in 1994 in Bellevue, Washington.",
    response="Amazon was founded by Elon Musk.",
)
r.score             # 0.0-1.0, probability the response is NOT grounded in the context
r.is_hallucination  # score >= calibrated threshold
r.features          # raw geometric features (always available)
r.latency_ms
```

## Install

```bash
pip install "tgp-probe[all] @ git+https://github.com/nishanthreddyr24-netizen/tgp-probe"
```

| Extra | Adds | Needed for |
|---|---|---|
| *(none)* | torch, transformers, numpy | scoring from Python |
| `[server]` | fastapi, uvicorn | `tgp-probe serve` HTTP API |
| `[train]` | scikit-learn, datasets, tqdm | fitting a calibrator |
| `[all]` | everything above | |

Requires Python ≥ 3.9, PyTorch ≥ 2.1 and transformers ≥ 4.45. A GPU is strongly recommended; CPU works for small evaluators.

## When to use it

**Use TGP when you have a source text and a piece of text that claims to be based on it, and you want a fast check of whether it actually is:**

- **RAG guardrail:** after your LLM answers from retrieved documents, score `(retrieved docs, answer)` before showing the answer to the user. Block, flag or regenerate when `is_hallucination` is true.
- **Summarization QA:** score `(article, summary)` to catch summaries that add or flip facts.
- **Bulk auditing / evaluation:** score thousands of logged `(context, output)` pairs offline as a cheap replacement for LLM-as-a-judge.
- **Monitoring:** run `tgp-probe serve` next to your app and log scores to watch groundedness over time.

**Do not use it for:**

- **Checking a model's own output with the same model.** When a model scores its *own* generations, AUROC collapses to about 0.56 because it is not "surprised" by its own tokens (the *Autoregressive Generation Gap*, see the paper). Always use an evaluator that is different from the generator, for example GPT-4 or Llama-70B generating and Qwen2.5-3B verifying.
- **Fact-checking without a context.** TGP measures grounding *in the context you pass*. It does not know world facts.
- **Very long responses as a single unit.** Responses are cut at `max_response_tokens` (150 by default). Split long outputs into sentences or paragraphs and score each one.

## Quick start: 3 steps

**1. Get a calibrator** (once per evaluator model; needs a GPU, and a free Colab T4 is enough):

```bash
tgp-probe train --model Qwen/Qwen2.5-3B-Instruct --out calib.json
```

This downloads HaluEval, extracts features, fits the calibrator, prints held-out AUROC and saves `calib.json`. Skip this step if the package already ships a calibrator for your model (`tgp-probe calibrators`).

**2. Call it from Python:**

```python
from tgp_probe import TGPProbe

probe = TGPProbe.from_pretrained("Qwen/Qwen2.5-3B-Instruct", calibrator="calib.json")  # load ONCE

def answer_with_guardrail(question, docs, llm):
    answer = llm(question, docs)                               # any generator
    check = probe.score(context=f"Question: {question}\nPassage: {docs}", response=answer)
    if check.is_hallucination:
        return "I couldn't find a reliable answer in the documents."
    return answer
```

**3. Or call it over HTTP** from any language:

```bash
tgp-probe serve --model Qwen/Qwen2.5-3B-Instruct --calibrator calib.json --port 8000
```

```python
import requests
r = requests.post("http://localhost:8000/score",
                  json={"context": docs, "response": answer}).json()
r["score"], r["is_hallucination"]
```

**Tips**

- Load the probe once at startup and reuse it; loading the model is the slow part, scoring is a single forward pass.
- Format `context` the way the calibrator was trained: `Question: ...\nPassage: ...` for QA, `Document: ...\nSummary:\n` for summaries. A calibrator fitted on your own data with `probe.fit(...)` works best of all.
- `r.score` is a probability; `r.is_hallucination` uses the threshold stored in the calibrator. Pick your own cut-off if you need fewer false alarms or fewer misses.

## Probe physics

For response token *t*, the trajectory is γₜ = (hₜ⁽⁰⁾, …, hₜ⁽ᴸ⁾) through the residual stream.

| Feature | Detects | Definition |
|---|---|---|
| **Curvature** κ (Full, Middle zone) | *Intrinsic* conflict, where the response contradicts the context | ‖v ∧ a‖ / ‖v‖³ with layer velocity v and acceleration a |
| **Nearest Context Neighbour** (NCN) | *Extrinsic* drift, where the response leaves the context's topic | early-layer max cosine similarity to any context token |
| NVR, Dispersion, MDR, Virial | auxiliary signals from the multi-zone experiments | see `src/tgp_probe/features.py` |

Curvature and NCN are complementary. Curvature wins on subtle factual flips (HaluEval-Sum), NCN wins on off-topic drift (XSum). The default calibrator is a logistic regression over the paper's three features `[max κ_full, max κ_mid, min NCN]`. Pass `feature_set="full"` to use all 12.

## Calibrators

Raw features are training-free. Turning them into a probability needs a small logistic **calibrator**, fitted **per evaluator model** because feature scales differ between models. A calibrator is a JSON file of a few hundred bytes; scoring with it needs only numpy.

```bash
# On a GPU: fit on HaluEval QA/Sum/Dialogue, report held-out AUROC, save the calibrator
tgp-probe train --model Qwen/Qwen2.5-3B-Instruct --out calib.json
tgp-probe train --model Qwen/Qwen2.5-3B-Instruct --datasets halueval_summarization xsum frank --out calib.json
```

Or fit on your own labelled data (1 = hallucinated, 0 = grounded):

```python
probe = TGPProbe.from_pretrained("Qwen/Qwen2.5-3B-Instruct", calibrator=None)
probe.fit(contexts, responses, labels)        # attaches the calibrator to the probe
probe.calibrator.save("my_domain.json")
```

Calibrators committed to `src/tgp_probe/calibrators/` ship with the package and load automatically with `calibrator="auto"` (the default); see that folder's README for the naming rule.

## Python API

```python
probe = TGPProbe.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    calibrator="calib.json",     # path | Calibrator | "auto" | None
    device="cuda",               # auto-detected; or pass device_map="auto" to shard
    max_context_tokens=2500,     # longer contexts keep their first token + the most recent tokens
    max_response_tokens=150,
)

probe.score(context, response, return_tokens=True)    # per-token curvature / NCN traces
probe.score_batch(contexts, responses, batch_size=8)  # padded, length-sorted batching
probe.extract(context, response)                      # features only

# Bring your own already-loaded model
probe = TGPProbe(model, tokenizer, calibrator="calib.json")
```

## HTTP service

```bash
tgp-probe serve --model Qwen/Qwen2.5-3B-Instruct --calibrator calib.json --host 0.0.0.0 --port 8000
```

```bash
curl -X POST localhost:8000/score -H 'content-type: application/json' \
  -d '{"context": "Amazon was founded by Jeff Bezos.", "response": "Elon Musk founded Amazon."}'
```

Endpoints: `GET /health`, `POST /score`, `POST /score/batch` (up to 256 items). Interactive docs are at `/docs`.

## CLI

```bash
tgp-probe score --calibrator calib.json --context-file doc.txt --response "..." --tokens
cat answer.txt | tgp-probe score --calibrator calib.json --context-file doc.txt --response-file -
tgp-probe calibrators   # list bundled calibrators
```

## Benchmark results

Combined AUROC with the paper's three-feature calibrator (held-out test split):

| Evaluator | HaluEval-QA | HaluEval-Sum | FRANK | XSum |
|---|---|---|---|---|
| Llama-3.2-1B | 0.990 | 0.601 | 0.807 | 0.917 |
| Qwen2.5-1.5B | 0.995 | 0.738 | 0.784 | 0.881 |
| Qwen2.5-3B | 0.995 | 0.839 | 0.783 | 0.867 |
| Qwen2.5-7B | 0.983 | 0.867 | 0.758 | 0.955 |

Wider evaluators handle long-context summarization better. Teacher-forced NLL, by comparison, sits around 0.55–0.60 on HaluEval-Sum.

## Repository layout

```
src/tgp_probe/      installable package (features, calibrator, probe, CLI, server)
tests/              offline test suite (tiny in-memory model, no downloads)
experiments/        original research scripts (v2 multi-zone, v3 zone ablation)
paper.tex           paper source
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
