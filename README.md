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

**Zero-setup option:** a calibrator for **`Qwen/Qwen2.5-0.5B-Instruct`** is bundled, so this works right after `pip install`, even on CPU:

```python
probe = TGPProbe.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")   # bundled calibrator loads automatically
```

It is the smallest evaluator and is good for trying TGP out. For real use, train a calibrator for a 3B+ model (see [Real test cases](#real-test-cases-what-it-actually-does)).

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

## Real test cases: what it actually does

[`examples/custom_cases.py`](examples/custom_cases.py) runs 14 hand-written cases: QA, a medical-dose and refund-policy RAG scenario, news summaries and dialogue. Each one is either grounded or a specific kind of hallucination (entity swap, number change, negation, made-up rule, off-topic drift). Add your own cases to the `CASES` list.

```bash
python examples/custom_cases.py                          # bundled Qwen2.5-0.5B calibrator
python examples/custom_cases.py --model Qwen/Qwen2.5-3B-Instruct --calibrator calib.json
```

### Calibrator quality (held-out HaluEval, 20% test split)

| Evaluator | HaluEval-QA | HaluEval-Sum | HaluEval-Dialogue |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct (bundled, trained on CPU) | 0.939 | 0.792 | 0.746 |

### Unedited output on the 14 custom cases (Qwen2.5-0.5B, CPU, ~0.2–1 s per case)

Threshold 0.605: a score at or above it is flagged as hallucinated. **8 / 14 correct.**

| Case | Response | Expected | Score | TGP says | | Curv | NCN |
|---|---|---|---|---|---|---|---|
| QA correct | Jeff Bezos | grounded | 0.051 | grounded | ✅ | 0.181 | 0.824 |
| QA wrong person | Elon Musk | hallucinated | 0.045 | grounded | ❌ | 0.191 | **0.369** |
| QA correct year | 1889 | grounded | 0.070 | grounded | ✅ | 0.189 | 0.651 |
| QA wrong year | 1925 | hallucinated | 0.070 | grounded | ❌ | 0.189 | 0.651 |
| QA wrong city | London, England | hallucinated | 0.097 | grounded | ❌ | 0.198 | 0.537 |
| Dose correct | No more than 4 g in 24 hours. | grounded | 0.586 | grounded | ✅ | 0.261 | 0.470 |
| Dose wrong | Up to 10 g in 24 hours is safe. | hallucinated | 0.730 | hallucinated | ✅ | 0.273 | 0.410 |
| Refund correct | You can return unused items within 30 days of delivery. | grounded | 0.650 | hallucinated | ❌ | 0.271 | 0.499 |
| Refund made-up rule | Yes, gift cards can be returned for store credit within 90 days. | hallucinated | 0.789 | hallucinated | ✅ | 0.281 | 0.372 |
| Summary faithful | The River Severn flooded over 200 homes in Shrewsbury and about 400 residents were moved to shelters. | grounded | 0.791 | hallucinated | ❌ | 0.278 | 0.458 |
| Summary negated | …and several people were seriously injured. | hallucinated | 0.839 | hallucinated | ✅ | 0.285 | 0.458 |
| Summary off-topic | The new smartphone features a faster processor and an improved camera system. | hallucinated | 0.741 | hallucinated | ✅ | 0.274 | **0.358** |
| Dialogue grounded | Yes, Inception was directed by Christopher Nolan. | grounded | 0.572 | grounded | ✅ | 0.255 | 0.361 |
| Dialogue wrong | Yes, Inception was directed by Steven Spielberg. | hallucinated | 0.587 | grounded | ❌ | 0.259 | 0.361 |

### What these results show

- **When the same context is compared, the hallucinated version usually scores higher:** dose 0.586 → 0.730, summary 0.791 → 0.839, dialogue 0.572 → 0.587. The raw signals also react as the paper predicts. NCN falls from 0.82 to 0.37 when "Jeff Bezos" becomes "Elon Musk", and the off-topic summary has the lowest NCN of all the summaries.
- **Absolute scores depend on response length and format.** The 0.5B calibrator relies almost entirely on curvature, and maximum curvature grows with the number of tokens. Short QA answers therefore score low and full sentences score high, whether they are right or not. That is why one shared threshold misclassifies items in both directions.
- **Answers that share a prefix look identical.** "1889" vs "1925" and "…directed by Nolan" vs "…by Spielberg" get the same features, because TGP keeps only the single most extreme token and that token is in the shared prefix. Only the later tokens differ.

### How to get better results

1. **Use a bigger evaluator.** The paper's 3B and 7B models reach 0.84–0.87 on HaluEval-Sum; 0.5B gets 0.79 here.
2. **Fit the calibrator on your own task and format** with `probe.fit(contexts, responses, labels)`. Even 100–200 labelled examples from your real traffic remove most of the length and format bias.
3. **Score short units:** one sentence or claim at a time instead of whole paragraphs.
4. **Look at per-token traces** with `return_tokens=True` to see which words caused a flag.

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
examples/           custom_cases.py: hand-written real-world test cases
experiments/        original research scripts (v2 multi-zone, v3 zone ablation)
paper.tex           paper source
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
