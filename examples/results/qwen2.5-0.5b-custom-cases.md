# Custom test case results: Qwen2.5-0.5B-Instruct

- **Command:** `python examples/custom_cases.py --device cpu --markdown <file>`
- **Calibrator:** bundled `qwen--qwen2.5-0.5b-instruct.json` (paper features: Full_Curv, Middle_Curv, NCN)
- **Calibrator held-out AUROC (HaluEval, 20% split):** QA 0.939 · Summarization 0.792 · Dialogue 0.746
- **Hardware:** laptop CPU, fp32, ~0.2–1 s per case
- **Threshold:** 0.605 (score ≥ threshold → hallucinated)

## Summary

| | Count |
|---|---|
| Hallucinations caught | 4 / 7 |
| Grounded answers correctly passed | 4 / 7 |
| **Total correct** | **8 / 14** |

## Successful cases

| Case | Kind | Response | Score | Verdict | Curv | NCN |
|---|---|---|---|---|---|---|
| QA correct | grounded | Jeff Bezos | 0.051 | grounded | 0.181 | 0.824 |
| QA correct year | grounded | 1889 | 0.070 | grounded | 0.189 | 0.651 |
| Dose correct | grounded | No more than 4 g in 24 hours. | 0.586 | grounded | 0.261 | 0.470 |
| Dose wrong | number change | Up to 10 g in 24 hours is safe. | 0.730 | hallucinated | 0.273 | 0.410 |
| Refund made-up rule | unsupported claim | Yes, gift cards can be returned for store credit within 90 days. | 0.789 | hallucinated | 0.281 | 0.372 |
| Summary negated | negation | The River Severn flooded over 200 homes in Shrewsbury and several people were seriously injured. | 0.839 | hallucinated | 0.285 | 0.458 |
| Summary off-topic | off-topic drift | The new smartphone features a faster processor and an improved camera system. | 0.741 | hallucinated | 0.274 | 0.358 |
| Dialogue grounded | grounded | Yes, Inception was directed by Christopher Nolan. | 0.572 | grounded | 0.255 | 0.361 |

## Missed cases

| Case | Kind | Response | Expected | Score | Verdict | Curv | NCN |
|---|---|---|---|---|---|---|---|
| QA wrong person | entity swap | Elon Musk | hallucinated | 0.045 | grounded | 0.191 | 0.369 |
| QA wrong year | number change | 1925 | hallucinated | 0.070 | grounded | 0.189 | 0.651 |
| QA wrong city | entity swap | London, England | hallucinated | 0.097 | grounded | 0.198 | 0.537 |
| Refund correct | grounded | You can return unused items within 30 days of delivery. | grounded | 0.650 | hallucinated | 0.271 | 0.499 |
| Summary faithful | grounded | The River Severn flooded over 200 homes in Shrewsbury and about 400 residents were moved to shelters. | grounded | 0.791 | hallucinated | 0.278 | 0.458 |
| Dialogue wrong | entity swap | Yes, Inception was directed by Steven Spielberg. | hallucinated | 0.587 | grounded | 0.259 | 0.361 |

See the main README, section "What these results show", for why these were missed.

## Unit tests

`pytest`: **29 passed**. The suite covers the feature math, the calibrator, the probe API, padded batching against single-sample scoring, the HTTP server and the CLI.
