# Trajectory Geometry Probe (TGP)

Official code and paper repository for **"Detecting Hallucinations via the Internal Geometry of Cross-Layer Trajectories"** (Accepted to TMLR).

The Trajectory Geometry Probe (TGP) is an ultra-lightweight, unsupervised, $O(L \times D)$ algorithm for detecting grounding failures (hallucinations) in decoder-only language models by monitoring the physical geometry of cross-layer hidden state trajectories.

### The "Two-Engine" Zone Shift
TGP demonstrates that hallucination is not a monolithic phenomenon occurring at a single layer. Instead, it proves the "Two-Engine" theory:
- **Engine 1 (Early Layers):** Immediate conflict when performing simple context retrieval (e.g., Question Answering).
- **Engine 2 (Middle/Late Layers):** Deep conflict when performing complex synthesis (e.g., Long Document Summarization).

TGP successfully extracts features (Normalized Vector Radius, Curvature, Dispersion, etc.) across Early and Middle zones, allowing it to seamlessly catch hallucinations across varying cognitive loads and architectures (1.5B, 3B, 7B).

## Usage
Run the single-file benchmark suite:
```bash
python tgp_probe.py
```
