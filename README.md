# Trajectory Geometry Probe (TGP)

Official code and paper repository for **"Detecting Hallucinations via the Internal Geometry of Cross-Layer Trajectories"** (Accepted to TMLR).

The Trajectory Geometry Probe (TGP) is an ultra-lightweight, unsupervised, $O(L \times D)$ algorithm for detecting grounding failures (hallucinations) in decoder-only language models. Rather than relying on a secondary LLM for grading or analyzing attention heatmaps, TGP directly monitors the physical geometry of the hidden state vectors as they pass through the residual stream.

## Probe Physics

When a language model hallucinates (i.e., its generated output contradicts the provided context), it experiences a severe internal conflict. The attention heads attempt to retrieve the true context, while the MLP layers push pre-trained parametric memory. This internal "tug-of-war" physically warps the trajectory of the hidden states across the layers. 

TGP detects this warping by computing several geometric features directly on the cross-layer representation trajectory:

- **Normalized Vector Radius (NVR):** Hallucinating trajectories experience a massive collapse in magnitude relative to their variance as the model loses confidence.
- **Curvature ($\kappa$):** Conflicting vectors cause the trajectory to sharply zig-zag, resulting in severe spikes in geometric curvature.
- **Dispersion:** The kinetic energy of the token's trajectory scatters across the residual stream.
- **Nearest Context Neighbour (NCN):** The generated token's representation physically drifts away from the manifold of the provided context tokens.

By combining these physical metrics, TGP can cleanly isolate ungrounded generations without any external supervision or massive computational overhead.

## Benchmark Results

TGP achieves state-of-the-art hallucination detection performance across multiple architectures and cognitive loads. Below are the 5-fold cross-validation AUROC scores for **Qwen2.5-3B-Instruct**:

| Dataset | Type | AUROC |
| :--- | :--- | :--- |
| **HaluEval QA** | Short Adversarial QA | **0.994** |
| **XSum** | Open-Domain Summarization | **0.934** |
| **HaluEval Summarization** | Long Document Summarization | **0.841** |
| **FRANK** | Factual Consistency Analysis | **0.817** |
| **HaluEval Dialogue** | Conversational Grounding | **0.742** |

## Usage
Run the single-file benchmark suite:
```bash
python tgp_probe.py
```
