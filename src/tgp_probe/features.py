"""Geometric features on cross-layer hidden-state trajectories.

All functions take hidden states shaped ``(L+1, seq_len, D)``: index 0 is the
embedding output, indices 1..L are the transformer layers. Every token ``t``
therefore has a trajectory ``gamma_t = (h_t^0, ..., h_t^L)`` through depth.

Zones (``third = (L+1) // 3``)::

    Full   = layers [0, L]
    Early  = layers [0, third)
    Middle = layers [third, 2*third)
    Late   = layers [2*third, L]

Every function here is pure torch, needs no model and no gradients, and costs
O(L x T x D) per sample.
"""

from __future__ import annotations

from typing import Dict, List

import torch

EPS = 1e-8

#: The three features used by the paper's calibrator:
#: [max kappa_full, max kappa_mid, min NCN].
PAPER_FEATURES: List[str] = ["Full_Curv", "Middle_Curv", "NCN"]

#: All 12 features from the multi-zone (v2) experiments.
FULL_FEATURES: List[str] = [
    "Full_NVR", "Full_Curv", "Early_NVR", "Early_Curv",
    "Middle_NVR", "Middle_Curv", "Late_NVR", "Late_Curv",
    "Dispersion", "MDR", "NCN", "Virial",
]

FEATURE_SETS: Dict[str, List[str]] = {"paper": PAPER_FEATURES, "full": FULL_FEATURES}

#: Features where a *lower* value means "more likely hallucinated".
LOWER_IS_HALLUCINATED = {"NCN"}


def resolve_feature_set(feature_set) -> List[str]:
    if isinstance(feature_set, str):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}")
        return list(FEATURE_SETS[feature_set])
    names = list(feature_set)
    unknown = [n for n in names if n not in FULL_FEATURES]
    if unknown:
        raise ValueError(f"Unknown feature names: {unknown}")
    return names


# ── per-token primitives ────────────────────────────────────────────────────

def token_curvature(traj: torch.Tensor) -> torch.Tensor:
    """Mean discrete curvature along depth, per token.

    ``kappa = ||v ^ a|| / (||v||^3 + eps)``. The bivector norm is computed as
    ``sqrt(|v|^2 |a|^2 - (v.a)^2)``, the D-dimensional generalisation of the
    3-D cross product.

    traj: ``(n_layers, n_tokens, D)`` -> ``(n_tokens,)``
    """
    nl, nt, _ = traj.shape
    if nl < 3 or nt == 0:
        return traj.new_zeros(nt)
    vel = traj[1:] - traj[:-1]
    acc = vel[1:] - vel[:-1]
    v = vel[:-1]
    v2 = (v * v).sum(-1)
    a2 = (acc * acc).sum(-1)
    va = (v * acc).sum(-1)
    wedge = (v2 * a2 - va ** 2).clamp(min=0).sqrt()
    kappa = wedge / (v2 ** 1.5 + EPS)
    return kappa.mean(0)


def token_nvr(traj: torch.Tensor) -> torch.Tensor:
    """Normalised Vector Radius: mean norm over trace of variance, per token."""
    nl, nt, _ = traj.shape
    if nl < 2 or nt == 0:
        return traj.new_zeros(nt)
    r_bar = traj.norm(dim=-1).mean(0)
    var_tr = traj.var(dim=0).sum(-1)
    return r_bar / (var_tr + EPS)


def token_ncn_per_layer(gen: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
    """Max cosine similarity of each generated token to any context token.

    gen: ``(n_layers, n_gen, D)``, ctx: ``(n_layers, n_ctx, D)`` -> ``(n_layers, n_gen)``
    """
    ctx_n = ctx / (ctx.norm(dim=-1, keepdim=True) + EPS)
    gen_n = gen / (gen.norm(dim=-1, keepdim=True) + EPS)
    sims = torch.bmm(gen_n, ctx_n.transpose(1, 2))  # (nl, n_gen, n_ctx)
    return sims.max(dim=-1).values


# ── full extraction ─────────────────────────────────────────────────────────

def zone_third(num_states: int) -> int:
    """Zone width for a trajectory with ``num_states = L + 1`` states."""
    return max(1, num_states // 3)


@torch.no_grad()
def compute_features(
    hidden: torch.Tensor,
    ctx_len: int,
    alpha: float = 1.0,
    return_token_traces: bool = False,
) -> Dict[str, object]:
    """Compute TGP features for one (context, response) sequence.

    Args:
        hidden: ``(L+1, seq_len, D)`` hidden states for ``context + response``
            (any dtype; only the slices that are needed get upcast to fp32).
        ctx_len: number of context tokens at the start of the sequence.
        alpha: potential-energy scale for the Virial ratio.
        return_token_traces: also return per-response-token curvature and NCN
            under the key ``"_tokens"``.

    Returns:
        A dict with every name in :data:`FULL_FEATURES`, mapped to a Python float.
    """
    if hidden.dim() != 3:
        raise ValueError(f"hidden must be (L+1, seq, D); got {tuple(hidden.shape)}")
    n_states, seq_len, _ = hidden.shape
    if not 0 < ctx_len < seq_len:
        raise ValueError(f"need 0 < ctx_len < seq_len (got ctx_len={ctx_len}, seq_len={seq_len})")

    third = zone_third(n_states)
    gen = hidden[:, ctx_len:, :].float()                       # (L+1, n_gen, D)
    # Early-zone context, skipping the embedding layer for cosine stability.
    ctx_early = hidden[1:third + 1, :ctx_len, :].float()       # (third, n_ctx, D)

    zones = {
        "Full": gen,
        "Early": gen[:third],
        "Middle": gen[third:2 * third],
        "Late": gen[2 * third:],
    }

    feats: Dict[str, object] = {}
    curv_tokens: Dict[str, torch.Tensor] = {}
    for name, z in zones.items():
        feats[f"{name}_NVR"] = _max(token_nvr(z))
        curv_tokens[name] = token_curvature(z)
        feats[f"{name}_Curv"] = _max(curv_tokens[name])

    vel = gen[1:] - gen[:-1]                                   # (L, n_gen, D)
    ke = (vel ** 2).sum(-1)                                    # (L, n_gen)

    # Dispersion: kinetic energy summed over the early zone.
    ke_early = ke[: max(0, third - 1)]
    feats["Dispersion"] = _max(ke_early.sum(0)) if ke_early.shape[0] else 0.0

    # MDR: mid-zone vs early-zone kinetic energy.
    ke_e = ke[:third].mean(0)
    ke_m = ke[third:2 * third].mean(0) if ke[third:2 * third].shape[0] else torch.zeros_like(ke_e)
    feats["MDR"] = _max(ke_m / (ke_e + EPS))

    # NCN: early-zone grounding to the context manifold (lower = drift).
    ncn_layer = token_ncn_per_layer(gen[1:third + 1], ctx_early)   # (third, n_gen)
    ncn_token = ncn_layer.mean(0)
    feats["NCN"] = float(ncn_token.min().item())

    # Virial: kinetic / potential balance in the early zone.
    pe = (-alpha * ncn_layer).mean(0)
    feats["Virial"] = _max(ke[:third].mean(0) / (-0.5 * pe + EPS))

    if return_token_traces:
        feats["_tokens"] = {
            "curvature": curv_tokens["Full"].tolist(),
            "ncn": ncn_token.tolist(),
        }
    return feats


def _max(x: torch.Tensor) -> float:
    return float(x.max().item()) if x.numel() else 0.0


def features_to_vector(feats: Dict[str, float], names: List[str]) -> List[float]:
    return [float(feats[n]) for n in names]
