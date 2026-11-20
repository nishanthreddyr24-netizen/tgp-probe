# ================================================================
# TGP v3 — BIGGER MODEL + FULL ZONE ABLATION
#
# Changes vs v2:
#   1. Qwen2.5-3B-Instruct (36 layers vs 28) — safe on T4
#   2. ALL metrics computed per zone (Early/Middle/Late/Full)
#      including NCN, Virial, MDR, Dispersion — not just NVR/Curv
#   3. Zone ablation table: which zone wins per metric per dataset
#   4. Cross-model comparison scaffold (run 1.5B first, then 3B)
# ================================================================

# !pip install datasets transformers accelerate scikit-learn matplotlib -q

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import warnings, requests, json, random
warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen2.5-3B-Instruct"
N_SAMPLES   = 500
MAX_ANS_LEN = 150
ALPHA       = 1.0
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

assert DEVICE == "cuda", f"GPU not found — switch runtime to T4"
print(f"GPU   : {torch.cuda.get_device_name(0)}")
print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"Model : {MODEL_NAME}")

CTX_LEN = {
    'HaluEval_QA'  : 300,
    'HaluEval_Sum' : 2500,
    'HaluEval_Dial': 1000,
    'FRANK'        : 2500,
    'XSum'         : 2500,
}

# ── LOAD MODEL ──────────────────────────────────────────────────
print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

L     = model.config.num_hidden_layers   # 36 for 3B
D     = model.config.hidden_size         # 2048 for 3B
THIRD = L // 3                           # 12 for 3B

print(f"Layers: {L}  |  D: {D}")
print(f"Zones : Early[1:{THIRD}]  Middle[{THIRD}:{2*THIRD}]  Late[{2*THIRD}:{L}]")
print(f"VRAM used after load: "
      f"{torch.cuda.memory_allocated()/1e9:.1f} GB / "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ── TOKENIZER HELPERS ───────────────────────────────────────────
def tokenize_context(text: str, max_len: int) -> torch.Tensor:
    ids = tokenizer(text, return_tensors='pt',
                    truncation=False, add_special_tokens=True)['input_ids']
    if ids.shape[1] > max_len:
        ids = torch.cat([ids[:, :1], ids[:, -(max_len - 1):]], dim=1)
    return ids

def tokenize_answer(text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors='pt', truncation=True,
                     max_length=MAX_ANS_LEN,
                     add_special_tokens=False)['input_ids']


# ── PER-ZONE METRIC HELPERS ─────────────────────────────────────
def _nvr(zone):
    nl, nt, _ = zone.shape
    if nl < 2 or nt == 0: return 0.0
    norms  = zone.norm(dim=-1)               # (nl, nt)
    r_bar  = norms.mean(dim=0)               # (nt,)
    var_tr = zone.var(dim=0).sum(dim=-1)     # (nt,)
    nvr    = r_bar / (var_tr + 1e-8)
    return float(nvr.min().item())           # min: lower NVR = hallucinated

def _curv(zone):
    nl, nt, _ = zone.shape
    if nl < 3 or nt == 0: return 0.0
    vel = zone[1:] - zone[:-1]              # (nl-1, nt, D)
    if vel.shape[0] < 2: return 0.0
    acc = vel[1:] - vel[:-1]                # (nl-2, nt, D)
    v_m = vel[:-1]
    v2  = (v_m * v_m).sum(dim=-1)
    a2  = (acc * acc).sum(dim=-1)
    va  = (v_m * acc).sum(dim=-1)
    c2  = (v2 * a2 - va**2).clamp(min=0)
    kappa = c2.sqrt() / (v2**1.5 + 1e-8)
    return float(kappa.mean(dim=0).max().item())  # max: higher curv = hallucinated

def _disp(zone):
    nl, nt, _ = zone.shape
    if nl < 2 or nt == 0: return 0.0
    vel  = zone[1:] - zone[:-1]             # (nl-1, nt, D)
    ke   = (vel**2).sum(dim=-1)             # (nl-1, nt)
    return float(ke.sum(dim=0).max().item())

def _mdr(zone_early, zone_mid):
    if zone_early.shape[0] < 2 or zone_mid.shape[0] < 2:
        return 1.0
    vel_e  = zone_early[1:] - zone_early[:-1]
    vel_m  = zone_mid[1:]   - zone_mid[:-1]
    ke_e   = (vel_e**2).sum(dim=-1).mean(dim=0)   # (n_tokens,)
    ke_m   = (vel_m**2).sum(dim=-1).mean(dim=0)
    mdr    = ke_m / (ke_e + 1e-8)
    return float(mdr.max().item())

def _ncn(zone, ctx_zone):
    nl, nt, _ = zone.shape
    if nl == 0 or nt == 0 or ctx_zone.shape[1] == 0:
        return 0.0
    ctx_n = ctx_zone / (ctx_zone.norm(dim=-1, keepdim=True) + 1e-8)
    gen_n = zone     / (zone    .norm(dim=-1, keepdim=True) + 1e-8)
    sims      = torch.bmm(ctx_n, gen_n.transpose(1, 2))
    ncn_layer = sims.max(dim=1).values          # (nl, n_gen)
    ncn_token = ncn_layer.mean(dim=0)           # (n_gen,)
    return float(ncn_token.min().item())

def _virial(zone, ctx_zone):
    nl, nt, _ = zone.shape
    if nl < 2 or nt == 0 or ctx_zone.shape[1] == 0:
        return 1.0
    vel = zone[1:] - zone[:-1]
    ke  = (vel**2).sum(dim=-1)              # (nl-1, n_gen)
    ctx_n = ctx_zone / (ctx_zone.norm(dim=-1, keepdim=True) + 1e-8)
    gen_n = zone     / (zone    .norm(dim=-1, keepdim=True) + 1e-8)
    sims      = torch.bmm(ctx_n, gen_n.transpose(1, 2))
    ncn_layer = sims.max(dim=1).values      # (nl, n_gen)
    ncn_aligned = ncn_layer[:-1]            # (nl-1, n_gen)
    pe_layer    = -ALPHA * ncn_aligned      # (nl-1, n_gen)
    ke_mean  = ke.mean(dim=0)              # (n_gen,)
    pe_mean  = pe_layer.mean(dim=0)        # (n_gen,)
    virial   = ke_mean / (-0.5 * pe_mean + 1e-8)
    return float(virial.max().item())


# ── FULL METRIC EXTRACTION ──────────────────────────────────────
@torch.no_grad()
def compute_metrics(hidden: torch.Tensor, ctx_len: int) -> dict | None:
    ctx = hidden[:, :ctx_len, :].float()      # (L+1, ctx_len, D)
    gen = hidden[:, ctx_len:, :].float()      # (L+1, n_gen,   D)
    if gen.shape[1] == 0:
        return None
    gen = gen[1:]     # (L, n_gen, D)
    ctx = ctx[1:]     # (L, ctx_len, D)
    t = THIRD   # zone boundary

    zones = {
        'Full'  : (gen,           ctx),
        'Early' : (gen[:t],       ctx[:t]),
        'Middle': (gen[t:2*t],    ctx[t:2*t]),
        'Late'  : (gen[2*t:],     ctx[2*t:]),
    }

    feats = {}
    for zname, (g, c) in zones.items():
        feats[f'{zname}_NVR']   = _nvr(g)
        feats[f'{zname}_Curv']  = _curv(g)
        feats[f'{zname}_Disp']  = _disp(g)
        feats[f'{zname}_NCN']   = _ncn(g, c)
        feats[f'{zname}_Virial'] = _virial(g, c)

    feats['MDR_EarlyMid']  = _mdr(gen[:t],    gen[t:2*t])
    feats['MDR_MidLate']   = _mdr(gen[t:2*t], gen[2*t:])
    feats['MDR_EarlyLate'] = _mdr(gen[:t],    gen[2*t:])

    return feats


# ── PROCESS ONE PAIR ────────────────────────────────────────────
def process_pair(context: str, answer: str, max_ctx: int) -> dict | None:
    ctx_ids = tokenize_context(context, max_ctx)
    ans_ids = tokenize_answer(answer)
    if ctx_ids.shape[1] == 0 or ans_ids.shape[1] == 0:
        return None
    full_ids = torch.cat([ctx_ids, ans_ids], dim=1).to(DEVICE)
    ctx_len  = ctx_ids.shape[1]
    try:
        with torch.no_grad():
            out    = model(input_ids=full_ids, output_hidden_states=True)
            hidden = torch.stack(out.hidden_states, dim=0).squeeze(1)
        return compute_metrics(hidden, ctx_len)
    except Exception as e:
        return None
    finally:
        torch.cuda.empty_cache()


# ── FEATURE NAMES ───────────────────────────────────────────────
ZONES        = ['Full', 'Early', 'Middle', 'Late']
BASE_METRICS = ['NVR', 'Curv', 'Disp', 'NCN', 'Virial']
MDR_METRICS  = ['MDR_EarlyMid', 'MDR_MidLate', 'MDR_EarlyLate']

FEAT_NAMES = (
    [f'{z}_{m}' for z in ZONES for m in BASE_METRICS]
    + MDR_METRICS
)

FLIP_METRICS = {'NVR', 'NCN'}
FLIP_SET     = {f'{z}_{m}' for z in ZONES for m in FLIP_METRICS}


# ── DATASET LOADERS ─────────────────────────────────────────────
def load_halueval_qa(n=N_SAMPLES):
    ds = load_dataset("pminervini/HaluEval", "qa",
                      split="data").shuffle(seed=42)
    pairs = []
    for row in ds:
        ctx = f"Question: {row['question']}\nPassage: {row['knowledge']}"
        pairs.append((ctx, row['right_answer'],        0))
        pairs.append((ctx, row['hallucinated_answer'], 1))
    fac = [p for p in pairs if p[2] == 0][:n // 2]
    hal = [p for p in pairs if p[2] == 1][:n // 2]
    return fac + hal

def load_halueval_sum(n=N_SAMPLES):
    ds = load_dataset("pminervini/HaluEval", "summarization",
                      split="data").shuffle(seed=42)
    pairs = []
    for row in ds:
        ctx = f"Document: {row['document']}\nSummary:\n"
        pairs.append((ctx, row['right_summary'],        0))
        pairs.append((ctx, row['hallucinated_summary'], 1))
    fac = [p for p in pairs if p[2] == 0][:n // 2]
    hal = [p for p in pairs if p[2] == 1][:n // 2]
    return fac + hal

def load_halueval_dial(n=N_SAMPLES):
    ds = load_dataset("pminervini/HaluEval", "dialogue",
                      split="data").shuffle(seed=42)
    pairs = []
    for row in ds:
        ctx = (f"Knowledge: {row['knowledge']}\n"
               f"Dialogue History: {row['dialogue_history']}\nResponse:\n")
        pairs.append((ctx, row['right_response'],        0))
        pairs.append((ctx, row['hallucinated_response'], 1))
    fac = [p for p in pairs if p[2] == 0][:n // 2]
    hal = [p for p in pairs if p[2] == 1][:n // 2]
    return fac + hal

def load_frank(n=N_SAMPLES):
    print("  Downloading FRANK from GitHub...")
    url = ('https://raw.githubusercontent.com/artidoro/frank/main/'
           'data/human_annotations_sentence.json')
    data = json.loads(requests.get(url).text)
    pairs = []
    for ex in data:
        if ex.get('split') != 'valid':
            continue
        lbl = 0
        for sent_anno in ex.get('summary_sentences_annotations', []):
            sent_lbls = [1 if anno[0] == 'NoE' else 0
                         for anno in sent_anno.values()]
            if not int(round(sum(sent_lbls) / max(1, len(sent_lbls)))):
                lbl = 1
        article = ex.get('article', '')
        summary = ex.get('summary', '')
        if article and summary:
            pairs.append((f"Article: {article}\nSummary:\n", summary, lbl))
    random.seed(42); random.shuffle(pairs)
    fac = [p for p in pairs if p[2] == 0][:n // 2]
    hal = [p for p in pairs if p[2] == 1][:n // 2]
    k   = min(len(fac), len(hal))
    if k == 0: raise RuntimeError("FRANK: 0 usable pairs")
    return fac[:k] + hal[:k]

def load_xsum(n=N_SAMPLES):
    try:
        ds = load_dataset("orange-share/xsum-factuality",
                          split="test").shuffle(seed=42)
        pairs = []
        for row in ds:
            ctx   = f"Article: {row['document']}\nSummary:\n"
            label = 0 if str(row.get('label','faithful')).lower() \
                         in ['faithful','0','consistent'] else 1
            pairs.append((ctx, row['summary'], label))
    except Exception:
        ds = (load_dataset("EdinburghNLP/xsum", split="test")
              .shuffle(seed=42).select(range(n)))
        arts = [r['document'] for r in ds]
        sums = [r['summary']  for r in ds]
        h    = n // 2
        pairs  = [(f"Article: {arts[i]}\nSummary:\n", sums[i], 0)
                  for i in range(h)]
        pairs += [(f"Article: {arts[i]}\nSummary:\n", sums[(i+1)%h], 1)
                  for i in range(h)]
    fac = [p for p in pairs if p[2] == 0][:n // 2]
    hal = [p for p in pairs if p[2] == 1][:n // 2]
    k   = min(len(fac), len(hal))
    return fac[:k] + hal[:k]

DATASETS = {
    'HaluEval_QA'  : (load_halueval_qa,  CTX_LEN['HaluEval_QA']),
    'HaluEval_Sum' : (load_halueval_sum,  CTX_LEN['HaluEval_Sum']),
    'HaluEval_Dial': (load_halueval_dial, CTX_LEN['HaluEval_Dial']),
    'FRANK'        : (load_frank,         CTX_LEN['FRANK']),
    'XSum'         : (load_xsum,          CTX_LEN['XSum']),
}


# ── EVALUATE ONE DATASET ─────────────────────────────────────────
def evaluate(ds_name: str, loader, max_ctx: int) -> dict | None:
    print(f"\n{'='*62}\n  {ds_name}  (max_ctx={max_ctx})\n{'='*62}")
    try:
        pairs = loader()
    except Exception as e:
        print(f"  LOAD ERROR: {e}"); return None

    n_fac = sum(1 for p in pairs if p[2] == 0)
    n_hal = sum(1 for p in pairs if p[2] == 1)
    print(f"  Pairs: {len(pairs)}  ({n_fac} factual / {n_hal} hallucinated)")

    feats, labels, errors = [], [], 0
    for ctx, ans, lbl in tqdm(pairs, desc="  Extracting"):
        m = process_pair(ctx, ans, max_ctx)
        if m is None:
            errors += 1; continue
        feats.append([m.get(fn, 0.0) for fn in FEAT_NAMES])
        labels.append(lbl)

    print(f"  OK: {len(feats)} / {len(pairs)}  ({errors} errors)")
    if len(feats) < 50:
        print("  SKIP: too few samples"); return None

    X = np.nan_to_num(np.array(feats, dtype=np.float32),
                      nan=0.0, posinf=10.0, neginf=-10.0)
    y = np.array(labels, dtype=int)

    # ── Per-feature AUROC ──────────────────────────────────────
    print(f"\n  Per-feature AUROC:")
    per_metric = {}
    for i, fn in enumerate(FEAT_NAMES):
        col = -X[:, i] if fn in FLIP_SET else X[:, i]
        try:    auc = roc_auc_score(y, col)
        except: auc = 0.5
        per_metric[fn] = auc

    print(f"  {'Feature':<22}  {'Full':>6}  {'Early':>6}  "
          f"{'Middle':>6}  {'Late':>6}")
    print(f"  {'-'*50}")
    for bm in BASE_METRICS:
        row_vals = {z: per_metric.get(f'{z}_{bm}', 0.5) for z in ZONES}
        best_z   = max(row_vals, key=row_vals.get)
        line     = f"  {bm:<22}"
        for z in ZONES:
            v    = row_vals[z]
            mark = ' ←' if z == best_z else '  '
            line += f"  {v:.3f}{mark}" if z == best_z else f"  {v:.3f}  "
        print(line)
    print(f"  {'-'*50}")
    for mdr_k in MDR_METRICS:
        v = per_metric.get(mdr_k, 0.5)
        print(f"  {mdr_k:<22}  {v:.3f}")

    # ── Combined — 5-fold CV ───────────────────────────────────
    X_sc   = StandardScaler().fit_transform(X)
    clf    = LogisticRegression(class_weight='balanced',
                                max_iter=1000, C=1.0, random_state=42)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = cross_val_predict(clf, X_sc, y,
                               cv=cv, method='predict_proba')[:, 1]

    auc_comb      = roc_auc_score(y, y_prob)
    fpr, tpr, thr = roc_curve(y, y_prob)
    best          = int(np.argmax(tpr - fpr))

    print(f"\n  Combined ({len(FEAT_NAMES)} features, 5-fold CV):")
    print(f"    AUROC  : {auc_comb:.3f}")
    print(f"    Optimal: TPR={tpr[best]:.3f}  FPR={fpr[best]:.3f}  "
          f"thr={thr[best]:.3f}")

    return dict(
        name=ds_name, n=len(feats),
        per_metric=per_metric,
        combined_auroc=auc_comb,
        fpr=fpr, tpr=tpr, thresholds=thr,
        best_tpr=float(tpr[best]),
        best_fpr=float(fpr[best]),
        best_thr=float(thr[best]),
        X=X, y=y,
    )


# ── RUN ALL ──────────────────────────────────────────────────────
if __name__ == "__main__":
    results = {}
    for name, (loader, max_ctx) in DATASETS.items():
        r = evaluate(name, loader, max_ctx)
        if r:
            results[name] = r
        torch.cuda.empty_cache()

    # ── FINAL SUMMARY ────────────────────────────────────────────────
    print("\n\n" + "=" * 68)
    print(f"FINAL SUMMARY — {MODEL_NAME}")
    print("=" * 68)
    print(f"{'Dataset':<20} {'N':>5}  {'Combined':>8}  "
          f"{'TPR':>6}  {'FPR':>6}  {'Status'}")
    print("-" * 68)
    for name, r in results.items():
        auc    = r['combined_auroc']
        status = '✅' if auc >= 0.85 else ('⚠️ ' if auc >= 0.70 else '❌')
        print(f"{name:<20} {r['n']:>5}  {auc:>8.3f}  "
              f"{r['best_tpr']:>6.3f}  {r['best_fpr']:>6.3f}  {status}")
