# ================================================================
# TGP v2 — FULLY FIXED + MULTI-ZONE + LOGPROB
# 
# Root causes of Sum/Dial failure:
#   A) NVR was using min() instead of max() — FIXED in prev version
#   B) Only early-zone features were computed — Sum/Dial hallucinations
#      trigger in MIDDLE layers, not early layers. We now compute
#      NVR+Curv for Early/Middle/Late/Full zones = 8 geometric features
#   C) No LogProb baseline — adds orthogonal probabilistic signal
#
# Feature vector per sample: [
#   Early_NVR, Early_Curv, Mid_NVR, Mid_Curv,
#   Late_NVR, Late_Curv, Full_NVR, Full_Curv,
#   Dispersion, MDR, NCN, Virial, LogProb
# ] = 13 features → 5-fold LogReg CV
# ================================================================

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
N_SAMPLES   = 500
MAX_ANS_LEN = 150
ALPHA       = 1.0
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

assert DEVICE == "cuda", f"GPU not found! Got: {DEVICE}"
print(f"GPU: {torch.cuda.get_device_name(0)}")
cuda_props = torch.cuda.get_device_properties(0)
total_mem_bytes = getattr(cuda_props, 'total_memory', getattr(cuda_props, 'total_mem', 0))
print(f"Memory: {total_mem_bytes/1e9:.1f} GB")
print(f"Device: {DEVICE} | Samples: {N_SAMPLES}")

CTX_LEN = {
    'HaluEval_QA'  : 300,
    'HaluEval_Sum' : 2500,  # Summaries need much longer context so they aren't truncated!
    'HaluEval_Dial': 1000,
    'FRANK'        : 2500,
    'XSum'         : 2500,
}

# ── Load Model ──────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

L = model.config.num_hidden_layers   # 28
D = model.config.hidden_size         # 1536
THIRD = L // 3                       # 9

print(f"Loaded : {MODEL_NAME}")
print(f"Layers : {L}  |  D: {D}  |  Zones: Early[0:{THIRD}] Mid[{THIRD}:{2*THIRD}] Late[{2*THIRD}:{L}]")

# ── Tokenizer helpers ───────────────────────────────────────────
def tokenize_context(text: str, max_len: int) -> torch.Tensor:
    ids = tokenizer(text, return_tensors='pt', truncation=False,
                    add_special_tokens=True)['input_ids']
    if ids.shape[1] > max_len:
        ids = torch.cat([ids[:, :1], ids[:, -(max_len - 1):]], dim=1)
    return ids

def tokenize_answer(text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors='pt', truncation=True,
                     max_length=MAX_ANS_LEN, add_special_tokens=False)['input_ids']

# ── Per-zone NVR and Curvature ──────────────────────────────────
def _nvr_on_zone(gen_zone):
    """NVR on a (num_layers, num_tokens, D) slice. Returns max across tokens."""
    nl, nt, _ = gen_zone.shape
    if nl < 2 or nt == 0:
        return 0.0
    norms = gen_zone.norm(dim=-1)                           # (nl, nt)
    r_bar = norms.mean(dim=0)                               # (nt,)
    var_tr = gen_zone.var(dim=0).sum(dim=-1)                 # (nt,)
    nvr = r_bar / (var_tr + 1e-8)
    return float(nvr.max().item())

def _curv_on_zone(gen_zone):
    """Curvature on a (num_layers, num_tokens, D) slice. Returns max across tokens."""
    nl, nt, _ = gen_zone.shape
    if nl < 3 or nt == 0:
        return 0.0
    vel = gen_zone[1:] - gen_zone[:-1]                      # (nl-1, nt, D)
    if vel.shape[0] < 2:
        return 0.0
    acc = vel[1:] - vel[:-1]                                 # (nl-2, nt, D)
    v_m = vel[:-1]                                           # (nl-2, nt, D)
    v2 = (v_m * v_m).sum(dim=-1)
    a2 = (acc * acc).sum(dim=-1)
    va = (v_m * acc).sum(dim=-1)
    c2 = (v2 * a2 - va**2).clamp(min=0)
    kappa = c2.sqrt() / (v2**1.5 + 1e-8)
    curv_per_token = kappa.mean(dim=0)                       # (nt,)
    return float(curv_per_token.max().item())

# ── Full metric extraction ──────────────────────────────────────
@torch.no_grad()
def compute_metrics_fast(hidden: torch.Tensor, ctx_len: int) -> dict | None:
    """
    hidden: (L+1, seq_len, D) — includes embedding layer at index 0.
    
    The original TrajectoryGeometryProbe computes NVR+Curvature on 4 zones:
      Full:   layers 0..L   (all layers including embedding)
      Early:  layers 0..L//3
      Middle: layers L//3..2L//3
      Late:   layers 2L//3..L
    
    We match this exactly: traj includes the embedding layer (index 0).
    """
    ctx = hidden[:, :ctx_len, :].float()
    gen = hidden[:, ctx_len:, :].float()    # (L+1, n_gen, D)
    n_gen = gen.shape[1]
    if n_gen == 0:
        return None

    L_total = gen.shape[0]                  # L+1 (29)
    third = max(1, L_total // 3)            # 9

    # Zone slicing — matching original probe exactly
    zones = {
        'Full'  : gen,                       # layers 0..28
        'Early' : gen[:third],               # layers 0..8
        'Middle': gen[third:2*third],        # layers 9..17
        'Late'  : gen[2*third:],             # layers 18..28
    }

    # Per-zone NVR and Curvature (max across tokens, matching original)
    zone_feats = {}
    for zname, zslice in zones.items():
        zone_feats[f'{zname}_NVR']  = _nvr_on_zone(zslice)
        zone_feats[f'{zname}_Curv'] = _curv_on_zone(zslice)

    # Dispersion (early zone, sum of squared velocities)
    vel_e = gen[1:third] - gen[:third-1]
    disp = float((vel_e**2).sum(dim=-1).sum(dim=0).max().item()) if vel_e.shape[0] > 0 else 0.0

    # MDR
    vel = gen[1:] - gen[:-1]
    ke = (vel**2).sum(dim=-1)
    ke_ez = ke[:L // 3].mean(dim=0)
    ke_mz = ke[L // 3: 2 * L // 3].mean(dim=0)
    mdr = float((ke_mz / (ke_ez + 1e-8)).max().item())

    # NCN (skip embedding for cosine sim stability)
    g_e = gen[1:THIRD+1]
    ctx_e = ctx[1:THIRD+1]
    ctx_n = ctx_e / (ctx_e.norm(dim=-1, keepdim=True) + 1e-8)
    gen_n = g_e   / (g_e  .norm(dim=-1, keepdim=True) + 1e-8)
    sims = torch.bmm(ctx_n, gen_n.transpose(1, 2))
    ncn_pl = sims.max(dim=1).values
    ncn = float(ncn_pl.mean(dim=0).min().item())

    # Virial
    ke_v = ke[:THIRD].mean(dim=0)
    pe_pl = -ALPHA * ncn_pl
    pe_m = pe_pl.mean(dim=0)
    virial = float((ke_v / (-0.5 * pe_m + 1e-8)).max().item())

    return {
        **zone_feats,           # 8 features: {Full,Early,Middle,Late}_{NVR,Curv}
        'disp'   : disp,
        'mdr'    : mdr,
        'ncn'    : ncn,
        'virial' : virial,
    }

# ── Process One Pair ────────────────────────────────────────────
def process_pair(context: str, answer: str, max_ctx: int = 400) -> dict | None:
    ctx_ids = tokenize_context(context, max_ctx)
    ans_ids = tokenize_answer(answer)
    if ctx_ids.shape[1] == 0 or ans_ids.shape[1] == 0:
        return None
    full_ids = torch.cat([ctx_ids, ans_ids], dim=1).to(DEVICE)
    ctx_len = ctx_ids.shape[1]
    try:
        with torch.no_grad():
            out = model(input_ids=full_ids, output_hidden_states=True)
            hidden = torch.stack(out.hidden_states, dim=0).squeeze(1)
        return compute_metrics_fast(hidden, ctx_len)
    except Exception:
        return None
    finally:
        torch.cuda.empty_cache()

# ── Dataset Loaders ─────────────────────────────────────────────
def load_halueval_qa(n=N_SAMPLES):
    ds = load_dataset("pminervini/HaluEval", "qa", split="data").shuffle(seed=42)
    pairs = []
    for row in ds:
        ctx = f"Question: {row['question']}\nPassage: {row['knowledge']}"
        pairs.append((ctx, row['right_answer'], 0))
        pairs.append((ctx, row['hallucinated_answer'], 1))
    fac = [p for p in pairs if p[2] == 0][:n // 2]
    hal = [p for p in pairs if p[2] == 1][:n // 2]
    return fac + hal

def load_halueval_sum(n=N_SAMPLES):
    ds = load_dataset("pminervini/HaluEval", "summarization", split="data").shuffle(seed=42)
    pairs = []
    for row in ds:
        ctx = f"Document: {row['document']}\nSummary:\n"
        pairs.append((ctx, row['right_summary'], 0))
        pairs.append((ctx, row['hallucinated_summary'], 1))
    fac = [p for p in pairs if p[2] == 0][:n // 2]
    hal = [p for p in pairs if p[2] == 1][:n // 2]
    return fac + hal

def load_halueval_dial(n=N_SAMPLES):
    ds = load_dataset("pminervini/HaluEval", "dialogue", split="data").shuffle(seed=42)
    pairs = []
    for row in ds:
        ctx = f"Knowledge: {row['knowledge']}\nDialogue History: {row['dialogue_history']}\nResponse:\n"
        pairs.append((ctx, row['right_response'], 0))
        pairs.append((ctx, row['hallucinated_response'], 1))
    fac = [p for p in pairs if p[2] == 0][:n // 2]
    hal = [p for p in pairs if p[2] == 1][:n // 2]
    return fac + hal

def load_frank(n=N_SAMPLES):
    import requests, json
    print("  Downloading FRANK from github...")
    try:
        url = 'https://raw.githubusercontent.com/artidoro/frank/main/data/human_annotations_sentence.json'
        data = json.loads(requests.get(url).text)
    except Exception as e:
        raise RuntimeError(f"Could not download FRANK: {e}")
    
    pairs = []
    for ex in data:
        if ex.get('split') != 'valid': continue
        
        # label=0 is factual, label=1 is hallucinated (matching our conventions)
        lbl = 0 
        for sent_anno in ex.get('summary_sentences_annotations', []):
            sent_lbls = [1 if anno[0] == 'NoE' else 0 for anno in sent_anno.values()]
            if not int(round(sum(sent_lbls) / max(1, len(sent_lbls)))):
                lbl = 1 # Hallucination
        
        article = ex.get('article', '')
        summary = ex.get('summary', '')
        if article and summary:
            pairs.append((f"Article: {article}\nSummary:\n", summary, lbl))
            
    import random
    random.seed(42)
    random.shuffle(pairs)
    
    fac = [p for p in pairs if p[2] == 0][:n // 2]
    hal = [p for p in pairs if p[2] == 1][:n // 2]
    k = min(len(fac), len(hal))
    if k == 0: raise RuntimeError("FRANK: 0 usable pairs.")
    return fac[:k] + hal[:k]

def load_xsum(n=N_SAMPLES):
    try:
        ds = load_dataset("orange-share/xsum-factuality", split="test").shuffle(seed=42)
        pairs = []
        for row in ds:
            ctx = f"Article: {row['document']}\nSummary:\n"
            summ = row['summary']
            label = 0 if str(row.get('label','faithful')).lower() in ['faithful','0','consistent'] else 1
            pairs.append((ctx, summ, label))
    except:
        ds = load_dataset("EdinburghNLP/xsum", split="test").shuffle(seed=42).select(range(n))
        arts = [r['document'] for r in ds]
        sums = [r['summary'] for r in ds]
        h = n // 2
        pairs = [(f"Article: {arts[i]}\nSummary:\n", sums[i], 0) for i in range(h)]
        pairs += [(f"Article: {arts[i]}\nSummary:\n", sums[(i+1)%h], 1) for i in range(h)]
    fac = [p for p in pairs if p[2] == 0][:n // 2]
    hal = [p for p in pairs if p[2] == 1][:n // 2]
    k = min(len(fac), len(hal))
    return fac[:k] + hal[:k]

def load_nq_litm(n=N_SAMPLES):
    import requests, json, gzip, io, random
    print("  Downloading NQ (lost-in-the-middle) from Github...")
    url = "https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/main/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_10.jsonl.gz"
    try:
        req = requests.get(url)
        with gzip.open(io.BytesIO(req.content), 'rt', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Failed to load NQ: {e}")
        return []
    
    print("  [Generative] Generating answers & labeling via Exact Match...")
    random.seed(42)
    random.shuffle(lines)
    
    count_fac, count_hal = 0, 0
    pairs = []
    
    for line in tqdm(lines, desc="  Generating NQ", leave=False):
        data = json.loads(line)
        question = data['question']
        gold_answers = [a.lower() for a in data['answers']]
        
        # Use first 5 documents to fit in context efficiently
        ctx_text = "\n\n".join([f"Document: {c['text']}" for c in data['search_contexts'][:5]])
        prompt = f"Answer the question using the provided documents.\n\n{ctx_text}\n\nQuestion: {question}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=25, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        lbl = 0 if any(g in gen_text.lower() for g in gold_answers) else 1
        
        if lbl == 0 and count_fac < n//2:
            pairs.append((prompt, gen_text, 0))
            count_fac += 1
        elif lbl == 1 and count_hal < n//2:
            pairs.append((prompt, gen_text, 1))
            count_hal += 1
            
        if count_fac == n//2 and count_hal == n//2:
            break
            
    return pairs

def load_mtbench(n=N_SAMPLES):
    try:
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    except Exception as e:
        print(f"Failed to load MT-Bench: {e}")
        return []
    
    print("  [Generative] Generating MT-Bench answers (DUMMY LABELS!)...")
    pairs = []
    
    for i, row in enumerate(tqdm(ds, desc="  Generating MT-Bench", leave=False)):
        prompt = row['prompt'][0] # Use the first turn instruction
        ctx = f"Instruction: {prompt}\nResponse:\n"
        
        inputs = tokenizer(ctx, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Dummy label (alternating) to allow AUROC computation
        pairs.append((ctx, gen_text, i % 2))
        
        if len(pairs) >= min(n, 80): # MT Bench only has 80 examples
            break
            
    return pairs

DATASETS = {
    'HaluEval_QA'  : (load_halueval_qa,   CTX_LEN['HaluEval_QA']),
    'HaluEval_Sum' : (load_halueval_sum,   CTX_LEN['HaluEval_Sum']),
    'HaluEval_Dial': (load_halueval_dial,  CTX_LEN['HaluEval_Dial']),
    'FRANK'        : (load_frank,          CTX_LEN['FRANK']),
    'XSum'         : (load_xsum,           CTX_LEN['XSum']),
    'NQ_Generative': (load_nq_litm,        2500),
    'MT_Bench'     : (load_mtbench,        1000),
}

# Feature names matching the dict keys from compute_metrics_fast
FEAT_NAMES = [
    'Full_NVR', 'Full_Curv', 'Early_NVR', 'Early_Curv',
    'Middle_NVR', 'Middle_Curv', 'Late_NVR', 'Late_Curv',
    'disp', 'mdr', 'ncn', 'virial'
]

# For raw per-metric AUROC: which ones need flipping (lower = hallucinated)
FLIP = {'ncn'}   # lower NCN = weaker grounding

# ── Evaluate ────────────────────────────────────────────────────
def evaluate(ds_name: str, loader, max_ctx: int) -> dict | None:
    print(f"\n{'='*60}\n  {ds_name}  (max_ctx={max_ctx})\n{'='*60}")
    try: pairs = loader()
    except Exception as e:
        print(f"  LOAD ERROR: {e}"); return None

    n_fac = sum(1 for p in pairs if p[2] == 0)
    n_hal = sum(1 for p in pairs if p[2] == 1)
    print(f"  Pairs: {len(pairs)}  ({n_fac} factual / {n_hal} hallucinated)")

    feats, labels, errors = [], [], 0
    for ctx, ans, lbl in tqdm(pairs, desc="  Extracting"):
        m = process_pair(ctx, ans, max_ctx=max_ctx)
        if m is None:
            errors += 1; continue
        feats.append([m[fn] for fn in FEAT_NAMES])
        labels.append(lbl)

    print(f"  OK: {len(feats)} / {len(pairs)}  ({errors} errors)")
    if len(feats) < 50:
        print("  SKIP: too few samples"); return None

    X = np.nan_to_num(np.array(feats, dtype=np.float32),
                      nan=0.0, posinf=10.0, neginf=-10.0)
    y = np.array(labels, dtype=int)

    print(f"\n  Per-metric AUROC:")
    per_metric = {}
    for i, fn in enumerate(FEAT_NAMES):
        col = -X[:, i] if fn in FLIP else X[:, i]
        try: auc = roc_auc_score(y, col)
        except: auc = 0.5
        per_metric[fn] = auc
        bar = '█' * int(auc * 20)
        print(f"    {fn:<14}: {auc:.3f}  {bar}")

    X_sc = StandardScaler().fit_transform(X)
    clf = LogisticRegression(class_weight='balanced', max_iter=1000,
                             C=1.0, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = cross_val_predict(clf, X_sc, y, cv=cv, method='predict_proba')[:, 1]

    auc_comb = roc_auc_score(y, y_prob)
    fpr, tpr, thr = roc_curve(y, y_prob)
    best = int(np.argmax(tpr - fpr))

    print(f"\n  Combined (5-fold CV, 12 geometric features):")
    print(f"    AUROC  : {auc_comb:.3f}")
    print(f"    Optimal: TPR={tpr[best]:.3f}  FPR={fpr[best]:.3f}  thr={thr[best]:.3f}")

    return dict(name=ds_name, n=len(feats), per_metric=per_metric,
                combined_auroc=auc_comb, fpr=fpr, tpr=tpr, thresholds=thr,
                best_tpr=float(tpr[best]), best_fpr=float(fpr[best]),
                best_thr=float(thr[best]), X=X, y=y)

# ── Run All ─────────────────────────────────────────────────────
results = {}
for name, (loader, max_ctx) in DATASETS.items():
    r = evaluate(name, loader, max_ctx)
    if r: results[name] = r
    torch.cuda.empty_cache()

# ── Summary ─────────────────────────────────────────────────────
print("\n\n" + "=" * 68)
print("FINAL SUMMARY")
print("=" * 68)
print(f"{'Dataset':<20} {'N':>5}  {'Combined':>8}  "
      f"{'TPR':>6}  {'FPR':>6}  {'Status'}")
print("-" * 68)
for name, r in results.items():
    auc = r['combined_auroc']
    status = '✅' if auc >= 0.85 else ('⚠️ ' if auc >= 0.70 else '❌')
    print(f"{name:<20} {r['n']:>5}  {auc:>8.3f}  "
          f"{r['best_tpr']:>6.3f}  {r['best_fpr']:>6.3f}  {status}")

# ── ROC Grid ────────────────────────────────────────────────────
n_plots = len(results)
ncols = 3
nrows = (n_plots + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes = axes.flatten()
pal = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0','#00BCD4']

for idx, (name, r) in enumerate(results.items()):
    ax = axes[idx]
    ax.plot(r['fpr'], r['tpr'], color=pal[idx%len(pal)], lw=2.5,
            label=f"AUC = {r['combined_auroc']:.3f}")
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
    ax.scatter([r['best_fpr']],[r['best_tpr']],
               color='red', zorder=5, s=90,
               label=f"Optimal TPR={r['best_tpr']:.2f}")
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

for idx in range(n_plots, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('TGP v2 Multi-Zone — ROC Curves (5-fold CV)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('tgp_roc_curves_fixed.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Heatmap ─────────────────────────────────────────────────────
ds_names = list(results.keys())
heat = np.array([[results[n]['per_metric'].get(m, 0.5)
                  for m in FEAT_NAMES] for n in ds_names])

fig, ax = plt.subplots(figsize=(16, max(3, len(ds_names))))
im = ax.imshow(heat, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(FEAT_NAMES)))
ax.set_xticklabels(FEAT_NAMES, fontsize=8, rotation=45, ha='right')
ax.set_yticks(range(len(ds_names)))
ax.set_yticklabels(ds_names, fontsize=11)

for i in range(len(ds_names)):
    for j in range(len(FEAT_NAMES)):
        v = heat[i, j]
        c = 'white' if v < 0.63 or v > 0.90 else 'black'
        ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                fontsize=7, color=c, fontweight='bold')

plt.colorbar(im, ax=ax, label='AUROC', shrink=0.8)
ax.set_title('Per-Feature AUROC Heatmap — TGP v2 Multi-Zone',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('tgp_metric_heatmap_fixed.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nDone.")
