"""
linear_probe.py  --  linear probe (difference of means) for AI vs human text
using an LLM's hidden states.  Single file, Python only.

Packages: numpy, scikit-learn, matplotlib  (+ torch, transformers for MODE="real")
    pip install numpy scikit-learn matplotlib torch transformers

Usage:
    python linear_probe.py                       # synthetic demo (SIMULATED data)
    python linear_probe.py --mode real           # GPT-2 on human_texts.txt / ai_texts.txt
    python linear_probe.py --mode real --layer 8 --zscore
Real mode reads one text per line from --human-file and --ai-file.

Pipeline
    1-2. texts -> observer LLM -> mean-pooled hidden state h (all 13 layers cached)
    3.   d   = (mu_AI - mu_Human) / ||mu_AI - mu_Human||        (train only)
    4.   s   = h . d
    5.   tau = midpoint of the two class-mean scores            (train only)
    6.   s > tau -> AI, else Human
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # save figures to files (works headless)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.decomposition import PCA

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
p.add_argument("--layer", type=int, default=8, help="layer for the main analysis (0=embeddings .. 12)")
p.add_argument("--zscore", action="store_true", help="z-score features with TRAIN statistics (main analysis)")
p.add_argument("--human-file", default="human_texts.txt")
p.add_argument("--ai-file", default="ai_texts.txt")
p.add_argument("--model", default="gpt2")
p.add_argument("--max-len", type=int, default=256)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--outdir", default=".")
args = p.parse_args()
rng = np.random.default_rng(args.seed)
SIM = args.mode == "synthetic"
TAG = "SIMULATED DATA - " if SIM else ""


# ------------------------------------------------------------------ data
def get_synthetic(D=768, n_per=400, n_layers=13, shift=4.0, n_outlier=3):
    """SIMULATED hidden states. Isotropic noise; AI class shifted by a small amount
    along ONE random unit direction per layer. Separation follows an inverted U over
    layers. A few huge-variance 'outlier' dimensions mimic GPT-2 (they are
    class-independent nuisance), so that z-scoring has something to fix."""
    y = np.r_[np.zeros(n_per), np.ones(n_per)].astype(int)
    layers = np.arange(n_layers)
    shifts = 0.4 * shift + 0.6 * shift * np.exp(-(layers - 7) ** 2 / (2 * 2.5 ** 2))
    scale = np.ones(D)
    scale[rng.choice(D, n_outlier, replace=False)] = 40.0
    H_all = np.empty((len(y), D, n_layers))
    for l in range(n_layers):
        u = rng.standard_normal(D); u /= np.linalg.norm(u)
        H_all[:, :, l] = rng.standard_normal((len(y), D)) * scale + np.outer(y * shifts[l], u)
    return H_all, y


def get_real(args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    read = lambda f: [ln.strip() for ln in open(f, encoding="utf-8") if ln.strip()]
    human, ai = read(args.human_file), read(args.ai_file)
    texts, y = human + ai, np.r_[np.zeros(len(human)), np.ones(len(ai))].astype(int)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token                       # GPT-2 has no pad token
    model = AutoModel.from_pretrained(args.model).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            enc = tok(texts[i:i + args.batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_len)
            # model run ONCE per batch; all 13 hidden-state tensors are cached
            hs = torch.stack(model(**enc, output_hidden_states=True).hidden_states)  # (13,B,T,D)
            m = enc["attention_mask"][None, :, :, None].float()
            out.append(((hs * m).sum(2) / m.sum(2)).permute(1, 2, 0).numpy())        # mean-pool tokens -> (B,D,13)
    return np.concatenate(out).astype(np.float64), y


H_all, y = get_synthetic() if SIM else get_real(args)
N, D, L = H_all.shape
print(f"MODE={args.mode} ({'SIMULATED' if SIM else 'real GPT-2'}) | N={N} D={D} layers={L}")

# ------------------------------------------------------------------ split
idx = np.arange(N)
idx_tr, idx_te = train_test_split(idx, test_size=0.30, stratify=y, random_state=args.seed)
y_tr, y_te = y[idx_tr], y[idx_te]


# ------------------------------------------------------------------ probe
def fit_probe(Htr, ytr, zscore):
    """Steps 3 and 5 on TRAIN data only. Returns (d, tau, mu, sd) where mu/sd are the
    train statistics applied to any data before projecting (identity if no z-score)."""
    if zscore:
        mu, sd = Htr.mean(0), Htr.std(0) + 1e-8
    else:
        mu, sd = np.zeros(Htr.shape[1]), np.ones(Htr.shape[1])
    Z = (Htr - mu) / sd
    m_h, m_a = Z[ytr == 0].mean(0), Z[ytr == 1].mean(0)
    d = (m_a - m_h) / np.linalg.norm(m_a - m_h)          # step 3
    tau = 0.5 * (m_h @ d + m_a @ d)                      # step 5
    return d, tau, mu, sd


def score(H, d, mu, sd):
    return ((H - mu) / sd) @ d                           # step 4: s = h . d


def evaluate(l, zscore):
    Hl = H_all[:, :, l]
    d, tau, mu, sd = fit_probe(Hl[idx_tr], y_tr, zscore)
    s = score(Hl[idx_te], d, mu, sd)
    return dict(d=d, tau=tau, mu=mu, sd=sd, s=s, acc=np.mean((s > tau) == y_te),  # step 6
                auc=roc_auc_score(y_te, s))


# ------------------------------------------------------------------ main analysis
R = evaluate(args.layer, args.zscore)
zlab = "z-scored" if args.zscore else "raw"
print(f"layer {args.layer} ({zlab}): tau={R['tau']:.4f}  TEST acc={100*R['acc']:.2f}%  AUC={R['auc']:.4f}")

blue, red = "#3344e6", "#e62626"
import os
os.makedirs(args.outdir, exist_ok=True)
save = lambda fig, name: fig.savefig(os.path.join(args.outdir, name), dpi=130, bbox_inches="tight")

# Fig 1: PCA scatter + d projected into the PCA plane (in the same feature space as the probe)
Hl = H_all[:, :, args.layer]
Xp = (Hl - R["mu"]) / R["sd"]
pca = PCA(n_components=2).fit(Xp)
Z2 = pca.transform(Xp)
d2 = pca.components_ @ R["d"]                            # d expressed in PC1/PC2 coordinates
arrow = 0.6 * np.abs(Z2).max() * d2 / max(np.linalg.norm(d2), 1e-12)
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(*Z2[y == 0].T, s=14, c=blue, alpha=.5, label="Human")
ax.scatter(*Z2[y == 1].T, s=14, c=red, alpha=.5, label="AI")
ax.annotate("", xy=arrow, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="k"))
ax.text(*arrow, "  d", fontsize=14, weight="bold")
ax.set(xlabel="PC 1", ylabel="PC 2", title=f"{TAG}PCA, layer {args.layer} ({zlab})\n"
       f"arrow = d projected into PCA plane (|proj| = {np.linalg.norm(d2):.2f} of |d|=1)")
ax.legend(); ax.grid(alpha=.3); ax.set_aspect("equal", "datalim")
save(fig, "fig1_pca.png")

# Fig 2: histograms of test scores with tau
fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.linspace(R["s"].min(), R["s"].max(), 30)
ax.hist(R["s"][y_te == 0], bins, color=blue, alpha=.5, density=True, label="Human")
ax.hist(R["s"][y_te == 1], bins, color=red, alpha=.5, density=True, label="AI")
ax.axvline(R["tau"], ls="--", c="k", lw=2, label=fr"$\tau$ = {R['tau']:.2f}")
ax.set(xlabel=r"projection score $s = h^\top d$", ylabel="density",
       title=f"{TAG}test-set scores, layer {args.layer} ({zlab}), acc {100*R['acc']:.1f}%")
ax.legend()
save(fig, "fig2_hist.png")

# Fig 3: ROC
fpr, tpr, _ = roc_curve(y_te, R["s"])
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(fpr, tpr, c=red, lw=2.5, label=f"linear probe (AUC = {R['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="chance")
ax.set(xlabel="False positive rate", ylabel="True positive rate",
       title=f"{TAG}ROC, layer {args.layer} ({zlab})")
ax.legend(loc="lower right"); ax.grid(alpha=.3); ax.set_aspect("equal")
save(fig, "fig3_roc.png")

# Fig 4 (bonus): test AUC vs layer, raw vs z-scored (same split, train-only stats)
auc_raw = np.array([evaluate(l, False)["auc"] for l in range(L)])
auc_z = np.array([evaluate(l, True)["auc"] for l in range(L)])
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(range(L), auc_raw, "-o", c=red, lw=2, label="raw")
ax.plot(range(L), auc_z, "-s", c=blue, lw=2, label="z-scored (train stats)")
for a, c in ((auc_raw, red), (auc_z, blue)):
    b = a.argmax(); ax.plot(b, a[b], "*", ms=16, c=c, mec="k")
ax.set(xlabel="layer (0 = embeddings, 12 = last block)", ylabel="test AUC",
       title=f"{TAG}AUC vs layer  (best raw: L{auc_raw.argmax()} {auc_raw.max():.3f}; "
             f"best z: L{auc_z.argmax()} {auc_z.max():.3f})")
ax.title.set_fontsize(10)
ax.set_xticks(range(L)); ax.grid(alpha=.3); ax.legend()
save(fig, "fig4_auc_vs_layer.png")
print("AUC raw :", np.round(auc_raw, 3))
print("AUC zsc :", np.round(auc_z, 3))
print("figures saved in", os.path.abspath(args.outdir))
