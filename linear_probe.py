"""
linear_probe.py  --  linear probe (difference of means) for AI vs human text
using an LLM's hidden states.  Single file, Python only.

Packages: numpy, scikit-learn, matplotlib  (+ torch, transformers for MODE="real")
    pip install numpy scikit-learn matplotlib torch transformers

Usage:
    python linear_probe.py                       # synthetic demo (SIMULATED data)
    python linear_probe.py --mode real           # GPT-2 on human_texts.txt / ai_texts.txt
    python linear_probe.py --mode real --layer 8 --zscore
Figures (saved to figures/): pipeline_<mode>.png/.pdf (pipeline diagram with the actual data)
and results_<mode>.png (2x2: PCA, histogram, ROC, AUC vs layer); --separate-figs for single plots.
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
p.add_argument("--outdir", default="figures", help="where figures are saved")
p.add_argument("--separate-figs", action="store_true", help="also save the 4 results plots as separate PNGs")
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


# ------------------------------------------------------------------ figures
import os
os.makedirs(args.outdir, exist_ok=True)
MODE = args.mode
blue, red = "#3344e6", "#e62626"
zlab = "z-scored" if args.zscore else "raw"

# PCA of the hidden states used by the probe (same feature space as d); d expressed in PC coordinates
Hl = H_all[:, :, args.layer]
Xp = (Hl - R["mu"]) / R["sd"]
pca = PCA(n_components=2, svd_solver="full").fit(Xp)   # exact solver -> reproducible figures
Z2 = pca.transform(Xp)
d2 = pca.components_ @ R["d"]                            # d projected into the PCA plane

fpr, tpr, _ = roc_curve(y_te, R["s"])
auc_raw = np.array([evaluate(l, False)["auc"] for l in range(L)])
auc_z = np.array([evaluate(l, True)["auc"] for l in range(L)])
print("AUC raw :", np.round(auc_raw, 3))
print("AUC zsc :", np.round(auc_z, 3))


def plot_pca(ax):
    arrow = 0.6 * np.abs(Z2).max() * d2 / max(np.linalg.norm(d2), 1e-12)
    ax.scatter(*Z2[y == 0].T, s=12, c=blue, alpha=.5, label="Human")
    ax.scatter(*Z2[y == 1].T, s=12, c=red, alpha=.5, label="AI")
    ax.annotate("", xy=arrow, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="k"))
    ax.text(*arrow, "  d", fontsize=13, weight="bold")
    ax.set(xlabel="PC 1", ylabel="PC 2", title=f"PCA, layer {args.layer} ({zlab})\n"
           f"arrow = d in PCA plane (|proj| = {np.linalg.norm(d2):.2f})")
    ax.legend(); ax.grid(alpha=.3); ax.set_aspect("equal", "datalim")


def plot_hist(ax):
    bins = np.linspace(R["s"].min(), R["s"].max(), 30)
    ax.hist(R["s"][y_te == 0], bins, color=blue, alpha=.5, density=True, label="Human")
    ax.hist(R["s"][y_te == 1], bins, color=red, alpha=.5, density=True, label="AI")
    ax.axvline(R["tau"], ls="--", c="k", lw=2, label=fr"$\tau$ = {R['tau']:.2f}")
    ax.set(xlabel=r"projection score $s = h^\top d$", ylabel="density",
           title=f"Test-set scores (accuracy {100*R['acc']:.1f}%)")
    ax.legend()


def plot_roc(ax):
    ax.plot(fpr, tpr, c=red, lw=2.5, label=f"linear probe (AUC = {R['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC, layer {args.layer} ({zlab})")
    ax.legend(loc="lower right"); ax.grid(alpha=.3); ax.set_aspect("equal")


def plot_auc(ax):
    ax.plot(range(L), auc_raw, "-o", c=red, lw=2, label="raw")
    ax.plot(range(L), auc_z, "-s", c=blue, lw=2, label="z-scored (train stats)")
    for a, c in ((auc_raw, red), (auc_z, blue)):
        b = a.argmax(); ax.plot(b, a[b], "*", ms=16, c=c, mec="k")
    ax.set(xlabel="layer (0 = embeddings, 12 = last block)", ylabel="test AUC",
           title=f"AUC vs layer (best raw: L{auc_raw.argmax()} {auc_raw.max():.3f}; "
                 f"best z: L{auc_z.argmax()} {auc_z.max():.3f})")
    ax.title.set_fontsize(10)
    ax.set_xticks(range(L)); ax.grid(alpha=.3); ax.legend()


def results_figure(path):
    """2x2 results figure: PCA, histogram, ROC, AUC vs layer."""
    fig, axs = plt.subplots(2, 2, figsize=(13, 10.5))
    for f, ax in zip((plot_pca, plot_hist, plot_roc, plot_auc), axs.ravel()):
        f(ax)
    sup = "SIMULATED DATA (synthetic mode)" if SIM else "Real mode: " + args.model
    fig.suptitle(f"{sup} - layer {args.layer}, {zlab}", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pipeline_figure(path_base, n_show=30):
    """Pipeline diagram (text -> observer LLM -> h -> hidden space -> s = h'd -> s > tau ?)
    drawn in matplotlib. The hidden-space box shows the ACTUAL 2-D PCA projection of the
    hidden states (n_show points per class) and d projected into the PCA plane.
    Layout coordinates are in pixels of a 1568 x 515 canvas (y axis downwards)."""
    from matplotlib.patches import FancyBboxPatch
    with plt.rc_context({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix"}):
        W, Hh = 1568, 515
        fig = plt.figure(figsize=(W / 100, Hh / 100), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(Hh, 0); ax.axis("off")

        def box(x0, y0, x1, y1, fc, text=None, fs=27, ls="-", lw=1.3, r=16, z=1):
            ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                        fc=fc, ec="k", lw=lw, ls=ls, zorder=z))
            if text:
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs,
                        linespacing=1.35, zorder=z + 1)

        def arrow(x0, y0, x1, y1):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=3,
                        arrowprops=dict(arrowstyle="-|>", lw=2.2, color="k", shrinkA=0, shrinkB=0, mutation_scale=24))

        box(25, 80, 328, 200, "#EAEAFF", "Human-written\ntext")
        box(25, 268, 328, 388, "#FFEAEA", "AI-generated\ntext")
        arrow(330, 140, 472, 140); arrow(330, 327, 472, 327)

        box(475, 118, 690, 338, "white", ls=(0, (4, 3)), r=22)           # observer LLM (dashed)
        for k in range(4):                                 # 4 stacked layers; ONE (k = 2) is the layer read
            y0 = 133 + 50 * k
            ax.add_patch(plt.Rectangle((495, y0), 173, 35, fc="#FFD9A0" if k == 2 else "#EBEBEB",
                                       ec="k", lw=1.3, zorder=2))
        llm = "GPT-2" if args.model == "gpt2" else args.model
        ax.text(582, 88, f"Observer LLM ({llm})", ha="center", va="center", fontsize=23)
        ax.text(582, 372, r"one layer $\ell$ read", ha="center", va="center", fontsize=23)

        DX = 36                                            # extra gap so the longer "h (mean-pooled)" label fits
        arrow(690, 225, 808 + DX, 225)
        ax.text(749 + DX / 2, 176, r"$\mathbf{h}$", ha="center", va="center", fontsize=23)
        ax.text(749 + DX / 2, 202, "(mean-pooled)", ha="center", va="center", fontsize=15)

        box(812 + DX, 85, 1140 + DX, 366, "white", r=18)                    # 2-D PCA of the hidden states
        ax.text(976 + DX, 47, "2-D PCA of hidden states", ha="center", va="center", fontsize=23)
        arrow(1142 + DX, 225, 1245 + DX, 225)

        box(1250 + DX, 168, 1510 + DX, 285, "#FFFFDD", "Projection\nscore $s = \\mathbf{h}^{\\top}\\mathbf{d}$", fs=25)
        arrow(1380 + DX, 287, 1380 + DX, 362)
        box(1250 + DX, 365, 1510 + DX, 485, "#E6FFE6", "$s > \\tau$ ?\nAI : Human", fs=27)

        # --- actual data inside the hidden-space box
        r_ = np.random.default_rng(args.seed + 1)
        sel = np.concatenate([r_.choice(np.where(y == c)[0], min(n_show, (y == c).sum()), replace=False)
                              for c in (0, 1)])
        P, ys = Z2[sel], y[sel]
        x0, y0, x1, y1 = 822 + DX, 95, 1130 + DX, 356
        iax = fig.add_axes([x0 / W, 1 - y1 / Hh, (x1 - x0) / W, (y1 - y0) / Hh], facecolor="none")
        iax.scatter(*P[ys == 0].T, s=55, c="#6464F0", zorder=3, lw=0)
        iax.scatter(*P[ys == 1].T, s=55, c="#F04646", zorder=3, lw=0)
        # d = (mu_AI - mu_Human)/||.|| is linear, so in the PCA plane it points from the projected
        # Human centroid to the projected AI centroid: draw it exactly there (start in the blue
        # cluster, tip in the red cluster). Length is for display only; direction is exact.
        c_h, c_a = Z2[y == 0].mean(0), Z2[y == 1].mean(0)
        u = d2 / max(np.linalg.norm(d2), 1e-12)
        # equal x/y scale (so the direction of d is not distorted), limits fitted to the box shape
        hx = 1.18 * max(np.abs(P[:, 0]).max(), np.abs(c_h[0]), np.abs(c_a[0]))
        hy = hx * (y1 - y0) / (x1 - x0)
        hy = max(hy, 1.18 * np.abs(P[:, 1]).max()); hx = hy * (x1 - x0) / (y1 - y0)
        iax.annotate("", xy=c_a, xytext=c_h, zorder=4,
                     arrowprops=dict(arrowstyle="-|>", lw=2.6, color="k", mutation_scale=24, shrinkA=0, shrinkB=0))
        # label "d": among positions on a ring around the arrow tip, pick the one farthest from all
        # plotted points and from the arrow shaft, so the label never sits on data
        best, best_clear = None, -1.0
        for rad in (0.18, 0.24, 0.30):
            for ang in np.linspace(0, 2 * np.pi, 48, endpoint=False):
                cand = c_a + rad * hy * np.array([np.cos(ang), np.sin(ang)])
                if abs(cand[0]) > 0.9 * hx or abs(cand[1]) > 0.9 * hy:
                    continue
                t = np.clip(np.dot(cand - c_h, c_a - c_h) / np.dot(c_a - c_h, c_a - c_h), 0, 1)
                shaft = np.linalg.norm(cand - (c_h + t * (c_a - c_h)))
                clear = min(np.linalg.norm(P - cand, axis=1).min(), shaft)
                if clear > best_clear:
                    best, best_clear = cand, clear
        iax.text(*best, r"$\mathbf{d}$", fontsize=26, ha="center", va="center", zorder=5)
        iax.set_xlim(-hx, hx); iax.set_ylim(-hy, hy)
        iax.set_xticks([]); iax.set_yticks([])
        for sp in iax.spines.values():
            sp.set_visible(False)

        if SIM:
            ax.text(976 + DX, 408, "simulated hidden states\n(illustrative, not experimental)", ha="center",
                    va="center", fontsize=17, style="italic", color="#b00020", linespacing=1.3)
        for ext_ in ("png", "pdf"):
            fig.savefig(f"{path_base}.{ext_}", dpi=300, facecolor="white")
        plt.close(fig)


pipeline_figure(os.path.join(args.outdir, f"pipeline_{MODE}"))
results_figure(os.path.join(args.outdir, f"results_{MODE}.png"))

if args.separate_figs:                                   # optional single-plot versions
    for name, f, size in (("fig1_pca", plot_pca, (7, 6)), ("fig2_hist", plot_hist, (7, 4.5)),
                          ("fig3_roc", plot_roc, (5.5, 5.5)), ("fig4_auc_vs_layer", plot_auc, (7.5, 4.5))):
        fig, ax = plt.subplots(figsize=size); f(ax)
        fig.savefig(os.path.join(args.outdir, f"{name}_{MODE}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
print("figures saved in", os.path.abspath(args.outdir))
