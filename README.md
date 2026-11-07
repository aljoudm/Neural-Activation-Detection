# Neural-Activation-Detection

A **linear probe** for detecting AI-generated vs. human-written text from an LLM's hidden states.
It accompanies a research presentation on the idea that *LLMs process human-written and
AI-generated texts through distinct neural activation patterns*.

> **Important:** the figures in this repo (`figures/`) come from **simulated (synthetic) data**.
> They demonstrate that the code works; they are **not** results on real GPT-2 activations or
> real human/AI text.

## Pipeline

```
text ──► observer LLM (GPT-2) ──► h  (mean-pooled hidden state of one layer, D = 768)
                                   │
        d = (μ_AI − μ_Human) / ‖μ_AI − μ_Human‖      difference of class means (train set only)
                                   │
                  s = hᵀ d                            projection score
                                   │
        τ = ½ (mean s of Human + mean s of AI)        midpoint (train set only)
                                   │
                  s > τ ?  →  AI : Human
```

* Stratified **70/30** train/test split; `d` and `τ` (and the optional z-score statistics) are
  computed from the **training set only**. Reported metrics are on the test set.
* The model is run **once**; the hidden states of all 13 layers (embeddings + 12 blocks) are cached,
  so the analysis can be repeated for every layer cheaply.
* Optional `--zscore` standardises each feature with training-set statistics. This is meant to
  tame GPT-2's few very-large-magnitude "outlier" dimensions, which can dominate a raw mean difference.

## Install

```bash
pip install -r requirements.txt
```

`numpy`, `scikit-learn` and `matplotlib` are enough for synthetic mode. `torch` and `transformers`
are needed only for real mode.

## Run

```bash
python linear_probe.py                          # synthetic mode (simulated data)
python linear_probe.py --zscore                 # ... with z-scoring
python linear_probe.py --mode real --layer 8 --zscore
```

Options: `--layer` (0 = embeddings … 12), `--zscore`, `--human-file`, `--ai-file`, `--model`,
`--max-len`, `--batch-size`, `--seed`, `--outdir` (where the PNGs are written).

Outputs (written to `figures/`):

* `pipeline_<mode>.png` / `.pdf` (300 dpi): the pipeline diagram, with the hidden-representation box
  showing the *actual* 2-D PCA projection of the hidden states (30 points per class) and `d` as an arrow.
  Labelled "simulated data" in synthetic mode.
* `results_<mode>.png`: one 2x2 figure with the PCA scatter (arrow = `d`), histograms of `s` with a dashed
  line at `τ`, the ROC curve with AUC, and test AUC vs. layer with and without z-scoring.
* `--separate-figs` additionally saves each of the four results plots as its own PNG.

`figures/pipeline_synthetic.*` and `figures/results_synthetic.png` were generated with
`python linear_probe.py --zscore`.

## Synthetic vs. real mode

| | Synthetic (default) | Real (`--mode real`) |
|---|---|---|
| Data | **Simulated** 768-D vectors, 400 per class | GPT-2 hidden states of your texts |
| Needs LLM | No | Yes (`torch`, `transformers`, downloads GPT-2 weights ~0.5 GB) |
| Input | none | `human_texts.txt` and `ai_texts.txt`, one text per line |
| Purpose | Demo of the pipeline and plots | The actual experiment |

The synthetic generator shifts the AI class by a small amount along one random direction per layer,
makes class separation peak in the middle layers (an inverted U), and adds three class-independent
high-variance dimensions to mimic GPT-2 outlier dimensions. **Therefore the layer curve and the
large raw-vs-z-score gap in `figures/` are properties of the simulation, not findings about GPT-2.**

The `human_texts.txt` / `ai_texts.txt` in this repo are tiny placeholder samples (15 short sentences
each) only to show the file format. Use hundreds or thousands of texts, matched for length and topic,
for any real conclusion.

## Repository layout

```
linear_probe.py            single-file Python implementation (synthetic + real modes)
human_texts.txt            sample human-written texts (placeholder)
ai_texts.txt               sample AI-style texts (placeholder)
requirements.txt
figures/                   pipeline_synthetic.{png,pdf}, results_synthetic.png and older single plots
                           (all from SIMULATED data)
matlab_version/            earlier version
    linear_probe_ai_detection.m   MATLAB probe (needs Statistics and Machine Learning Toolbox)
    extract_hidden_states.py      saves hidden_states.mat for the MATLAB script
```

## Status / caveats

* Synthetic mode has been run end to end and its figures inspected.
* Real mode (GPT-2 extraction in `linear_probe.py`) follows the same pipeline but had **not been
  verified end to end** when this README was written.
* Confounds such as text length, topic and domain can leak into a probe; control for them.

## Why a single direction can work

By the *linear representation hypothesis*, high-level properties are often encoded as directions in
activation space, so class means differ along one axis and a projection onto it is a usable readout.
Middle layers tend to separate best: early layers mostly encode token identity, and last layers
specialise for next-token prediction.
