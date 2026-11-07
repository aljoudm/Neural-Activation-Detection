"""
extract_hidden_states.py
------------------------
PIPELINE steps 1-2: pass human-written and AI-generated texts through an
"observer LLM" (GPT-2) and extract one mean-pooled hidden-state vector h per text
from a chosen layer.  Output: hidden_states.mat  (read by linear_probe_ai_detection.m)

Saved variables
    H      (N x D)        hidden states of layer LAYER (D = 768 for GPT-2 small)
    y      (N x 1)        labels, 0 = Human, 1 = AI
    H_all  (N x D x 13)   hidden states of ALL layers (index 1 = embeddings,
                          index 13 = last block) -> used for the MATLAB layer sweep
    LAYER  scalar         the layer stored in H

Required packages:  pip install torch transformers numpy scipy
"""

import os
import numpy as np
import torch
from scipy.io import savemat
from transformers import GPT2Model, GPT2TokenizerFast

# ----------------------------- configuration --------------------------------
MODEL_NAME = "gpt2"      # 12 transformer blocks, D = 768
LAYER      = 8           # 0 = embedding output, 1..12 = output of block 1..12
MAX_LEN    = 256         # truncate long texts (tokens)
BATCH_SIZE = 16
OUT_FILE   = "hidden_states.mat"

# Provide your own data: one text per line in these files ...
HUMAN_FILE = "human_texts.txt"
AI_FILE    = "ai_texts.txt"

# ... or edit these tiny placeholder lists (far too small for real results!).
human_texts = [
    "I missed the bus again, so I walked and got soaked. Not my best Monday.",
    "My grandmother's recipe never had measurements, just 'enough' and 'a bit'.",
]
ai_texts = [
    "Artificial intelligence is a transformative technology that offers numerous benefits across various industries.",
    "In conclusion, it is important to consider multiple perspectives when evaluating this complex issue.",
]


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


if os.path.exists(HUMAN_FILE) and os.path.exists(AI_FILE):
    human_texts, ai_texts = read_lines(HUMAN_FILE), read_lines(AI_FILE)

# ----------------------------- load observer LLM ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token          # GPT-2 has no pad token
model = GPT2Model.from_pretrained(MODEL_NAME).to(device).eval()


@torch.no_grad()
def extract(texts):
    """Return an (N x D x n_layers) array: mean-pooled hidden state of every layer."""
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LEN).to(device)
        # PIPELINE step 1: texts go through the observer LLM.
        res = model(**enc, output_hidden_states=True)
        # res.hidden_states = tuple of (n_layers = 13) tensors of shape (B, T, D)
        hs = torch.stack(res.hidden_states, dim=0)              # (13, B, T, D)
        # PIPELINE step 2: mean-pool over tokens, ignoring padding.
        mask = enc["attention_mask"][None, :, :, None].float()  # (1, B, T, 1)
        pooled = (hs * mask).sum(dim=2) / mask.sum(dim=2)       # (13, B, D)
        out.append(pooled.permute(1, 2, 0).cpu().numpy())       # (B, D, 13)
    return np.concatenate(out, axis=0)


H_h = extract(human_texts)
H_a = extract(ai_texts)

H_all = np.concatenate([H_h, H_a], axis=0).astype(np.float64)    # (N, D, 13)
y = np.concatenate([np.zeros(len(human_texts)),
                    np.ones(len(ai_texts))]).reshape(-1, 1)      # 0 = Human, 1 = AI
H = H_all[:, :, LAYER]                                           # chosen layer (N, D)

savemat(OUT_FILE, {"H": H, "y": y, "H_all": H_all, "LAYER": LAYER})
print(f"Saved {OUT_FILE}: H {H.shape}, y {y.shape}, H_all {H_all.shape}, layer {LAYER}")
