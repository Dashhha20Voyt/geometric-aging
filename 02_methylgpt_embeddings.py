"""Extract MethylGPT sample-level embeddings for GSE40279 on CPU.

CPU has no flash-attention, so full-length (49k) attention is infeasible
(~300 GB). We instead feed the model a FIXED subset of high-variance,
fully-observed CpGs (same set for every sample), tokenized directly with the
model's learned CpG token embeddings. The <cls> token embedding is used as the
sample-level representation, exactly as in the model's get_cell_embeddings.
"""
import os, sys, json, glob
sys.path.insert(0, "/home/user/workspace/MethylGPT/methylgpt/modules/scGPT")
sys.path.insert(0, "/home/user/workspace/MethylGPT")
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch

from methylgpt.model.methyl_model import MethylGPTModel
from methylgpt.model.methyl_vocab import MethylVocab

REPO = "/home/user/workspace/MethylGPT"
MODEL_DIR = glob.glob(REPO + "/pretrained_models/*/")[0]
MODEL_PT = glob.glob(MODEL_DIR + "*.pt")[0]
ARGS = json.load(open(MODEL_DIR + "args.json"))
PROBE = "/home/user/workspace/gse40279/probe_ids_type3.csv"

N_SUBSET = 2000  # sequence length fed to the model (CPU-safe)

print("model checkpoint:", MODEL_PT, flush=True)
print("layer_size", ARGS["layer_size"], "nlayers", ARGS["nlayers"], flush=True)

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
config = dict(ARGS)
config.update(dict(load_model=False, probe_id_dir=PROBE, mask_ratio=0.0,
                   mask_value=-1, pad_value=-2, pad_token=pad_token,
                   special_tokens=special_tokens, fast_transformer=False,
                   dropout=0.0))
config["max_seq_len"] = config["n_hvg"] + 1

vocab = MethylVocab(PROBE, pad_token, special_tokens, save_dir=None)
print("vocab len", len(vocab), flush=True)

model = MethylGPTModel(config, vocab)
state = torch.load(MODEL_PT, map_location="cpu", weights_only=True)
remap = {}
for k, v in state.items():
    nk = k.replace(".self_attn.Wqkv.weight", ".self_attn.in_proj_weight")
    nk = nk.replace(".self_attn.Wqkv.bias", ".self_attn.in_proj_bias")
    remap[nk] = v
model_dict = model.state_dict()
compatible = {k: v for k, v in remap.items() if k in model_dict and v.shape == model_dict[k].shape}
missing = set(model_dict) - set(compatible)
print(f"loaded {len(compatible)}/{len(model_dict)} params; missing {len(missing)}", flush=True)
if missing:
    print("MISSING:", sorted(missing)[:20], flush=True)
model_dict.update(compatible)
model.load_state_dict(model_dict)
model.eval()
torch.set_grad_enabled(False)
model.float()
print("model params:", sum(p.numel() for p in model.parameters()), flush=True)

# ---- data ----
d = np.load("/home/user/workspace/gse40279/methylgpt_input.npz", allow_pickle=True)
X = d["X"].astype(np.float32)     # (656, 49156), NaN where missing
ages = d["ages"].astype(float)
sample_ids = d["sample_ids"]
cpg_ids = d["cpg_ids"]            # vocab order
print("data", X.shape, flush=True)

# select fully-observed, high-variance CpGs
obs = ~np.isnan(X)
frac_obs = obs.mean(axis=0)
fully = np.where(frac_obs >= 0.999)[0]
print("fully-observed CpGs:", len(fully), flush=True)
var = np.nanvar(X[:, fully], axis=0)
order = fully[np.argsort(var)[::-1]]
sel = np.sort(order[:N_SUBSET])
print("selected", len(sel), "CpGs for the model input", flush=True)

# token ids for the selected CpGs (vocab index for each cg id)
sel_cpg = cpg_ids[sel]
tok = np.array([vocab[c] for c in sel_cpg], dtype=np.int64)
cls_id = vocab["<cls>"]

Xsel = X[:, sel]
# impute any residual NaN with column mean
col_mean = np.nanmean(Xsel, axis=0)
inds = np.where(np.isnan(Xsel))
Xsel[inds] = np.take(col_mean, inds[1])

n = Xsel.shape[0]
BATCH = 4
torch.set_num_threads(2)

# resumable checkpoint of per-sample embeddings
CKPT = "/home/user/workspace/gse40279/emb_ckpt.npy"
if os.path.exists(CKPT):
    E_all = np.load(CKPT)
    print("resuming from checkpoint, rows filled:", int((~np.isnan(E_all[:,0])).sum()), flush=True)
else:
    E_all = np.full((n, 64), np.nan, dtype=np.float32)

for start in range(0, n, BATCH):
    end = min(start + BATCH, n)
    if not np.isnan(E_all[start:end, 0]).any():
        continue  # already done
    b = end - start
    vals = torch.from_numpy(Xsel[start:end]).float()           # (b, L)
    vals = torch.cat([torch.zeros(b, 1), vals], dim=1)          # (b, L+1)
    genes = torch.from_numpy(np.concatenate([[cls_id], tok]))   # (L+1,)
    genes = genes.unsqueeze(0).repeat(b, 1).long()              # (b, L+1)
    cell_emb = model.get_cell_embeddings(genes, vals)
    E_all[start:end] = cell_emb.cpu().numpy()
    if (start // BATCH) % 10 == 0:
        np.save(CKPT, E_all)
    print(f"  batch {start}-{end} done", flush=True)

np.save(CKPT, E_all)
E = E_all
print("embeddings", E.shape, flush=True)
np.savez_compressed("/home/user/workspace/gse40279/methylgpt_embeddings.npz",
                    E=E, ages=ages, sample_ids=sample_ids,
                    n_cpg_used=np.array([len(sel)]))
print("saved methylgpt_embeddings.npz", flush=True)
