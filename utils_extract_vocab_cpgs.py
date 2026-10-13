"""Stream the full GSE40279 beta matrix and extract only the CpGs in the
MethylGPT type3 vocabulary (49,156 probes), for all samples. Output a matrix
aligned to the vocab order with NaN for missing CpGs (model handles them)."""
import sys, gzip, numpy as np, pandas as pd

VOCAB_CSV = "probe_ids_type3.csv"
OUT = "methylgpt_input.npz"

# vocab CpGs in order
vocab = pd.read_csv(VOCAB_CSV)["illumina_probe_id"].tolist()
vocab_set = set(vocab)
vocab_index = {c: i for i, c in enumerate(vocab)}
n_vocab = len(vocab)
print(f"vocab CpGs: {n_vocab}", flush=True)

# ages
ages_df = pd.read_csv("ages.csv")  # geo, sample_id, age
# sample_id as string to match header X<id>

reader = sys.stdin  # gz-decompressed stream piped in
header = reader.readline().rstrip("\n").split("\t")
# columns: ID_REF, X1001, X1002, ...
col_samples = header[1:]
sample_ids = [c[1:] if c.startswith("X") else c for c in col_samples]
print(f"samples in beta file: {len(sample_ids)}", flush=True)

# map sample_id -> age
age_map = {str(r.sample_id): r.age for r in ages_df.itertuples()}
ages = np.array([age_map.get(sid, np.nan) for sid in sample_ids], dtype=float)
n_with_age = np.sum(~np.isnan(ages))
print(f"samples with known age: {n_with_age}", flush=True)

n_samp = len(sample_ids)
# matrix vocab x samples, fill NaN
X = np.full((n_vocab, n_samp), np.nan, dtype=np.float32)

found = 0
line_no = 0
for line in reader:
    line_no += 1
    if line_no % 50000 == 0:
        print(f"  read {line_no} rows, matched {found}", flush=True)
    tab = line.find("\t")
    if tab < 0:
        continue
    cpg = line[:tab]
    if cpg not in vocab_set:
        continue
    vals = line.rstrip("\n").split("\t")[1:]
    if len(vals) != n_samp:
        continue
    row = np.array([float(v) if v not in ("", "NA", "NaN") else np.nan for v in vals], dtype=np.float32)
    X[vocab_index[cpg]] = row
    found += 1

print(f"total rows read: {line_no}, vocab CpGs found: {found} / {n_vocab}", flush=True)

# transpose to samples x vocab
Xs = X.T  # (n_samp, n_vocab)
np.savez_compressed(OUT,
                    X=Xs,
                    ages=ages,
                    sample_ids=np.array(sample_ids),
                    cpg_ids=np.array(vocab),
                    coverage=np.array([found, n_vocab]))
print(f"saved {OUT}: X shape {Xs.shape}", flush=True)
print(f"per-sample NaN fraction (mean): {np.mean(np.isnan(Xs)):.3f}", flush=True)
