"""Memory-safe assembly of a subsampled beta matrix from the 4 GSE40279 parts.
Keeps ~30k CpGs (every k-th row of part 1), then aligns the same CpGs across parts.
Output: betas_sub.npz  (X: samples x cpgs, ages, sample_ids, cpg_ids)
"""
import gzip, numpy as np, csv

parts = [
    "GSE40279_average_beta_GSM989827-GSM989990.txt.gz",
    "GSE40279_average_beta_GSM989991-GSM990299.txt.gz",
    "GSE40279_average_beta_GSM990300-GSM990463.txt.gz",
    "GSE40279_average_beta_GSM990464-GSM990627.txt.gz",
]
TARGET = 30000  # approx CpGs to keep

# --- pass 1: determine total CpG count in part 1 to set stride ---
with gzip.open(parts[0], "rt") as f:
    header = f.readline()
    ncpg = sum(1 for _ in f)
stride = max(1, ncpg // TARGET)
print(f"part1 CpGs={ncpg}, stride={stride}")

# --- read part 1: collect chosen CpG ids + values ---
def read_part(path, keep_ids=None):
    """Return (sample_ids, {cpg_id: np.array(values)}). If keep_ids given, only those rows."""
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        samples = [h.lstrip("X") for h in header[1:]]
        data = {}
        for i, line in enumerate(f):
            parts_ = line.rstrip("\n").split("\t")
            cid = parts_[0]
            if keep_ids is None:
                if i % stride != 0:
                    continue
            else:
                if cid not in keep_ids:
                    continue
            try:
                vals = np.array(parts_[1:], dtype=np.float32)
            except ValueError:
                vals = np.array([np.nan if v in ("", "NA", "NaN") else float(v)
                                 for v in parts_[1:]], dtype=np.float32)
            data[cid] = vals
        return samples, data

s1, d1 = read_part(parts[0])
keep = set(d1.keys())
print(f"kept {len(keep)} CpGs from part1, {len(s1)} samples")

all_samples = list(s1)
# dict cpg -> list of arrays aligned to growing sample list
col_blocks = [d1]
sample_blocks = [s1]
for p in parts[1:]:
    s, d = read_part(p, keep_ids=keep)
    sample_blocks.append(s)
    col_blocks.append(d)
    print(f"{p}: {len(s)} samples, {len(d)} CpGs matched")

# intersect CpGs present in ALL parts
common = set(col_blocks[0].keys())
for d in col_blocks[1:]:
    common &= set(d.keys())
common = sorted(common)
print(f"common CpGs across all parts: {len(common)}")

# assemble matrix: rows=samples (concat), cols=common CpGs
sample_ids = []
mats = []
for s, d in zip(sample_blocks, col_blocks):
    block = np.vstack([d[c] for c in common]).T  # samples x cpgs
    mats.append(block)
    sample_ids.extend(s)
X = np.vstack(mats)
sample_ids = np.array(sample_ids)
print("X shape (samples x cpgs):", X.shape)

# --- attach ages ---
age_map = {}
with open("ages.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        age_map[row["sample_id"]] = float(row["age"])
ages = np.array([age_map.get(sid, np.nan) for sid in sample_ids])
mask = ~np.isnan(ages)
X, sample_ids, ages = X[mask], sample_ids[mask], ages[mask]
print(f"samples with age: {mask.sum()}; age range {ages.min()}-{ages.max()}")

# drop CpGs with any NaN
colmask = ~np.isnan(X).any(axis=0)
X = X[:, colmask]
cpg_ids = np.array(common)[colmask]
print(f"final X: {X.shape}, CpGs after NaN drop: {len(cpg_ids)}")

np.savez_compressed("betas_sub.npz", X=X, ages=ages,
                    sample_ids=sample_ids, cpg_ids=cpg_ids)
print("saved betas_sub.npz")
