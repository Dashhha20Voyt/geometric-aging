"""
Geometric-topological aging invariants on the REAL GSE40279 (Hannum) cohort.
656 whole-blood samples, ages 19-101, ~31.5k CpGs (subsampled).

Pipeline
  1. Feature selection: top-K CpGs by |corr with age| (supervised manifold axis).
  2. Manifold: kNN graph -> Isomap/diffusion geodesic distances (graph shortest path).
     Low-dim spectral embedding for visualization + trajectory geometry.
  3. Reference trajectory gamma_H: Frechet(Euclid-embedding) mean per age bin.
  4. Invariants:
       MV(age)  = metric speed of gamma_H  = ||d gamma_H / d age||
       TDI(i)   = geodesic distance from sample i to gamma_H
       BRS(age) = local dispersion / branching proxy (EWS: variance + lag-1 AC of residuals)
       TE(age)  = entropy rate proxy = growth of neighbourhood dispersion with age
  5. Biological anchor: do MV / BRS peak near the 44 & 60 transitions (Shen 2024)?
  6. Age prediction on the manifold (kNN geodesic regression) vs scalar-baseline (ElasticNet).
     External validity via train/test split (MAE, Pearson r).
"""
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import kneighbors_graph, KNeighborsRegressor
from sklearn.manifold import SpectralEmbedding
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import shortest_path
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
R = {}

d = np.load("betas_sub.npz", allow_pickle=True)
X, ages = d["X"].astype(np.float64), d["ages"].astype(np.float64)
print("loaded", X.shape, "ages", ages.min(), ages.max())

# ---------------------------------------------------------------
# 1. Feature selection: top-K age-correlated CpGs
# ---------------------------------------------------------------
K = 2000
corr = np.array([pearsonr(X[:, j], ages)[0] for j in range(X.shape[1])])
top = np.argsort(-np.abs(corr))[:K]
Xs = X[:, top]
# standardize
Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)
R["n_samples"] = int(X.shape[0])
R["n_cpg_total"] = int(X.shape[1])
R["n_cpg_selected"] = int(K)
R["max_abs_age_corr_cpg"] = round(float(np.nanmax(np.abs(corr))), 4)

# ---------------------------------------------------------------
# 2. Manifold: kNN geodesic distances + spectral embedding
# ---------------------------------------------------------------
kNN = 15
G = kneighbors_graph(Xs, n_neighbors=kNN, mode="distance")
Dgeo = shortest_path(G, method="D", directed=False)
finite = np.isfinite(Dgeo)
R["geodesic_finite_fraction"] = round(float(finite.mean()), 4)
# replace inf (disconnected) with max finite for downstream robustness
Dfix = Dgeo.copy()
Dfix[~finite] = np.nanmax(Dgeo[finite])

emb = SpectralEmbedding(n_components=3, affinity="nearest_neighbors",
                        n_neighbors=kNN, random_state=0).fit_transform(Xs)
# orient component 1 to increase with age
if pearsonr(emb[:, 0], ages)[0] < 0:
    emb[:, 0] *= -1
r_emb_age = pearsonr(emb[:, 0], ages)[0]
R["embedding_axis1_age_pearson"] = round(float(r_emb_age), 4)

# ---------------------------------------------------------------
# 3. Reference trajectory gamma_H: mean embedding per age bin
# ---------------------------------------------------------------
bin_edges = np.arange(19, 102, 3.0)
centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
gamma = []
valid_centers = []
for lo, hi, c in zip(bin_edges[:-1], bin_edges[1:], centers):
    m = (ages >= lo) & (ages < hi)
    if m.sum() >= 3:
        gamma.append(emb[m].mean(0))
        valid_centers.append(c)
gamma = np.array(gamma)
valid_centers = np.array(valid_centers)

# ---------------------------------------------------------------
# 4a. MV: metric speed of gamma_H vs age (finite difference)
# ---------------------------------------------------------------
dg = np.gradient(gamma, valid_centers, axis=0)
MV = norm(dg, axis=1)

# ---------------------------------------------------------------
# 4b. BRS / EWS: local dispersion of embedding residuals per age bin
#     residual = distance of samples to gamma_H at their age; variance & lag-1 AC
# ---------------------------------------------------------------
# nearest gamma point (by age) for each sample
def gamma_at(age):
    idx = np.argmin(np.abs(valid_centers - age))
    return gamma[idx]
resid = np.array([norm(emb[i] - gamma_at(ages[i])) for i in range(len(ages))])
TDI_all = resid.copy()  # per-sample trajectory-deviation index

disp, ac1, ent = [], [], []
order_centers = []
for lo, hi, c in zip(bin_edges[:-1], bin_edges[1:], centers):
    m = (ages >= lo) & (ages < hi)
    if m.sum() >= 5:
        r = resid[m]
        disp.append(r.var())
        # lag-1 autocorr of residuals ordered by age within bin
        o = np.argsort(ages[m]); rr = r[o]
        ac1.append(np.corrcoef(rr[:-1], rr[1:])[0, 1] if len(rr) > 2 else np.nan)
        # entropy-rate proxy: log determinant of local covariance (spread of states)
        pts = emb[m]
        cov = np.cov(pts.T) + 1e-6 * np.eye(pts.shape[1])
        ent.append(0.5 * np.log(np.linalg.det(cov)))
        order_centers.append(c)
disp = np.array(disp); ac1 = np.array(ac1); ent = np.array(ent)
order_centers = np.array(order_centers)
BRS = disp / disp.max()          # normalized bifurcation-risk proxy

# ---------------------------------------------------------------
# 5. Biological anchor: peaks near 44 and 60 ?
# ---------------------------------------------------------------
def peak_near(centers_arr, signal, target, window=6):
    m = np.abs(centers_arr - target) <= window
    if m.sum() == 0:
        return None
    local_max = centers_arr[m][np.argmax(signal[m])]
    # is the windowed region elevated vs global median?
    elevated = np.median(signal[m]) > np.median(signal)
    return float(local_max), bool(elevated)

R["MV_peak_near_44"] = peak_near(valid_centers, MV, 44)
R["MV_peak_near_60"] = peak_near(valid_centers, MV, 60)
R["BRS_peak_near_44"] = peak_near(order_centers, BRS, 44)
R["BRS_peak_near_60"] = peak_near(order_centers, BRS, 60)
# global MV structure: correlation of MV with |age-distance to nearest transition|
R["MV_mean"] = round(float(MV.mean()), 4)
R["MV_max_age"] = round(float(valid_centers[np.argmax(MV)]), 1)
R["BRS_max_age"] = round(float(order_centers[np.argmax(BRS)]), 1)

# ---------------------------------------------------------------
# 6. Age prediction: manifold kNN geodesic regression vs ElasticNet baseline
# ---------------------------------------------------------------
idx = np.arange(len(ages))
tr, te = train_test_split(idx, test_size=0.3, random_state=0)

# manifold kNN regression using precomputed geodesic distances
knn = KNeighborsRegressor(n_neighbors=20, metric="precomputed")
knn.fit(Dfix[np.ix_(tr, tr)], ages[tr])
pred_geo = knn.predict(Dfix[np.ix_(te, tr)])
mae_geo = np.mean(np.abs(pred_geo - ages[te]))
r_geo = pearsonr(pred_geo, ages[te])[0]

# scalar baseline: ElasticNet on selected CpGs (a linear "clock")
en = ElasticNetCV(cv=5, random_state=0, max_iter=5000)
en.fit(Xs[tr], ages[tr])
pred_en = en.predict(Xs[te])
mae_en = np.mean(np.abs(pred_en - ages[te]))
r_en = pearsonr(pred_en, ages[te])[0]

R["age_pred_manifold_knn_MAE"] = round(float(mae_geo), 3)
R["age_pred_manifold_knn_pearson"] = round(float(r_geo), 4)
R["age_pred_elasticnet_MAE"] = round(float(mae_en), 3)
R["age_pred_elasticnet_pearson"] = round(float(r_en), 4)

# TDI clinical-ish check: does deviation correlate with anything?  (no outcome here) -> report distribution
R["TDI_mean"] = round(float(TDI_all.mean()), 4)
R["TDI_std"] = round(float(TDI_all.std()), 4)

print(json.dumps(R, indent=2))
json.dump(R, open("real_results.json", "w"), indent=2)

# ---------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(16, 9))

sc = ax[0, 0].scatter(emb[:, 0], emb[:, 1], c=ages, cmap="viridis", s=14)
ax[0, 0].plot(gamma[:, 0], gamma[:, 1], "r-o", ms=3, lw=1.5, label="gamma_H")
ax[0, 0].set_title(f"Methylation manifold (GSE40279)\naxis1 vs age r={R['embedding_axis1_age_pearson']}")
ax[0, 0].legend(); plt.colorbar(sc, ax=ax[0, 0], label="age")

ax[0, 1].plot(valid_centers, MV, "-o", ms=3)
for t in (44, 60):
    ax[0, 1].axvline(t, color="r", ls="--", alpha=0.6)
ax[0, 1].set_title("Manifold Velocity MV(age)\nred = 44/60 transitions")
ax[0, 1].set_xlabel("age"); ax[0, 1].set_ylabel("||d gamma/d age||")

ax[0, 2].plot(order_centers, BRS, "-o", ms=3, label="BRS (norm. dispersion)")
ax[0, 2].plot(order_centers, ac1, "-s", ms=3, label="lag-1 autocorr")
for t in (44, 60):
    ax[0, 2].axvline(t, color="r", ls="--", alpha=0.6)
ax[0, 2].set_title("BRS / early-warning signals"); ax[0, 2].set_xlabel("age"); ax[0, 2].legend()

ax[1, 0].plot(order_centers, ent, "-o", ms=3)
ax[1, 0].set_title("Trajectory Entropy proxy\n(log-det local covariance)")
ax[1, 0].set_xlabel("age"); ax[1, 0].set_ylabel("entropy proxy")

ax[1, 1].scatter(ages[te], pred_geo, s=14, alpha=0.6)
ax[1, 1].plot([19, 101], [19, 101], "r--")
ax[1, 1].set_title(f"Manifold kNN age prediction\nMAE={R['age_pred_manifold_knn_MAE']}, r={R['age_pred_manifold_knn_pearson']}")
ax[1, 1].set_xlabel("true age"); ax[1, 1].set_ylabel("predicted")

ax[1, 2].hist(TDI_all, bins=30, color="steelblue")
ax[1, 2].set_title(f"Trajectory Deviation Index\nmean={R['TDI_mean']}, std={R['TDI_std']}")
ax[1, 2].set_xlabel("geodesic deviation from gamma_H")

plt.tight_layout()
plt.savefig("real_figure.png", dpi=130)
print("saved real_figure.png")
