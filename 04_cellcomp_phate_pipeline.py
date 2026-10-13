"""
VARIANT 1: reference-free cell-composition correction + PHATE on raw beta.

Data: GSE40279, 656 whole-blood samples, ages 19-101, 45,309 fully-observed CpGs
(same input matrix used for MethylGPT, for a fair comparison).

Steps
  1. Standardize CpGs; PCA to capture dominant variance (in blood this is largely
     cell composition -- ReFACTor rationale, Rahmani et al. Nat Methods 2016).
  2. Split PCs into age-associated (PROTECT) vs nuisance. Orthogonalize nuisance
     PCs against age, then regress them out of every CpG -> cell-composition-
     adjusted beta that RETAINS age-correlated variance.
  3. PHATE embedding on the adjusted matrix (denoised diffusion geometry).
  4. Recompute invariants MV / TDI / BRS / TE, test 44/60 peaks.
  5. Age prediction (manifold kNN geodesic vs ElasticNet) on adjusted data.
"""
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph, KNeighborsRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import shortest_path
import phate
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
os.environ["OMP_NUM_THREADS"] = "2"
R = {}
d = np.load("methylgpt_input.npz", allow_pickle=True)
X = d["X"].astype(np.float64)
ages = d["ages"].astype(np.float64)
obs = ~np.isnan(X)
fully = np.where(obs.mean(0) >= 0.999)[0]
X = X[:, fully]
# impute residual NaN with column mean
cm = np.nanmean(X, 0); ii = np.where(np.isnan(X)); X[ii] = np.take(cm, ii[1])
print("matrix", X.shape, "ages", ages.min(), ages.max(), flush=True)
R["n_samples"] = int(X.shape[0]); R["n_cpg"] = int(X.shape[1])

# ---- 1. standardize + PCA ----
Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
NPC = 30
pca = PCA(n_components=NPC, random_state=0)
PC = pca.fit_transform(Xz)            # (n, NPC)
evr = pca.explained_variance_ratio_
pc_age_r = np.array([pearsonr(PC[:, k], ages)[0] for k in range(NPC)])
print("PC var explained (top5):", np.round(evr[:5], 3), flush=True)
print("PC-age |r| (top8):", np.round(np.abs(pc_age_r[:8]), 3), flush=True)

# ---- 2. protect age-associated PCs, regress out nuisance ----
AGE_R_THRESH = 0.30
protect = np.where(np.abs(pc_age_r) >= AGE_R_THRESH)[0]
nuisance = np.array([k for k in range(NPC) if k not in protect])
R["protected_PCs"] = protect.tolist()
R["n_nuisance_PCs"] = int(len(nuisance))
R["PC_age_r"] = [round(float(x), 3) for x in pc_age_r]

# orthogonalize nuisance PCs against age (remove age-correlated part) so that
# regressing them out does NOT strip age signal
age_c = (ages - ages.mean())
N = PC[:, nuisance].copy()
for j in range(N.shape[1]):
    b = np.dot(N[:, j], age_c) / np.dot(age_c, age_c)
    N[:, j] = N[:, j] - b * age_c
# regress each CpG on orthogonalized nuisance PCs, keep residuals
lr = LinearRegression().fit(N, Xz)
Xadj = Xz - lr.predict(N)            # cell-composition-adjusted, age retained
# sanity: age correlation of leading adjusted PC
adj_pc = PCA(n_components=3, random_state=0).fit_transform(Xadj)
R["adj_leadPC_age_r"] = round(float(max(abs(pearsonr(adj_pc[:, k], ages)[0]) for k in range(3))), 4)
print("adjusted lead-PC |age r|", R["adj_leadPC_age_r"], flush=True)

# ---- 3. PHATE embedding (checkpointed) ----
if os.path.exists("phate_emb.npy"):
    emb = np.load("phate_emb.npy")
    print("loaded cached PHATE embedding", emb.shape, flush=True)
else:
    op = phate.PHATE(n_components=3, knn=15, decay=40, t="auto",
                     random_state=0, verbose=False, n_jobs=2)
    emb = op.fit_transform(Xadj)
    np.save("phate_emb.npy", emb)
    np.save("phate_Xadj.npy", Xadj.astype(np.float32))
    print("computed + cached PHATE embedding", emb.shape, flush=True)
if pearsonr(emb[:, 0], ages)[0] < 0:
    emb[:, 0] *= -1
R["phate_axis1_age_pearson"] = round(float(pearsonr(emb[:, 0], ages)[0]), 4)
print("PHATE axis1 vs age r", R["phate_axis1_age_pearson"], flush=True)

# ---- geodesics on adjusted data ----
kNN = 15
G = kneighbors_graph(Xadj, n_neighbors=kNN, mode="distance")
Dgeo = shortest_path(G, method="D", directed=False)
finite = np.isfinite(Dgeo)
R["geodesic_finite_fraction"] = round(float(finite.mean()), 4)
Dfix = Dgeo.copy(); Dfix[~finite] = np.nanmax(Dgeo[finite])

# ---- gamma_H reference trajectory (per age bin, in PHATE space) ----
bin_edges = np.arange(19, 102, 3.0)
centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
gamma, vc = [], []
for lo, hi, c in zip(bin_edges[:-1], bin_edges[1:], centers):
    m = (ages >= lo) & (ages < hi)
    if m.sum() >= 3:
        gamma.append(emb[m].mean(0)); vc.append(c)
gamma = np.array(gamma); vc = np.array(vc)
dg = np.gradient(gamma, vc, axis=0); MV = norm(dg, axis=1)

def gamma_at(a): return gamma[np.argmin(np.abs(vc - a))]
resid = np.array([norm(emb[i] - gamma_at(ages[i])) for i in range(len(ages))])
TDI = resid.copy()
disp, ac1, ent, oc = [], [], [], []
for lo, hi, c in zip(bin_edges[:-1], bin_edges[1:], centers):
    m = (ages >= lo) & (ages < hi)
    if m.sum() >= 5:
        r = resid[m]; disp.append(r.var())
        o = np.argsort(ages[m]); rr = r[o]
        ac1.append(np.corrcoef(rr[:-1], rr[1:])[0, 1] if len(rr) > 2 else np.nan)
        pts = emb[m]; cov = np.cov(pts.T) + 1e-9 * np.eye(pts.shape[1])
        ent.append(0.5 * np.log(np.linalg.det(cov) + 1e-30)); oc.append(c)
disp = np.array(disp); ac1 = np.array(ac1); ent = np.array(ent); oc = np.array(oc)
BRS = disp / disp.max()

def peak_near(carr, sig, target, w=6):
    m = np.abs(carr - target) <= w
    if m.sum() == 0: return None
    return float(carr[m][np.argmax(sig[m])]), bool(np.median(sig[m]) > np.median(sig))
R["MV_peak_near_44"] = peak_near(vc, MV, 44)
R["MV_peak_near_60"] = peak_near(vc, MV, 60)
R["BRS_peak_near_44"] = peak_near(oc, BRS, 44)
R["BRS_peak_near_60"] = peak_near(oc, BRS, 60)
R["MV_max_age"] = round(float(vc[np.argmax(MV)]), 1)
R["BRS_max_age"] = round(float(oc[np.argmax(BRS)]), 1)

# ---- age prediction ----
idx = np.arange(len(ages)); tr, te = train_test_split(idx, test_size=0.3, random_state=0)
knn = KNeighborsRegressor(n_neighbors=20, metric="precomputed")
knn.fit(Dfix[np.ix_(tr, tr)], ages[tr]); pg = knn.predict(Dfix[np.ix_(te, tr)])
R["age_pred_manifold_knn_MAE"] = round(float(np.mean(np.abs(pg - ages[te]))), 3)
R["age_pred_manifold_knn_pearson"] = round(float(pearsonr(pg, ages[te])[0]), 4)
en = ElasticNetCV(cv=5, random_state=0, max_iter=5000)
en.fit(Xadj[tr], ages[tr]); pe = en.predict(Xadj[te])
R["age_pred_elasticnet_MAE"] = round(float(np.mean(np.abs(pe - ages[te]))), 3)
R["age_pred_elasticnet_pearson"] = round(float(pearsonr(pe, ages[te])[0]), 4)
R["TDI_mean"] = round(float(TDI.mean()), 4); R["TDI_std"] = round(float(TDI.std()), 4)

print(json.dumps(R, indent=2))
json.dump(R, open("phate_results.json", "w"), indent=2)

# ---- figure ----
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
sc = ax[0, 0].scatter(emb[:, 0], emb[:, 1], c=ages, cmap="viridis", s=14)
ax[0, 0].plot(gamma[:, 0], gamma[:, 1], "r-o", ms=3, lw=1.5, label="gamma_H")
ax[0, 0].set_title(f"PHATE manifold, cell-comp adjusted (GSE40279)\naxis1 vs age r={R['phate_axis1_age_pearson']}")
ax[0, 0].legend(); plt.colorbar(sc, ax=ax[0, 0], label="age")
ax[0, 1].plot(vc, MV, "-o", ms=3)
for t in (44, 60): ax[0, 1].axvline(t, color="r", ls="--", alpha=0.6)
ax[0, 1].set_title("Manifold Velocity MV(age)\nred = 44/60 transitions"); ax[0, 1].set_xlabel("age")
ax[0, 2].plot(oc, BRS, "-o", ms=3, label="BRS")
ax[0, 2].plot(oc, ac1, "-s", ms=3, label="lag-1 autocorr")
for t in (44, 60): ax[0, 2].axvline(t, color="r", ls="--", alpha=0.6)
ax[0, 2].set_title("BRS / early-warning signals"); ax[0, 2].set_xlabel("age"); ax[0, 2].legend()
ax[1, 0].plot(oc, ent, "-o", ms=3)
ax[1, 0].set_title("Trajectory Entropy proxy\n(log-det local covariance)"); ax[1, 0].set_xlabel("age")
ax[1, 1].scatter(ages[te], pg, s=14, alpha=0.6); ax[1, 1].plot([19, 101], [19, 101], "r--")
ax[1, 1].set_title(f"Manifold kNN age prediction\nMAE={R['age_pred_manifold_knn_MAE']}, r={R['age_pred_manifold_knn_pearson']}")
ax[1, 1].set_xlabel("true age"); ax[1, 1].set_ylabel("predicted")
ax[1, 2].hist(TDI, bins=30, color="steelblue")
ax[1, 2].set_title(f"Trajectory Deviation Index\nmean={R['TDI_mean']}, std={R['TDI_std']}")
ax[1, 2].set_xlabel("geodesic deviation from gamma_H")
plt.tight_layout(); plt.savefig("phate_figure.png", dpi=130)
print("saved phate_figure.png")
