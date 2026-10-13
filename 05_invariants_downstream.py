"""Downstream invariants from cached PHATE embedding + adjusted matrix."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from sklearn.neighbors import kneighbors_graph, KNeighborsRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import shortest_path
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("phate_results.json")) if os.path.exists("phate_results.json") else {}
d = np.load("methylgpt_input.npz", allow_pickle=True)
ages = d["ages"].astype(np.float64)
emb = np.load("phate_emb.npy")
Xadj = np.load("phate_Xadj.npy").astype(np.float64)
print("emb", emb.shape, "Xadj", Xadj.shape, flush=True)
if pearsonr(emb[:, 0], ages)[0] < 0:
    emb[:, 0] *= -1
R["n_samples"] = int(len(ages)); R["n_cpg"] = int(Xadj.shape[1])
R["phate_axis1_age_pearson"] = round(float(pearsonr(emb[:, 0], ages)[0]), 4)

# geodesics on PHATE embedding (low-dim -> fast, and diffusion-denoised)
kNN = 15
G = kneighbors_graph(emb, n_neighbors=kNN, mode="distance")
Dgeo = shortest_path(G, method="D", directed=False)
finite = np.isfinite(Dgeo)
R["geodesic_finite_fraction"] = round(float(finite.mean()), 4)
Dfix = Dgeo.copy(); Dfix[~finite] = np.nanmax(Dgeo[finite])
print("geodesics done", flush=True)

# gamma_H
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
print("invariants done", flush=True)

# age prediction: manifold kNN (fast). ElasticNet on top-2000 age-corr adjusted CpGs
idx = np.arange(len(ages)); tr, te = train_test_split(idx, test_size=0.3, random_state=0)
knn = KNeighborsRegressor(n_neighbors=20, metric="precomputed")
knn.fit(Dfix[np.ix_(tr, tr)], ages[tr]); pg = knn.predict(Dfix[np.ix_(te, tr)])
R["age_pred_manifold_knn_MAE"] = round(float(np.mean(np.abs(pg - ages[te]))), 3)
R["age_pred_manifold_knn_pearson"] = round(float(pearsonr(pg, ages[te])[0]), 4)

corr = np.array([pearsonr(Xadj[tr, j], ages[tr])[0] for j in range(Xadj.shape[1])])
topf = np.argsort(-np.abs(corr))[:2000]
en = ElasticNetCV(cv=5, random_state=0, max_iter=3000, n_jobs=2)
en.fit(Xadj[np.ix_(tr, topf)], ages[tr]); pe = en.predict(Xadj[np.ix_(te, topf)])
R["age_pred_elasticnet_MAE"] = round(float(np.mean(np.abs(pe - ages[te]))), 3)
R["age_pred_elasticnet_pearson"] = round(float(pearsonr(pe, ages[te])[0]), 4)
R["TDI_mean"] = round(float(TDI.mean()), 4); R["TDI_std"] = round(float(TDI.std()), 4)
print("age prediction done", flush=True)

print(json.dumps(R, indent=2))
json.dump(R, open("phate_results.json", "w"), indent=2)

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
