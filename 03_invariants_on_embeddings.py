"""
Geometric-topological aging invariants on MethylGPT foundation-model EMBEDDINGS
of GSE40279 (656 whole-blood samples, ages 19-101).

Input: 64-dim MethylGPT cell embeddings (methylgpt_embeddings.npz).
We repeat the SAME invariant pipeline as on raw beta and compare head to head.
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

R = {}
d = np.load("methylgpt_embeddings.npz", allow_pickle=True)
E, ages = d["E"].astype(np.float64), d["ages"].astype(np.float64)
print("loaded embeddings", E.shape, "ages", ages.min(), ages.max())

# standardize embedding dims
Xs = (E - E.mean(0)) / (E.std(0) + 1e-8)
R["n_samples"] = int(E.shape[0])
R["embedding_dim"] = int(E.shape[1])
# best single-dim age correlation (context)
cors = [abs(pearsonr(Xs[:, i], ages)[0]) for i in range(Xs.shape[1])]
R["max_abs_age_corr_dim"] = round(float(max(cors)), 4)

# ---- manifold: kNN geodesics + spectral embedding ----
kNN = 15
G = kneighbors_graph(Xs, n_neighbors=kNN, mode="distance")
Dgeo = shortest_path(G, method="D", directed=False)
finite = np.isfinite(Dgeo)
R["geodesic_finite_fraction"] = round(float(finite.mean()), 4)
Dfix = Dgeo.copy(); Dfix[~finite] = np.nanmax(Dgeo[finite])

emb = SpectralEmbedding(n_components=3, affinity="nearest_neighbors",
                        n_neighbors=kNN, random_state=0).fit_transform(Xs)
if pearsonr(emb[:, 0], ages)[0] < 0:
    emb[:, 0] *= -1
R["embedding_axis1_age_pearson"] = round(float(pearsonr(emb[:, 0], ages)[0]), 4)

# ---- reference trajectory gamma_H ----
bin_edges = np.arange(19, 102, 3.0)
centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
gamma, valid_centers = [], []
for lo, hi, c in zip(bin_edges[:-1], bin_edges[1:], centers):
    m = (ages >= lo) & (ages < hi)
    if m.sum() >= 3:
        gamma.append(emb[m].mean(0)); valid_centers.append(c)
gamma = np.array(gamma); valid_centers = np.array(valid_centers)

# ---- MV ----
dg = np.gradient(gamma, valid_centers, axis=0)
MV = norm(dg, axis=1)

# ---- BRS / EWS / TE ----
def gamma_at(age):
    return gamma[np.argmin(np.abs(valid_centers - age))]
resid = np.array([norm(emb[i] - gamma_at(ages[i])) for i in range(len(ages))])
TDI_all = resid.copy()
disp, ac1, ent, order_centers = [], [], [], []
for lo, hi, c in zip(bin_edges[:-1], bin_edges[1:], centers):
    m = (ages >= lo) & (ages < hi)
    if m.sum() >= 5:
        r = resid[m]; disp.append(r.var())
        o = np.argsort(ages[m]); rr = r[o]
        ac1.append(np.corrcoef(rr[:-1], rr[1:])[0, 1] if len(rr) > 2 else np.nan)
        pts = emb[m]; cov = np.cov(pts.T) + 1e-6 * np.eye(pts.shape[1])
        ent.append(0.5 * np.log(np.linalg.det(cov))); order_centers.append(c)
disp = np.array(disp); ac1 = np.array(ac1); ent = np.array(ent)
order_centers = np.array(order_centers)
BRS = disp / disp.max()

def peak_near(carr, sig, target, window=6):
    m = np.abs(carr - target) <= window
    if m.sum() == 0: return None
    return float(carr[m][np.argmax(sig[m])]), bool(np.median(sig[m]) > np.median(sig))

R["MV_peak_near_44"] = peak_near(valid_centers, MV, 44)
R["MV_peak_near_60"] = peak_near(valid_centers, MV, 60)
R["BRS_peak_near_44"] = peak_near(order_centers, BRS, 44)
R["BRS_peak_near_60"] = peak_near(order_centers, BRS, 60)
R["MV_max_age"] = round(float(valid_centers[np.argmax(MV)]), 1)
R["BRS_max_age"] = round(float(order_centers[np.argmax(BRS)]), 1)

# ---- age prediction ----
idx = np.arange(len(ages))
tr, te = train_test_split(idx, test_size=0.3, random_state=0)
knn = KNeighborsRegressor(n_neighbors=20, metric="precomputed")
knn.fit(Dfix[np.ix_(tr, tr)], ages[tr])
pred_geo = knn.predict(Dfix[np.ix_(te, tr)])
R["age_pred_manifold_knn_MAE"] = round(float(np.mean(np.abs(pred_geo - ages[te]))), 3)
R["age_pred_manifold_knn_pearson"] = round(float(pearsonr(pred_geo, ages[te])[0]), 4)

en = ElasticNetCV(cv=5, random_state=0, max_iter=5000)
en.fit(Xs[tr], ages[tr]); pred_en = en.predict(Xs[te])
R["age_pred_elasticnet_MAE"] = round(float(np.mean(np.abs(pred_en - ages[te]))), 3)
R["age_pred_elasticnet_pearson"] = round(float(pearsonr(pred_en, ages[te])[0]), 4)

R["TDI_mean"] = round(float(TDI_all.mean()), 4)
R["TDI_std"] = round(float(TDI_all.std()), 4)

print(json.dumps(R, indent=2))
json.dump(R, open("emb_results.json", "w"), indent=2)

# ---- figure ----
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
sc = ax[0, 0].scatter(emb[:, 0], emb[:, 1], c=ages, cmap="viridis", s=14)
ax[0, 0].plot(gamma[:, 0], gamma[:, 1], "r-o", ms=3, lw=1.5, label="gamma_H")
ax[0, 0].set_title(f"MethylGPT-embedding manifold (GSE40279)\naxis1 vs age r={R['embedding_axis1_age_pearson']}")
ax[0, 0].legend(); plt.colorbar(sc, ax=ax[0, 0], label="age")
ax[0, 1].plot(valid_centers, MV, "-o", ms=3)
for t in (44, 60): ax[0, 1].axvline(t, color="r", ls="--", alpha=0.6)
ax[0, 1].set_title("Manifold Velocity MV(age)\nred = 44/60 transitions"); ax[0, 1].set_xlabel("age")
ax[0, 2].plot(order_centers, BRS, "-o", ms=3, label="BRS")
ax[0, 2].plot(order_centers, ac1, "-s", ms=3, label="lag-1 autocorr")
for t in (44, 60): ax[0, 2].axvline(t, color="r", ls="--", alpha=0.6)
ax[0, 2].set_title("BRS / early-warning signals"); ax[0, 2].set_xlabel("age"); ax[0, 2].legend()
ax[1, 0].plot(order_centers, ent, "-o", ms=3)
ax[1, 0].set_title("Trajectory Entropy proxy\n(log-det local covariance)"); ax[1, 0].set_xlabel("age")
ax[1, 1].scatter(ages[te], pred_geo, s=14, alpha=0.6); ax[1, 1].plot([19, 101], [19, 101], "r--")
ax[1, 1].set_title(f"Manifold kNN age prediction\nMAE={R['age_pred_manifold_knn_MAE']}, r={R['age_pred_manifold_knn_pearson']}")
ax[1, 1].set_xlabel("true age"); ax[1, 1].set_ylabel("predicted")
ax[1, 2].hist(TDI_all, bins=30, color="steelblue")
ax[1, 2].set_title(f"Trajectory Deviation Index\nmean={R['TDI_mean']}, std={R['TDI_std']}")
ax[1, 2].set_xlabel("geodesic deviation from gamma_H")
plt.tight_layout(); plt.savefig("emb_figure.png", dpi=130)
print("saved emb_figure.png")
