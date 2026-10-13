"""
Publication-quality figures for the ASCA-2026 talk
"Beyond the Scalar Clock" (biologist/bioinformatician audience).

Inputs (produced by scripts 04-05, cached in ../gse40279/):
    phate_emb.npy            -- 656 x 3 PHATE embedding (cell-composition corrected)
    methylgpt_input.npz      -- ages (and vocab-aligned betas)
Outputs: figs/fig_concept.png, fig_manifold.png, fig_age_prediction.png,
         fig_transitions.png, fig_profiles.png
"""
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ---------- global style ----------
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.labelsize": 15, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    "legend.frameon": False,
})
CRIMSON = "#B22222"; TEAL = "#0E7C7B"; ORANGE = "#E07B39"; NAVY = "#1F3A5F"
OUT = "figs/"
DATA = "../gse40279/"

# ---------- load verified data ----------
emb = np.load(DATA + "phate_emb.npy")
ages = np.load(DATA + "methylgpt_input.npz", allow_pickle=True)["ages"].astype(float)
if pearsonr(emb[:, 0], ages)[0] < 0: emb[:, 0] *= -1
n = len(ages)
print("loaded", emb.shape, "ages", ages.min(), ages.max())

# reference trajectory gamma_H: 3-yr bin means, lightly smoothed
edges = np.arange(19, 102, 3.0)
cent = 0.5 * (edges[:-1] + edges[1:])
g_raw, vc = [], []
for lo, hi, c in zip(edges[:-1], edges[1:], cent):
    m = (ages >= lo) & (ages < hi)
    if m.sum() >= 3:
        g_raw.append(emb[m].mean(0)); vc.append(c)
g_raw = np.array(g_raw); vc = np.array(vc)
def smooth(y, w=3):
    k = np.ones(w) / w
    pad = np.r_[y[:w//2][::-1], y, y[-(w//2):][::-1]]
    return np.convolve(pad, k, "valid")[:len(y)]
gam = np.column_stack([smooth(g_raw[:, j]) for j in range(g_raw.shape[1])])

# geodesics + LOO prediction
G = kneighbors_graph(emb, n_neighbors=15, mode="distance")
Dg = shortest_path(G, method="D", directed=False)
fin = np.isfinite(Dg); Dg[~fin] = np.nanmax(Dg[fin])
preds = np.empty(n)
for i in range(n):
    drow = Dg[i].copy(); drow[i] = np.inf
    idx = np.argsort(drow)[:20]
    w = 1.0 / (drow[idx] + 1e-12)
    preds[i] = np.sum(w * ages[idx]) / np.sum(w)
MAE = np.mean(np.abs(preds - ages)); R = pearsonr(preds, ages)[0]
print(f"LOO geodesic kNN: MAE={MAE:.2f}, r={R:.3f}")

# per-sample TDI; binned MV / variance / autocorrelation
def gam_at(a): return gam[np.argmin(np.abs(vc - a))]
TDI = np.array([norm(emb[i] - gam_at(ages[i])) for i in range(n)])
dg = np.gradient(gam, vc, axis=0); MV = norm(dg, axis=1)
var_b, ac_b, oc = [], [], []
for lo, hi, c in zip(edges[:-1], edges[1:], cent):
    m = (ages >= lo) & (ages < hi)
    if m.sum() >= 5:
        r_ = TDI[m]; var_b.append(r_.var())
        o = np.argsort(ages[m]); rr = r_[o]
        ac_b.append(np.corrcoef(rr[:-1], rr[1:])[0, 1] if len(rr) > 2 else np.nan)
        oc.append(c)
var_b = np.array(var_b); ac_b = np.array(ac_b); oc = np.array(oc)
BRS_b = var_b / var_b.max()

# ============ FIG 1: concept (clock vs map) ============
rng = np.random.default_rng(3)
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
t = rng.random(300)
cx = t * 60 + 20 + rng.normal(0, 2.5, 300)
cy = 0.9 * np.sin((t * 60 + 20) / 14) + rng.normal(0, 0.33, 300)
ax = axes[0]; ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.scatter(cx, cy, c=t, cmap="viridis", s=18, alpha=0.75)
iA = np.argmin(np.abs(cx - 52) + np.abs(cy - 1.05)); iB = np.argmin(np.abs(cx - 52) + np.abs(cy + 1.15))
for i_, col, lab in [(iA, TEAL, "Person A"), (iB, ORANGE, "Person B")]:
    ax.scatter([cx[i_]], [cy[i_]], s=330, marker="*", color=col, edgecolor="k", zorder=6)
    ax.annotate(lab, (cx[i_], cy[i_]), xytext=(cx[i_] - 17, cy[i_] + 0.33), fontsize=14, fontweight="bold", color=col)
ax.annotate("", xy=(84, -2.15), xytext=(16, -2.15), arrowprops=dict(arrowstyle="-|>", lw=2.4, color=CRIMSON))
ax.text(50, -2.5, "epigenetic clock output: one number", ha="center", fontsize=13.5, color=CRIMSON)
for i_ in (iA, iB):
    ax.plot([cx[i_], 52], [cy[i_], -2.05], ls=":", lw=1.6, color="grey")
ax.scatter([52], [-2.15], s=120, marker="v", color=CRIMSON, zorder=6)
ax.text(52, -1.92, 'both read "52 years"', ha="center", fontsize=12.5, style="italic")
ax.set_title("A scalar clock cannot tell A and B apart")
ax.set_xlim(10, 92); ax.set_ylim(-2.75, 1.95)

ax = axes[1]; ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.scatter(cx, cy, c=t, cmap="viridis", s=18, alpha=0.75)
xs = np.linspace(20, 80, 100)
ax.plot(xs, 0.9 * np.sin(xs / 14), color=CRIMSON, lw=3.2, label="healthy-aging route (geodesic)")
bx = np.linspace(55, 80, 40)
ax.plot(bx, 0.9 * np.sin(55 / 14) - (bx - 55) * 0.055, color="#B8860B", lw=2.6, ls="--", label="alternative branch")
ax.scatter([55], [0.9 * np.sin(55 / 14)], marker="X", s=240, color="k", zorder=6, label="crossroads (tipping point)")
ax.scatter([cx[iA]], [cy[iA]], s=330, marker="*", color=TEAL, edgecolor="k", zorder=6)
ax.annotate("A: on route", (cx[iA], cy[iA]), xytext=(cx[iA] + 9, cy[iA] + 0.62), fontsize=14, fontweight="bold", color=TEAL,
            arrowprops=dict(arrowstyle="->", lw=1.6, color=TEAL))
ax.scatter([cx[iB]], [cy[iB]], s=330, marker="*", color=ORANGE, edgecolor="k", zorder=6)
ax.annotate("B: far off route", (cx[iB], cy[iB]), xytext=(cx[iB] - 32, cy[iB] - 0.62), fontsize=14, fontweight="bold", color=ORANGE)
ax.legend(loc="lower left", fontsize=11.5)
ax.set_title("The map view: same age, different position")
ax.set_xlim(10, 92); ax.set_ylim(-2.75, 1.95)
plt.tight_layout(); plt.savefig(OUT + "fig_concept.png", dpi=190, bbox_inches="tight"); plt.close()

# ============ FIG 2: manifold ============
fig, ax = plt.subplots(figsize=(8.4, 6.4))
sc = ax.scatter(emb[:, 0], emb[:, 1], c=ages, cmap="viridis", s=26, alpha=0.85, edgecolor="none")
ax.plot(gam[:, 0], gam[:, 1], color=CRIMSON, lw=3.4, zorder=5, label="reference trajectory $\\gamma_H$ (healthy aging)")
ar = FancyArrowPatch((gam[-3, 0], gam[-3, 1]), (gam[-1, 0], gam[-1, 1]),
                     arrowstyle="-|>", mutation_scale=28, color=CRIMSON, lw=0, zorder=6)
ax.add_patch(ar)
y_m = ages <= 30; o_m = ages >= 85
ax.annotate("youngest\n(19\u201330 y)", emb[y_m].mean(0)[:2], xytext=(-0.085, -0.035),
            fontsize=13, fontweight="bold", color="#3b528b",
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#3b528b"))
ax.annotate("oldest\n(85+ y)", emb[o_m].mean(0)[:2], xytext=(0.058, -0.008),
            fontsize=13, fontweight="bold", color="#2e8b57",
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#2e8b57"))
cb = plt.colorbar(sc, ax=ax); cb.set_label("chronological age (years)", fontsize=14)
ax.set_xlabel("PHATE 1"); ax.set_ylabel("PHATE 2")
ax.set_title("The DNA-methylation aging map\nGSE40279 whole blood, n = 656, built without age labels")
ax.legend(loc="lower right", fontsize=12)
txt = f"map axis vs age:  r = {pearsonr(emb[:,0], ages)[0]:.2f}"
ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=13,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F2F4F7", ec=NAVY))
plt.tight_layout(); plt.savefig(OUT + "fig_manifold.png", dpi=190, bbox_inches="tight"); plt.close()

# ============ FIG 3: age prediction ============
fig, ax = plt.subplots(figsize=(7.4, 6.4))
ax.scatter(ages, preds, s=26, alpha=0.65, color=NAVY, edgecolor="none")
lims = [15, 105]
ax.plot(lims, lims, ls="--", lw=2, color=CRIMSON, label="perfect prediction (y = x)")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("chronological age (years)"); ax.set_ylabel("age read off the map (years)")
ax.set_title("Geometry alone recovers age\ngeodesic k-nearest-neighbour, leave-one-out")
ax.text(0.03, 0.97, f"MAE = {MAE:.1f} years\nr = {R:.2f}\nno clock was trained",
        transform=ax.transAxes, va="top", fontsize=14.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#F2F4F7", ec=NAVY))
ax.legend(loc="lower right", fontsize=12.5)
plt.tight_layout(); plt.savefig(OUT + "fig_age_prediction.png", dpi=190, bbox_inches="tight"); plt.close()

# ============ FIG 4: transitions ============
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))
for ax in axes:
    ax.axvspan(41, 47, color=CRIMSON, alpha=0.10)
    ax.axvspan(57, 63, color=CRIMSON, alpha=0.10)
ax = axes[0]
ax.plot(vc, MV, "o", color="#9aa7b5", ms=5, alpha=0.8)
ax.plot(vc, smooth(MV), color=NAVY, lw=3)
ax.set_xlabel("age (years)"); ax.set_ylabel("Manifold Velocity (a.u.)")
ax.set_title("Pace of movement along the map")
ax.text(44, ax.get_ylim()[1]*0.97, "mid-40s\nwave", ha="center", va="top", fontsize=12.5, color=CRIMSON, fontweight="bold")
ax.text(60, ax.get_ylim()[1]*0.97, "~60\nwave", ha="center", va="top", fontsize=12.5, color=CRIMSON, fontweight="bold")
ax = axes[1]
ax.plot(oc, BRS_b, "o", color="#9aa7b5", ms=5, alpha=0.8)
ax.plot(oc, smooth(BRS_b), color=NAVY, lw=3, label="Bifurcation Risk Score (binned)")
ax2 = ax.twinx(); ax2.grid(False); ax2.spines["top"].set_visible(False)
ax2.plot(oc, smooth(np.nan_to_num(ac_b)), color=ORANGE, lw=2.4, ls="--", label="lag-1 autocorrelation")
ax2.set_ylabel("lag-1 autocorrelation", color=ORANGE); ax2.tick_params(axis="y", colors=ORANGE)
ax.set_xlabel("age (years)"); ax.set_ylabel("BRS (normalised dispersion)")
ax.set_title("Early-warning signals around transitions")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=11.5)
fig.suptitle("Shaded bands: molecular \u201cwaves of aging\u201d reported near 44 and 60 (Shen et al., 2024)",
             fontsize=13.5, y=1.02, style="italic")
plt.tight_layout(); plt.savefig(OUT + "fig_transitions.png", dpi=190, bbox_inches="tight"); plt.close()

# ============ FIG 5: same age, different aging ============
m60 = np.abs(ages - 60) <= 2
cand = np.where(m60)[0]
lo_i = cand[np.argmin(TDI[cand])]; hi_i = cand[np.argmax(TDI[cand])]
def pct(v, arr): return 100.0 * (arr < v).mean()
labels = ["Off-route\ndeviation (TDI)", "Pace of aging\n(MV)", "Tipping-point\nrisk (BRS)", "Path instability\n(TE)"]
mv_s = MV[np.argmin(np.abs(vc[:, None] - ages[None, :]), axis=0)]
brs_s = BRS_b[np.argmin(np.abs(oc[:, None] - ages[None, :]), axis=0)]
# local TE proxy: distance to 10th neighbour in embedding (local sparseness)
d10 = np.sort(Dg, axis=1)[:, 10]
A_vals = [pct(TDI[lo_i], TDI), pct(mv_s[lo_i], mv_s), pct(brs_s[lo_i], brs_s), pct(d10[lo_i], d10)]
B_vals = [pct(TDI[hi_i], TDI), pct(mv_s[hi_i], mv_s), pct(brs_s[hi_i], brs_s), pct(d10[hi_i], d10)]
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), gridspec_kw={"width_ratios": [1.05, 1.3]})
ax = axes[0]
ax.scatter(emb[:, 0], emb[:, 1], c=ages, cmap="viridis", s=18, alpha=0.5, edgecolor="none")
ax.plot(gam[:, 0], gam[:, 1], color=CRIMSON, lw=2.6)
ax.scatter([emb[lo_i, 0]], [emb[lo_i, 1]], s=430, marker="*", color=TEAL, edgecolor="k", zorder=6)
ax.scatter([emb[hi_i, 0]], [emb[hi_i, 1]], s=430, marker="*", color=ORANGE, edgecolor="k", zorder=6)
ax.annotate(f"A, {ages[lo_i]:.0f} y\non the route", emb[lo_i, :2], xytext=(-0.088, 0.0155),
            fontsize=13, fontweight="bold", color=TEAL, arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.6))
ax.annotate(f"B, {ages[hi_i]:.0f} y\nfar off route", emb[hi_i, :2], xytext=(-0.012, -0.0415),
            fontsize=13, fontweight="bold", color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
ax.set_xlabel("PHATE 1"); ax.set_ylabel("PHATE 2")
ax.set_title("Two individuals, both \u2248 60 years old")
ax = axes[1]
x = np.arange(4); w = 0.38
ax.bar(x - w/2, A_vals, w, color=TEAL, label=f"Person A ({ages[lo_i]:.0f} y)")
ax.bar(x + w/2, B_vals, w, color=ORANGE, label=f"Person B ({ages[hi_i]:.0f} y)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12.5)
ax.tick_params(axis="x", pad=6)
ax.set_ylabel("cohort percentile (%)"); ax.set_ylim(0, 108)
ax.axhline(50, ls=":", color="grey", lw=1.5); ax.text(3.52, 52, "cohort median", fontsize=11, color="grey")
ax.set_title("Their geometric aging profiles differ sharply")
ax.legend(fontsize=12.5, loc="upper left")
plt.tight_layout(); plt.savefig(OUT + "fig_profiles.png", dpi=190, bbox_inches="tight"); plt.close()

print("figures saved:", MAE, R)
