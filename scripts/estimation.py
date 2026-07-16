"""
Causal estimation: effect of sustained prenatal smoking on infant mortality.

Estimators
----------
1. S-Learner  (logistic regression, T as feature)       → ATE
2. T-Learner  (separate outcome model per arm)          → ATE
3. IPW        (stabilised Hajek weights)                → ATE
4. AIPW / DR  (doubly-robust, primary estimator)        → ATE
5. PS Matching (1-to-1 NN on logit-PS, caliper 0.2 SD) → ATT

Standard errors use the efficient influence function (estimators 1–4)
and matched-pair differences (estimator 5).

Outputs
-------
results/estimates.csv          — point estimates, SEs, 95% CIs, match stats
results/estimates_forest.png   — forest plot
results/ipw_diagnostics.txt    — IPW ESS and max weight
results/scores.parquet         — propensity scores + mu0/mu1 for sensitivity script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import os, warnings
warnings.filterwarnings("ignore")

ROOT    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(ROOT, "data", "analytic_extract.csv")
OUT_DIR      = os.path.join(ROOT, "results")
OUT_DIR_IMGS = os.path.join(ROOT, "results", "imgs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR_IMGS, exist_ok=True)

# ── 1. Load and build analytic sample ────────────────────────────────────────

print("Loading data...")
df = pd.read_csv(DATA, dtype=str)
df = df[df["F_TOBACCO"] == "1"].copy()

num_cols = ["CIG0_R","CIG1_R","CIG2_R","CIG3_R",
            "MAGER","MRACEHISP","DMAR","MEDUC","PAY_REC",
            "BMI_R","FEDUC","FAGE11","PRIORLIVE","F_M_HT",
            "F_RF_PDIAB","F_RF_PHYPE","F_PAY_REC"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["DIED"] = pd.to_numeric(df["DIED"], errors="coerce")

# Treatment groups (recodes 1-5 are dose categories; 6 = unknown, excluded)
treated = (df["CIG0_R"].between(1, 5) & df["CIG1_R"].between(1, 5) &
           df["CIG2_R"].between(1, 5) & df["CIG3_R"].between(1, 5))
control  = (df["CIG0_R"] == 0) & (df["CIG_REC"] == "N")
df = df[treated | control].copy()
df["T"] = treated[df.index].astype(int)

# Confounder completeness filters
df = df[df["MEDUC"].between(1, 8)].copy()
df = df[df["PAY_REC"].between(1, 4)].copy()
df = df[df["MRACEHISP"].between(1, 9)].copy()
df = df[df["BMI_R"].between(1, 6)].copy()        # 6-cat recode; 9=unknown excluded
df = df[df["FEDUC"].between(1, 8)].copy()
df = df[df["FAGE11"].between(1, 11)].copy()
df = df[(df["F_RF_PDIAB"] == 1) & (df["F_RF_PHYPE"] == 1)].copy()
# Parity: exclude unknown code (99); cap at 5 (= "5 or more prior live births")
df = df[df["PRIORLIVE"] != 99].copy()
df["PRIORLIVE"] = df["PRIORLIVE"].clip(upper=5)
df = df[df["MAGER"].notna() & df["DIED"].notna()].copy()
df = df.reset_index(drop=True)

# Binary confounders
df["DM"]    = (df["RF_PDIAB"] == "Y").astype(int)
df["HTN"]   = (df["RF_PHYPE"] == "Y").astype(int)
df["WIC_Y"] = (df["WIC"]      == "Y").astype(int)

# Design matrix
cat_cols = ["MRACEHISP","DMAR","MEDUC","PAY_REC","BMI_R","FEDUC","FAGE11"]
X_df = pd.get_dummies(df[cat_cols].astype(str), drop_first=True)
X_df["MAGER"]     = df["MAGER"].values
X_df["WIC"]       = df["WIC_Y"].values
X_df["DM"]        = df["DM"].values
X_df["HTN"]       = df["HTN"].values
X_df["PRIORLIVE"] = df["PRIORLIVE"].values
X_cols = list(X_df.columns)

T = df["T"].values
Y = df["DIED"].values
n = len(T)

print(f"Analytic sample: {n:,}  (T=1: {T.sum():,}, T=0: {(T==0).sum():,})")
print(f"Outcome rate:  treated={Y[T==1].mean()*1000:.3f}/1k  "
      f"control={Y[T==0].mean()*1000:.3f}/1k  "
      f"crude diff={( Y[T==1].mean()-Y[T==0].mean() )*1000:+.3f}/1k")

# ── 2. Propensity score model ─────────────────────────────────────────────────

print("\nFitting propensity score model...")
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_df.values.astype(float))

ps_model = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
ps_model.fit(X_sc, T)
e        = ps_model.predict_proba(X_sc)[:, 1]
logit_e  = np.log(e / (1 - e))
e_clip   = np.clip(e, 1e-4, 1 - 1e-4)

print(f"  PS range treated: [{e[T==1].min():.4f}, {e[T==1].max():.4f}]  "
      f"control: [{e[T==0].min():.4f}, {e[T==0].max():.4f}]")

# ── 3. Outcome models ─────────────────────────────────────────────────────────

print("Fitting outcome models...")

t1_model = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
t0_model = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
t1_model.fit(X_sc[T == 1], Y[T == 1])
t0_model.fit(X_sc[T == 0], Y[T == 0])
mu1 = t1_model.predict_proba(X_sc)[:, 1]
mu0 = t0_model.predict_proba(X_sc)[:, 1]

XT_sc    = np.column_stack([X_sc, T])
s_model  = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
s_model.fit(XT_sc, Y)
mu1_s = s_model.predict_proba(np.column_stack([X_sc, np.ones(n)]))[:, 1]
mu0_s = s_model.predict_proba(np.column_stack([X_sc, np.zeros(n)]))[:, 1]

# ── 4. Estimators ─────────────────────────────────────────────────────────────

def eif_se(psi):
    """SE from centred efficient influence function scores."""
    return (psi - psi.mean()).std(ddof=1) / np.sqrt(len(psi))

# S-Learner
psi_s   = mu1_s - mu0_s
ate_s   = psi_s.mean()
se_s    = eif_se(psi_s)

# T-Learner
psi_t   = mu1 - mu0
ate_t   = psi_t.mean()
se_t    = eif_se(psi_t)

# IPW (stabilised Hajek)
p1 = T.mean()
p0 = 1 - p1
w_raw1 = np.where(T == 1, p1 / e_clip, 0.0)
w_raw0 = np.where(T == 0, p0 / (1 - e_clip), 0.0)
num1 = np.where(T == 1, Y / e_clip,       0.0)
num0 = np.where(T == 0, Y / (1 - e_clip), 0.0)
den1 = np.where(T == 1, 1 / e_clip,       0.0)
den0 = np.where(T == 0, 1 / (1 - e_clip), 0.0)
ipw1    = num1.sum() / den1.sum()
ipw0    = num0.sum() / den0.sum()
ate_ipw = ipw1 - ipw0
psi_ipw = (T * (Y - ipw1) / e_clip) - ((1 - T) * (Y - ipw0) / (1 - e_clip))
se_ipw  = eif_se(psi_ipw)

# IPW diagnostics: effective sample size and max stabilised weight
w_all = w_raw1 + w_raw0   # one non-zero per unit
ess_t  = w_raw1[T==1].sum()**2 / (w_raw1[T==1]**2).sum()
ess_c  = w_raw0[T==0].sum()**2 / (w_raw0[T==0]**2).sum()
max_w  = w_all.max()
print(f"\n  IPW diagnostics:")
print(f"    Effective sample size (treated): {ess_t:,.0f} / {T.sum():,} ({ess_t/T.sum()*100:.1f}%)")
print(f"    Effective sample size (control): {ess_c:,.0f} / {(T==0).sum():,} ({ess_c/(T==0).sum()*100:.1f}%)")
print(f"    Max stabilised weight: {max_w:.2f}")
ipw_diag_lines = [
    "=" * 55,
    "IPW WEIGHT DIAGNOSTICS",
    "=" * 55,
    f"Effective sample size (treated): {ess_t:,.0f} / {T.sum():,}  ({ess_t/T.sum()*100:.1f}%)",
    f"Effective sample size (control): {ess_c:,.0f} / {(T==0).sum():,}  ({ess_c/(T==0).sum()*100:.1f}%)",
    f"Max stabilised weight:           {max_w:.2f}",
    "=" * 55,
]
with open(os.path.join(OUT_DIR, "ipw_diagnostics.txt"), "w") as f:
    f.write("\n".join(ipw_diag_lines))

# AIPW (doubly-robust, primary)
psi_dr   = (mu1 - mu0
            + T       * (Y - mu1) / e_clip
            - (1 - T) * (Y - mu0) / (1 - e_clip))
ate_aipw = psi_dr.mean()
se_aipw  = eif_se(psi_dr)

# ── 5. PS Matching (ATT) ─────────────────────────────────────────────────────

print("Matching treated to controls on logit-PS (1:1 NN, caliper 0.2 SD)...")
caliper = 0.2 * np.std(logit_e)
idx_t   = np.where(T == 1)[0]
idx_c   = np.where(T == 0)[0]

tree         = cKDTree(logit_e[idx_c].reshape(-1, 1))
dists, match = tree.query(logit_e[idx_t].reshape(-1, 1), k=1)
within       = dists.ravel() <= caliper
idx_t_m      = idx_t[within]
idx_c_m      = idx_c[match.ravel()[within]]
diff         = Y[idx_t_m].astype(float) - Y[idx_c_m].astype(float)
att_m        = diff.mean()
se_m         = diff.std(ddof=1) / np.sqrt(len(diff))
n_matched    = len(diff)
pct_matched  = n_matched / len(idx_t) * 100
print(f"  Caliper={caliper:.4f}  matched {n_matched:,}/{len(idx_t):,} "
      f"treated ({pct_matched:.1f}%)")

# ── 6. Results table ──────────────────────────────────────────────────────────

Z95 = 1.96
rows = [
    ("S-Learner",       ate_s,    se_s,    "ATE"),
    ("T-Learner",       ate_t,    se_t,    "ATE"),
    ("IPW (Hajek)",     ate_ipw,  se_ipw,  "ATE"),
    ("AIPW / DR",       ate_aipw, se_aipw, "ATE"),
    ("PS Matching",     att_m,    se_m,    "ATT"),
]
res = pd.DataFrame(rows, columns=["Estimator","Estimate","SE","Target"])
res["CI_lo"] = res["Estimate"] - Z95 * res["SE"]
res["CI_hi"] = res["Estimate"] + Z95 * res["SE"]
for col in ["Estimate","CI_lo","CI_hi","SE"]:
    res[f"{col}_1k"] = res[col] * 1000

# Append matching metadata
res["n_matched"]   = [np.nan]*4 + [n_matched]
res["pct_matched"] = [np.nan]*4 + [pct_matched]

print("\n" + "="*72)
print("CAUSAL ESTIMATES  (additional infant deaths per 1,000 live births)")
print("="*72)
for _, r in res.iterrows():
    sig = "" if (r["CI_lo_1k"] < 0 < r["CI_hi_1k"]) else " *"
    print(f"  {r['Estimator']:<18} ({r['Target']})  "
          f"{r['Estimate_1k']:+.4f}  "
          f"95% CI [{r['CI_lo_1k']:+.4f}, {r['CI_hi_1k']:+.4f}]{sig}")
print("  (* CI excludes zero)")
print("="*72)

res.to_csv(os.path.join(OUT_DIR, "estimates.csv"), index=False, float_format="%.6f")

# ── 7. Forest plot ────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
palette = ["#5B8DB8", "#5B8DB8", "#5B8DB8", "#C00000", "#ED7D31"]
ypos    = np.arange(len(res))[::-1]

for y, (_, r), col in zip(ypos, res.iterrows(), palette):
    ax.plot([r["CI_lo_1k"], r["CI_hi_1k"]], [y, y],
            color=col, lw=2.5, solid_capstyle="round")
    ax.plot(r["Estimate_1k"], y, "o", color=col, ms=7, zorder=5)

ax.axvline(0, color="black", lw=1, ls="--", alpha=0.6, label="Null")
ax.set_yticks(ypos)
ax.set_yticklabels(
    [f"{r['Estimator']}  ({r['Target']})" for _, r in res.iterrows()],
    fontsize=9)
ax.set_xlabel("Additional infant deaths per 1,000 live births  (95% CI)", fontsize=9)
ax.set_title(
    "Effect of Sustained Prenatal Smoking on Infant Mortality\n"
    "Primary estimator: AIPW/DR (red) | ATT: PS Matching (orange)",
    fontsize=10)
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR_IMGS, "estimates_forest.png"), dpi=150)
plt.close()
print(f"\nSaved → results/estimates.csv")
print(f"Saved → results/estimates_forest.png")
print(f"Saved → results/ipw_diagnostics.txt")

# ── 8. Save scores for sensitivity analysis ───────────────────────────────────

scores_df = pd.DataFrame({
    "T":        T,
    "Y":        Y,
    "ps":       e,
    "logit_ps": logit_e,
    "mu1":      mu1,
    "mu0":      mu0,
    "psi_dr":   psi_dr,
    "CIG1_R":   df["CIG1_R"].values,
    "CIG2_R":   df["CIG2_R"].values,
    "CIG3_R":   df["CIG3_R"].values,
    "CIG0_R":   df["CIG0_R"].values,
})
scores_df.to_parquet(os.path.join(OUT_DIR, "scores.parquet"), index=False)
print(f"Saved → results/scores.parquet  (for sensitivity.py)")
