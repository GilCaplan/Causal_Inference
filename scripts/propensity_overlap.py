"""
Propensity score modelling, overlap (positivity) diagnostics, and
covariate balance assessment.

Outputs (saved to results/):
  ps_overlap_histogram.png         -- mirrored histogram of PS by treatment arm
  ps_overlap_logit.png             -- logit-PS density plot
  covariate_balance_love.png       -- Love plot (|SMD| before/after IPW)
  covariate_balance_smd.csv        -- SMD table (raw and IPW-weighted)
  overlap_summary.txt              -- numeric overlap statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os, warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(ROOT, "data", "analytic_extract.csv")
OUT_DIR      = os.path.join(ROOT, "results")
OUT_DIR_IMGS = os.path.join(ROOT, "results", "imgs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR_IMGS, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA, dtype=str)

# ── Analytic sample filters ──────────────────────────────────────────────────
df = df[df["F_TOBACCO"] == "1"].copy()

num_cols = ["CIG0_R","CIG1_R","CIG2_R","CIG3_R","MAGER","MRACEHISP",
            "DMAR","MEDUC","PAY_REC","BMI_R","FEDUC","FAGE11","PRIORLIVE",
            "F_M_HT"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["DIED"] = pd.to_numeric(df["DIED"], errors="coerce")

# Treatment definition
treated  = (df["CIG0_R"] >= 1) & (df["CIG1_R"] >= 1) & \
           (df["CIG2_R"] >= 1) & (df["CIG3_R"] >= 1)
control  = (df["CIG0_R"] == 0) & (df["CIG_REC"] == "N")

df = df[treated | control].copy()
df["T"] = treated[df.index].astype(int)
print(f"  Treated (T=1): {df['T'].sum():,}   Control (T=0): {(df['T']==0).sum():,}")

# Binary confounder flags
df["RF_PDIAB_str"] = df["RF_PDIAB"].astype(str)
df["RF_PHYPE_str"]  = df["RF_PHYPE"].astype(str)
df["WIC_str"]       = df["WIC"].astype(str)

df["F_RF_PDIAB"] = pd.to_numeric(df["F_RF_PDIAB"], errors="coerce")
df["F_RF_PHYPE"]  = pd.to_numeric(df["F_RF_PHYPE"], errors="coerce")

# Confounder completeness / validity filters
df = df[df["MEDUC"].between(1, 8)].copy()
df = df[df["PAY_REC"].between(1, 4)].copy()
df = df[df["MRACEHISP"].between(1, 9)].copy()
df = df[df["BMI_R"].between(1, 6)].copy()        # 6-cat recode; 9=unknown excluded
df = df[df["FEDUC"].between(1, 8)].copy()
df = df[df["FAGE11"].between(1, 11)].copy()
df = df[(df["F_RF_PDIAB"] == 1) & (df["F_RF_PHYPE"] == 1)].copy()
# Parity: exclude unknown code (99); cap at 5 (5 = "5 or more prior live births")
df = df[df["PRIORLIVE"] != 99].copy()
df["PRIORLIVE"] = df["PRIORLIVE"].clip(upper=5)

df["DM"]    = (df["RF_PDIAB_str"].loc[df.index] == "Y").astype(int)
df["HTN"]   = (df["RF_PHYPE_str"].loc[df.index]  == "Y").astype(int)
df["WIC_Y"] = (df["WIC_str"].loc[df.index]        == "Y").astype(int)

cat_cols = ["MRACEHISP","DMAR","MEDUC","PAY_REC","BMI_R","FEDUC","FAGE11"]
X_df = pd.get_dummies(df[cat_cols].astype(str), drop_first=True)
X_df["MAGER"]     = df["MAGER"].values
X_df["WIC"]       = df["WIC_Y"].values
X_df["DM"]        = df["DM"].values
X_df["HTN"]       = df["HTN"].values
X_df["PRIORLIVE"] = df["PRIORLIVE"].values

T = df["T"].values
print(f"  Analytic sample after exclusions: {len(T):,}  "
      f"(T=1: {T.sum():,}, T=0: {(T==0).sum():,})")

# ── Propensity score model ───────────────────────────────────────────────────
print("Fitting propensity score model (logistic regression)...")
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_df.values.astype(float))

lr = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
lr.fit(X_sc, T)
ps = lr.predict_proba(X_sc)[:, 1]

df["ps"]       = ps
df["logit_ps"] = np.log(ps / (1 - ps))
print(f"  PS range overall: [{ps.min():.4f}, {ps.max():.4f}]")
print(f"  PS mean treated={ps[T==1].mean():.4f}  control={ps[T==0].mean():.4f}")

# ── Overlap / positivity diagnostics ─────────────────────────────────────────
ps1 = ps[T == 1]
ps0 = ps[T == 0]

lo_cs = max(ps1.min(), ps0.min())
hi_cs = min(ps1.max(), ps0.max())

p5,  p95  = np.percentile(ps1, [5, 95])
p1,  p99  = np.percentile(ps1, [1, 99])
n_trim_595  = ((ps < p5) | (ps > p95)).sum()
n_trim_199  = ((ps < p1) | (ps > p99)).sum()

pct_treated_below_median_control = (ps1 < np.median(ps0)).mean() * 100

lines = [
    "=" * 60,
    "PROPENSITY SCORE OVERLAP DIAGNOSTICS",
    "=" * 60,
    f"Analytic sample (after all filters): {len(T):,}",
    f"  Treated (T=1):  {T.sum():,}  ({T.mean()*100:.2f}%)",
    f"  Control (T=0):  {(T==0).sum():,}  ({(1-T).mean()*100:.2f}%)",
    "",
    "PS distribution:",
    f"  Treated -- mean={ps1.mean():.4f}  sd={ps1.std():.4f}  "
    f"min={ps1.min():.4f}  max={ps1.max():.4f}",
    f"  Control -- mean={ps0.mean():.4f}  sd={ps0.std():.4f}  "
    f"min={ps0.min():.4f}  max={ps0.max():.4f}",
    "",
    "Common support region (max of mins to min of maxs):",
    f"  [{lo_cs:.4f}, {hi_cs:.4f}]",
    "",
    "Trimming impact (units outside treated PS range):",
    f"  1st/99th percentile trim: {n_trim_199:,} units removed ({n_trim_199/len(T)*100:.2f}%)",
    f"  5th/95th percentile trim: {n_trim_595:,} units removed ({n_trim_595/len(T)*100:.2f}%)",
    "",
    f"% of treated with PS below median PS of controls: {pct_treated_below_median_control:.1f}%",
    "  (should be low; high value = poor overlap)",
    "=" * 60,
]
summary = "\n".join(lines)
print(summary)
with open(os.path.join(OUT_DIR, "overlap_summary.txt"), "w") as f:
    f.write(summary)

# ── Plot 1: Mirrored histogram of PS ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.8))
bins = np.linspace(0, 1, 51)

counts1, _ = np.histogram(ps1, bins=bins)
counts0, _ = np.histogram(ps0, bins=bins)
mids = (bins[:-1] + bins[1:]) / 2
w    = bins[1] - bins[0]

ax.bar(mids,  counts1 / counts1.sum(), width=w * 0.9,
       color="#4472C4", alpha=0.8, label="Treated (T=1)")
ax.bar(mids, -counts0 / counts0.sum(), width=w * 0.9,
       color="#ED7D31", alpha=0.8, label="Control (T=0)")

ax.axvspan(lo_cs, hi_cs, alpha=0.07, color="green",
           label=f"Common support [{lo_cs:.3f}, {hi_cs:.3f}]")
ax.axvline(p5,  color="#4472C4", ls="--", lw=1.2, alpha=0.8)
ax.axvline(p95, color="#4472C4", ls="--", lw=1.2, alpha=0.8,
           label=f"5th/95th pctile trim ({p5:.3f}, {p95:.3f})")

ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Estimated Propensity Score  P(T=1 | X)", fontsize=10)
ax.set_ylabel("Proportion", fontsize=10)
ax.set_title("Overlap Check: PS Distribution by Treatment Arm", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{abs(x):.2f}"))
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR_IMGS, "ps_overlap_histogram.png"), dpi=150)
print(f"Saved: {OUT_DIR_IMGS}/ps_overlap_histogram.png")
plt.close()

# ── Plot 2: Logit-PS density ─────────────────────────────────────────────────
from scipy.stats import gaussian_kde

fig, ax = plt.subplots(figsize=(7, 3.8))
lps1 = df.loc[T == 1, "logit_ps"].values
lps0 = df.loc[T == 0, "logit_ps"].values

clip = (-8, 8)
lps1c = np.clip(lps1, *clip)
lps0c = np.clip(lps0, *clip)

xgrid = np.linspace(-8, 8, 500)
kde1  = gaussian_kde(lps1c, bw_method=0.3)
kde0  = gaussian_kde(lps0c, bw_method=0.3)

ax.fill_between(xgrid, kde1(xgrid), alpha=0.45, color="#4472C4", label="Treated (T=1)")
ax.fill_between(xgrid, kde0(xgrid), alpha=0.45, color="#ED7D31", label="Control (T=0)")
ax.plot(xgrid, kde1(xgrid), color="#4472C4", lw=1.5)
ax.plot(xgrid, kde0(xgrid), color="#ED7D31", lw=1.5)

ax.set_xlabel("Logit Propensity Score  logit P(T=1 | X)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("Overlap Check: Logit-PS Density by Treatment Arm", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR_IMGS, "ps_overlap_logit.png"), dpi=150)
print(f"Saved: {OUT_DIR_IMGS}/ps_overlap_logit.png")
plt.close()

# ── Covariate balance: SMD before and after IPW ───────────────────────────────
print("\nComputing covariate balance (SMD before/after IPW)...")

# Stabilised IPW weights (Hajek, same as in estimation.py)
p1_prop = T.mean()
p0_prop = 1 - p1_prop
e_clip  = np.clip(ps, 1e-4, 1 - 1e-4)
w_ipw   = np.where(T == 1, p1_prop / e_clip, p0_prop / (1 - e_clip))

def smd_raw(x, t):
    """Standardized mean difference (pooled-SD denominator)."""
    x1, x0 = x[t == 1], x[t == 0]
    s = np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2)
    return (x1.mean() - x0.mean()) / s if s > 0 else 0.0

def smd_wtd(x, t, w):
    """Weighted SMD using IPW weights."""
    x1, x0 = x[t == 1], x[t == 0]
    w1, w0 = w[t == 1], w[t == 0]
    m1 = np.average(x1, weights=w1)
    m0 = np.average(x0, weights=w0)
    v1 = np.average((x1 - m1) ** 2, weights=w1)
    v0 = np.average((x0 - m0) ** 2, weights=w0)
    s  = np.sqrt((v1 + v0) / 2)
    return (m1 - m0) / s if s > 0 else 0.0

# Variables for SMD (original scale, before dummying)
smd_vars = {
    "MAGER":     ("Mother's Age",                df["MAGER"].values.astype(float)),
    "MEDUC":     ("Mother's Education",           df["MEDUC"].values.astype(float)),
    "MRACEHISP": ("Race/Hispanic Origin",         df["MRACEHISP"].values.astype(float)),
    "DMAR":      ("Marital Status",               df["DMAR"].values.astype(float)),
    "WIC_Y":     ("WIC Participation",            df["WIC_Y"].values.astype(float)),
    "PAY_REC":   ("Payment/Insurance",            df["PAY_REC"].values.astype(float)),
    "BMI_R":     ("Pre-pregnancy BMI",            df["BMI_R"].values.astype(float)),
    "FEDUC":     ("Father's Education",           df["FEDUC"].values.astype(float)),
    "FAGE11":    ("Father's Age Group",           df["FAGE11"].values.astype(float)),
    "DM":        ("Pre-preg. Diabetes",           df["DM"].values.astype(float)),
    "HTN":       ("Pre-preg. Hypertension",       df["HTN"].values.astype(float)),
    "PRIORLIVE": ("Parity (prior live births)",   df["PRIORLIVE"].values.astype(float)),
}

smd_rows = []
for key, (label, x) in smd_vars.items():
    raw = smd_raw(x, T)
    wtd = smd_wtd(x, T, w_ipw)
    smd_rows.append({"Variable": label, "SMD_raw": raw, "SMD_ipw": wtd})
    print(f"  {label:<32}  raw={raw:+.4f}  ipw={wtd:+.4f}")

smd_df = pd.DataFrame(smd_rows)
smd_df.to_csv(os.path.join(OUT_DIR, "covariate_balance_smd.csv"),
              index=False, float_format="%.4f")
print("  Saved → results/covariate_balance_smd.csv")

# Love plot
fig, ax = plt.subplots(figsize=(7, 5.5))
ypos = np.arange(len(smd_df))
labels = smd_df["Variable"].tolist()

ax.scatter(smd_df["SMD_raw"].abs(), ypos, marker="o", s=60,
           color="#4472C4", label="Before IPW (raw)", zorder=3)
ax.scatter(smd_df["SMD_ipw"].abs(), ypos, marker="^", s=60,
           color="#C00000", label="After IPW (weighted)", zorder=3)

for i in ypos:
    ax.plot([smd_df["SMD_raw"].abs().iloc[i], smd_df["SMD_ipw"].abs().iloc[i]],
            [i, i], color="grey", lw=0.8, alpha=0.6, zorder=2)

ax.axvline(0.1, color="black", lw=1, ls="--", alpha=0.5,
           label="|SMD| = 0.10 threshold")
ax.axvline(0.0, color="black", lw=0.5, alpha=0.3)

ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Absolute Standardized Mean Difference  |SMD|", fontsize=10)
ax.set_title("Covariate Balance Before and After IPW Weighting", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="x", alpha=0.2)
ax.set_xlim(left=0)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR_IMGS, "covariate_balance_love.png"), dpi=150)
print("  Saved → results/imgs/covariate_balance_love.png")
plt.close()

print("\nDone.")
