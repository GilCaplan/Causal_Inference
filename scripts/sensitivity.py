"""
Sensitivity analyses for the prenatal smoking → infant mortality causal estimate.

Runs automatically after estimation.py (reads results/scores.parquet).

Checks
------
1. E-value for unmeasured confounding (alcohol / mental health)
2. Parametric hidden confounder sensitivity: contour plot over (δ, γ) grid
3. Dose-response: re-estimate AIPW with higher smoking thresholds
4. PS trimming: 1%/99% and 5%/95% of treated PS
5. "Throughout" operationalisation: require 2-of-3 vs all-3 trimesters
6. Self-reported smoking misclassification: corrected ATE under varying sensitivity

Outputs
-------
results/sensitivity_evalue.txt
results/sensitivity_hidden_confounder.png
results/sensitivity_dose_response.csv / .png
results/sensitivity_trimming.csv
results/sensitivity_throughout.csv
results/sensitivity_misclassification.csv
results/sensitivity_summary.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os, warnings
warnings.filterwarnings("ignore")

ROOT    = os.path.join(os.path.dirname(__file__), "..")
DATA    = os.path.join(ROOT, "data", "analytic_extract.csv")
SCORES  = os.path.join(ROOT, "results", "scores.parquet")
OUT_DIR      = os.path.join(ROOT, "results")
OUT_DIR_IMGS = os.path.join(ROOT, "results", "imgs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR_IMGS, exist_ok=True)

# ── Shared helpers ────────────────────────────────────────────────────────────

def eif_se(psi):
    return (psi - psi.mean()).std(ddof=1) / np.sqrt(len(psi))

def aipw(T, Y, e, mu1, mu0):
    e_c  = np.clip(e, 1e-4, 1 - 1e-4)
    psi  = (mu1 - mu0
            + T       * (Y - mu1) / e_c
            - (1 - T) * (Y - mu0) / (1 - e_c))
    return psi.mean(), eif_se(psi)

def fit_ps_outcome(X_sc, T, Y):
    ps_m = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    ps_m.fit(X_sc, T)
    e = ps_m.predict_proba(X_sc)[:, 1]
    t1 = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    t0 = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    t1.fit(X_sc[T == 1], Y[T == 1])
    t0.fit(X_sc[T == 0], Y[T == 0])
    mu1 = t1.predict_proba(X_sc)[:, 1]
    mu0 = t0.predict_proba(X_sc)[:, 1]
    return e, mu1, mu0

def load_base(path):
    """Load and filter analytic sample; return df with numeric covariates."""
    df = pd.read_csv(path, dtype=str)
    df = df[df["F_TOBACCO"] == "1"].copy()
    num_cols = ["CIG0_R","CIG1_R","CIG2_R","CIG3_R",
                "MAGER","MRACEHISP","DMAR","MEDUC","PAY_REC",
                "BMI_R","FEDUC","FAGE11","PRIORLIVE","F_M_HT",
                "F_RF_PDIAB","F_RF_PHYPE"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["DIED"] = pd.to_numeric(df["DIED"], errors="coerce")
    df["DM"]    = (df["RF_PDIAB"] == "Y").astype(int)
    df["HTN"]   = (df["RF_PHYPE"] == "Y").astype(int)
    df["WIC_Y"] = (df["WIC"]      == "Y").astype(int)
    df = df[df["MEDUC"].between(1, 8)].copy()
    df = df[df["PAY_REC"].between(1, 4)].copy()
    df = df[df["MRACEHISP"].between(1, 9)].copy()
    df = df[df["BMI_R"].between(1, 6)].copy()        # 6-cat recode; 9=unknown excluded
    df = df[df["FEDUC"].between(1, 8)].copy()
    df = df[df["FAGE11"].between(1, 11)].copy()
    df = df[(df["F_RF_PDIAB"] == 1) & (df["F_RF_PHYPE"] == 1)].copy()
    df = df[df["PRIORLIVE"] != 99].copy()
    df["PRIORLIVE"] = df["PRIORLIVE"].clip(upper=5)
    df = df[df["MAGER"].notna() & df["DIED"].notna()].copy()
    cat_cols = ["MRACEHISP","DMAR","MEDUC","PAY_REC","BMI_R","FEDUC","FAGE11"]
    X_df = pd.get_dummies(df[cat_cols].astype(str), drop_first=True)
    X_df["MAGER"]     = df["MAGER"].values
    X_df["WIC"]       = df["WIC_Y"].values
    X_df["DM"]        = df["DM"].values
    X_df["HTN"]       = df["HTN"].values
    X_df["PRIORLIVE"] = df["PRIORLIVE"].values
    return df, X_df

# ── Load primary results ──────────────────────────────────────────────────────

print("Loading primary AIPW result...")
sc = pd.read_parquet(SCORES)
T_main   = sc["T"].values
Y_main   = sc["Y"].values
e_main   = sc["ps"].values
mu1_main = sc["mu1"].values
mu0_main = sc["mu0"].values

ate_main, se_main = aipw(T_main, Y_main, e_main, mu1_main, mu0_main)
ci_lo = ate_main - 1.96 * se_main
ci_hi = ate_main + 1.96 * se_main

p0_main = Y_main[T_main == 0].mean()
rr_main = (p0_main + ate_main) / p0_main

print(f"  Primary ATE: {ate_main*1000:+.4f}/1k  95% CI [{ci_lo*1000:+.4f}, {ci_hi*1000:+.4f}]")
print(f"  Baseline mortality (controls): {p0_main*1000:.4f}/1k")
print(f"  Implied risk ratio: {rr_main:.4f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. E-VALUE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evalue(rr):
    rr = max(rr, 1 / rr)
    return rr + np.sqrt(rr * (rr - 1))

rr_ci_lo = (p0_main + ci_lo) / p0_main
ev_point = evalue(rr_main)
ev_ci    = evalue(rr_ci_lo)

evalue_lines = [
    "=" * 60,
    "E-VALUE FOR UNMEASURED CONFOUNDING",
    "=" * 60,
    f"Primary AIPW estimate (risk difference): {ate_main*1000:+.4f}/1k",
    f"95% CI: [{ci_lo*1000:+.4f}, {ci_hi*1000:+.4f}]/1k",
    f"Implied risk ratio (point): {rr_main:.4f}",
    f"Implied risk ratio (CI bound nearest null): {rr_ci_lo:.4f}",
    "",
    f"E-value (point estimate): {ev_point:.3f}",
    f"E-value (CI bound):       {ev_ci:.3f}",
    "",
    "Interpretation:",
    f"  An unmeasured confounder would need to be associated with",
    f"  BOTH treatment AND outcome by a risk ratio of at least",
    f"  {ev_point:.2f}-fold (point) / {ev_ci:.2f}-fold (CI bound) to",
    f"  fully explain away the observed effect.",
    "",
    "Context (alcohol/mental health):",
    "  Alcohol use during pregnancy is associated with smoking with",
    "  OR ≈ 2-4 (literature estimates). Its association with infant",
    "  death (all-cause) is estimated at RR ≈ 2-3 (FASD, SIDS).",
    "  For alcohol to explain away the result it would need RR_TU",
    "  AND RR_YU both >= E-value, which is not supported by evidence.",
    "  See parametric sensitivity plot for the full picture.",
    "=" * 60,
]
evalue_text = "\n".join(evalue_lines)
print("\n" + evalue_text)
with open(os.path.join(OUT_DIR, "sensitivity_evalue.txt"), "w") as f:
    f.write(evalue_text)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PARAMETRIC HIDDEN-CONFOUNDER SENSITIVITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\nBuilding hidden-confounder sensitivity surface...")

delta_vals = np.linspace(0, 0.60, 200)
gamma_vals = np.linspace(0, 0.030, 200)

D, G = np.meshgrid(delta_vals, gamma_vals)
ATE_adj = ate_main - D * G

fig, ax = plt.subplots(figsize=(8, 5.5))
levels_fill = np.linspace(ATE_adj.min(), ATE_adj.max(), 100)
cf = ax.contourf(D * 100, G * 1000, ATE_adj * 1000,
                 levels=levels_fill, cmap="RdBu_r")
plt.colorbar(cf, ax=ax, label="Bias-adjusted ATE (additional deaths/1k)")
cs0 = ax.contour(D * 100, G * 1000, ATE_adj * 1000,
                 levels=[0], colors=["black"], linewidths=2.5)
ax.clabel(cs0, fmt="ATE = 0", fontsize=9)
ax.add_patch(plt.Rectangle((5, 5), 10, 10,
                            fill=False, edgecolor="#FF7F00",
                            linewidth=2, linestyle="--",
                            label="Plausible range: alcohol"))
ax.add_patch(plt.Rectangle((10, 2), 10, 6,
                            fill=False, edgecolor="#9B59B6",
                            linewidth=2, linestyle="--",
                            label="Plausible range: mental health"))
ax.set_xlabel("δ = P(U=1|T=1) − P(U=1|T=0)  (percentage points)", fontsize=10)
ax.set_ylabel("γ = Risk-difference effect of U on Y  (deaths/1,000 births)", fontsize=10)
ax.set_title("Parametric Sensitivity: How strong would an unmeasured confounder\n"
             "need to be to explain away the observed ATE?", fontsize=10)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR_IMGS, "sensitivity_hidden_confounder.png"), dpi=150)
plt.close()
print("  Saved → results/sensitivity_hidden_confounder.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. DOSE-RESPONSE SENSITIVITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\nDose-response sensitivity (re-loading data)...")

df_base, X_base = load_base(DATA)
scaler_dr = StandardScaler()
X_sc_dr   = scaler_dr.fit_transform(X_base.values.astype(float))

threshold_labels = {
    1: "≥1 cig/day (any; primary)",
    2: "≥6 cig/day (≥1 pack/3)",
    3: "≥11 cig/day (half-pack+)",
}

dose_rows = []
for thresh, label in threshold_labels.items():
    treated_mask = ((df_base["CIG0_R"] >= thresh) &
                    (df_base["CIG1_R"] >= thresh) &
                    (df_base["CIG2_R"] >= thresh) &
                    (df_base["CIG3_R"] >= thresh))
    control_mask = (df_base["CIG0_R"] == 0) & (df_base["CIG_REC"] == "N")
    mask = treated_mask | control_mask
    T_d = treated_mask[mask].values.astype(int)
    Y_d = df_base.loc[mask, "DIED"].values
    Xsc  = X_sc_dr[mask.values]
    e_d, mu1_d, mu0_d = fit_ps_outcome(Xsc, T_d, Y_d)
    ate_d, se_d = aipw(T_d, Y_d, e_d, mu1_d, mu0_d)
    dose_rows.append({
        "Label":       label,
        "Threshold":   thresh,
        "N_treated":   T_d.sum(),
        "N_control":   (T_d == 0).sum(),
        "ATE_1k":      ate_d * 1000,
        "SE_1k":       se_d  * 1000,
        "CI_lo_1k":    (ate_d - 1.96*se_d) * 1000,
        "CI_hi_1k":    (ate_d + 1.96*se_d) * 1000,
    })
    print(f"  Thresh={thresh} ({label}): N_t={T_d.sum():,}  "
          f"ATE={ate_d*1000:+.4f}/1k  [{(ate_d-1.96*se_d)*1000:+.4f}, {(ate_d+1.96*se_d)*1000:+.4f}]")

dose_df = pd.DataFrame(dose_rows)
dose_df.to_csv(os.path.join(OUT_DIR, "sensitivity_dose_response.csv"),
               index=False, float_format="%.4f")

fig, ax = plt.subplots(figsize=(7, 3.5))
thresholds = dose_df["Threshold"].values
ates   = dose_df["ATE_1k"].values
ci_los = dose_df["CI_lo_1k"].values
ci_his = dose_df["CI_hi_1k"].values
ax.errorbar(thresholds, ates,
            yerr=[ates - ci_los, ci_his - ates],
            fmt="o-", color="#C00000", capsize=5, lw=2, ms=7)
ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
ax.set_xticks(thresholds)
ax.set_xticklabels([d["Label"] for d in dose_rows], fontsize=8)
ax.set_ylabel("AIPW ATE  (deaths/1,000 births)  95% CI", fontsize=9)
ax.set_title("Dose-Response Sensitivity: ATE by Smoking Intensity Threshold", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR_IMGS, "sensitivity_dose_response.png"), dpi=150)
plt.close()
print("  Saved → results/sensitivity_dose_response.csv / .png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. PS TRIMMING SENSITIVITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\nPS trimming sensitivity...")
logit_e_main = sc["logit_ps"].values

trim_rows = [{"Trim": "None (full sample)",
              "N": len(T_main), "ATE_1k": ate_main*1000,
              "SE_1k": se_main*1000,
              "CI_lo_1k": ci_lo*1000, "CI_hi_1k": ci_hi*1000}]

for lo_p, hi_p in [(1, 99), (5, 95)]:
    ps_t = e_main[T_main == 1]
    lo_v, hi_v = np.percentile(ps_t, [lo_p, hi_p])
    keep = (e_main >= lo_v) & (e_main <= hi_v)
    T_tr, Y_tr = T_main[keep], Y_main[keep]
    e_tr  = e_main[keep]
    mu1_tr = mu1_main[keep]
    mu0_tr = mu0_main[keep]
    ate_tr, se_tr = aipw(T_tr, Y_tr, e_tr, mu1_tr, mu0_tr)
    trim_rows.append({
        "Trim": f"{lo_p}th/{hi_p}th pct of treated PS",
        "N":    keep.sum(),
        "ATE_1k":   ate_tr * 1000,
        "SE_1k":    se_tr  * 1000,
        "CI_lo_1k": (ate_tr - 1.96*se_tr) * 1000,
        "CI_hi_1k": (ate_tr + 1.96*se_tr) * 1000,
    })
    print(f"  Trim {lo_p}/{hi_p}: N={keep.sum():,}  "
          f"ATE={ate_tr*1000:+.4f}  [{(ate_tr-1.96*se_tr)*1000:+.4f}, {(ate_tr+1.96*se_tr)*1000:+.4f}]")

trim_df = pd.DataFrame(trim_rows)
trim_df.to_csv(os.path.join(OUT_DIR, "sensitivity_trimming.csv"),
               index=False, float_format="%.4f")
print("  Saved → results/sensitivity_trimming.csv")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. "THROUGHOUT PREGNANCY" OPERATIONALISATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n'Throughout pregnancy' operationalisation sensitivity...")

throughout_defs = {
    "Primary (CIG0≥1 + all 3 tri)": lambda d: (
        (d["CIG0_R"] >= 1) & (d["CIG1_R"] >= 1) &
        (d["CIG2_R"] >= 1) & (d["CIG3_R"] >= 1)),
    "All 3 trimest. only (no CIG0)": lambda d: (
        (d["CIG1_R"] >= 1) & (d["CIG2_R"] >= 1) & (d["CIG3_R"] >= 1)),
    "Any 2-of-3 trimesters ≥1": lambda d: (
        ((d["CIG1_R"] >= 1).astype(int) +
         (d["CIG2_R"] >= 1).astype(int) +
         (d["CIG3_R"] >= 1).astype(int)) >= 2),
}

throughout_rows = []
for label, treated_fn in throughout_defs.items():
    treated_mask = treated_fn(df_base)
    control_mask = (df_base["CIG0_R"] == 0) & (df_base["CIG_REC"] == "N")
    mask  = treated_mask | control_mask
    T_w   = treated_mask[mask].values.astype(int)
    Y_w   = df_base.loc[mask, "DIED"].values
    Xsc_w = X_sc_dr[mask.values]
    e_w, mu1_w, mu0_w = fit_ps_outcome(Xsc_w, T_w, Y_w)
    ate_w, se_w = aipw(T_w, Y_w, e_w, mu1_w, mu0_w)
    throughout_rows.append({
        "Definition": label,
        "N_treated":  T_w.sum(),
        "ATE_1k":     ate_w * 1000,
        "SE_1k":      se_w  * 1000,
        "CI_lo_1k":   (ate_w - 1.96*se_w) * 1000,
        "CI_hi_1k":   (ate_w + 1.96*se_w) * 1000,
    })
    print(f"  {label:<40} N_t={T_w.sum():,}  "
          f"ATE={ate_w*1000:+.4f}  [{(ate_w-1.96*se_w)*1000:+.4f}, {(ate_w+1.96*se_w)*1000:+.4f}]")

throughout_df = pd.DataFrame(throughout_rows)
throughout_df.to_csv(os.path.join(OUT_DIR, "sensitivity_throughout.csv"),
                     index=False, float_format="%.4f")
print("  Saved → results/sensitivity_throughout.csv")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. MISCLASSIFICATION SENSITIVITY
#    Smoking is self-reported; social-desirability bias causes some true
#    sustained smokers to deny smoking (false negatives → dilute control group).
#    Under nondifferential misclassification with sensitivity Se (= fraction of
#    true smokers correctly classified as T=1):
#        ATE_observed ≈ Se × ATE_true
#    → ATE_true ≈ ATE_observed / Se
#    This is a conservative lower-bound correction; the observed estimate is an
#    attenuated (lower-bound) version of the true ATE.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\nMisclassification sensitivity (nondifferential, varying Se)...")

sensitivities = [1.0, 0.90, 0.80, 0.70]
misc_rows = []
for Se in sensitivities:
    ate_corrected   = ate_main / Se
    ci_lo_corrected = ci_lo / Se
    ci_hi_corrected = ci_hi / Se
    misc_rows.append({
        "Se (classification sensitivity)": Se,
        "Underreporting (1-Se)":           round(1 - Se, 2),
        "Corrected ATE (per 1k)":          ate_corrected * 1000,
        "Corrected CI lo (per 1k)":        ci_lo_corrected * 1000,
        "Corrected CI hi (per 1k)":        ci_hi_corrected * 1000,
    })
    print(f"  Se={Se:.2f}  corrected ATE={ate_corrected*1000:+.4f}  "
          f"[{ci_lo_corrected*1000:+.4f}, {ci_hi_corrected*1000:+.4f}]")

misc_df = pd.DataFrame(misc_rows)
misc_df.to_csv(os.path.join(OUT_DIR, "sensitivity_misclassification.csv"),
               index=False, float_format="%.4f")
print("  Saved → results/sensitivity_misclassification.csv")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Singleton restriction  (DPLURAL = 1 only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n6. Singleton restriction (DPLURAL = '1')...")
df_base, X_base = load_base(DATA)

# Apply treatment / control definition (same as main analysis)
treated_mask = ((df_base["CIG0_R"] >= 1) & (df_base["CIG1_R"] >= 1) &
                (df_base["CIG2_R"] >= 1) & (df_base["CIG3_R"] >= 1))
control_mask = (df_base["CIG0_R"] == 0) & (df_base["CIG_REC"] == "N")
df_base = df_base[treated_mask | control_mask].copy()
X_base  = X_base.loc[df_base.index]

T_base = treated_mask[df_base.index].astype(int).values
Y_base = df_base["DIED"].values

singleton_rows = []

for label, mask in [
    ("Full analytic sample",   pd.Series(True, index=df_base.index)),
    ("Singletons only (DPLURAL=1)", df_base["DPLURAL"] == "1"),
]:
    idx = mask[mask].index if hasattr(mask, "index") else df_base.index
    if hasattr(mask, "values"):
        sel = mask.values
    else:
        sel = np.ones(len(df_base), dtype=bool)

    X_s = X_base.loc[df_base.index[sel]].values.astype(float)
    T_s = T_base[sel]
    Y_s = Y_base[sel]

    sc_s = StandardScaler()
    X_sc_s = sc_s.fit_transform(X_s)
    e_s, mu1_s, mu0_s = fit_ps_outcome(X_sc_s, T_s, Y_s)
    ate_s, se_s = aipw(T_s, Y_s, e_s, mu1_s, mu0_s)
    ci_lo_s = ate_s - 1.96 * se_s
    ci_hi_s = ate_s + 1.96 * se_s

    n_treated = T_s.sum()
    n_total   = len(T_s)
    print(f"  {label}: N={n_total:,}  T={n_treated:,}  "
          f"ATE={ate_s*1000:+.4f}  95% CI [{ci_lo_s*1000:+.4f}, {ci_hi_s*1000:+.4f}]")
    singleton_rows.append({
        "Sample": label,
        "N": n_total,
        "N_treated": int(n_treated),
        "ATE_1k": round(ate_s * 1000, 4),
        "SE_1k":  round(se_s  * 1000, 4),
        "CI_lo_1k": round(ci_lo_s * 1000, 4),
        "CI_hi_1k": round(ci_hi_s * 1000, 4),
    })

singleton_df = pd.DataFrame(singleton_rows)
singleton_df.to_csv(os.path.join(OUT_DIR, "sensitivity_singleton.csv"),
                    index=False, float_format="%.4f")
print("  Saved → results/sensitivity_singleton.csv")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

summary_lines = [
    "=" * 70,
    "SENSITIVITY ANALYSIS SUMMARY",
    "=" * 70,
    f"Primary AIPW ATE: {ate_main*1000:+.4f}/1k  "
    f"95% CI [{ci_lo*1000:+.4f}, {ci_hi*1000:+.4f}]",
    "",
    "1. E-value (hidden confounder must have RR ≥ X with BOTH T and Y):",
    f"   Point estimate E-value: {ev_point:.2f}",
    f"   CI bound E-value:       {ev_ci:.2f}",
    "   (Literature: alcohol RR_T ≈ 2-4, RR_Y ≈ 2-3 → insufficient to explain away)",
    "",
    "2. Dose-response (monotone increase expected if causal):",
]
for r in dose_rows:
    summary_lines.append(
        f"   {r['Label']:<38}: ATE={r['ATE_1k']:+.4f}  "
        f"[{r['CI_lo_1k']:+.4f}, {r['CI_hi_1k']:+.4f}]")
summary_lines += [
    "",
    "3. PS trimming:",
]
for r in trim_rows:
    summary_lines.append(
        f"   {r['Trim']:<42}: ATE={r['ATE_1k']:+.4f}  "
        f"[{r['CI_lo_1k']:+.4f}, {r['CI_hi_1k']:+.4f}]")
summary_lines += [
    "",
    "4. 'Throughout' operationalisation:",
]
for r in throughout_rows:
    summary_lines.append(
        f"   {r['Definition']:<42}: ATE={r['ATE_1k']:+.4f}  "
        f"[{r['CI_lo_1k']:+.4f}, {r['CI_hi_1k']:+.4f}]")
summary_lines += [
    "",
    "5. Misclassification correction (nondifferential, Se = P(T_obs=1|T_true=1)):",
]
for r in misc_rows:
    summary_lines.append(
        f"   Se={r['Se (classification sensitivity)']:.2f}  "
        f"corrected ATE={r['Corrected ATE (per 1k)']:+.4f}  "
        f"[{r['Corrected CI lo (per 1k)']:+.4f}, {r['Corrected CI hi (per 1k)']:+.4f}]")
summary_lines += [
    "",
    "6. Singleton restriction (DPLURAL=1 excludes twins/multiples):",
]
for r in singleton_rows:
    summary_lines.append(
        f"   {r['Sample']:<42}: ATE={r['ATE_1k']:+.4f}  "
        f"[{r['CI_lo_1k']:+.4f}, {r['CI_hi_1k']:+.4f}]")
summary_lines.append("=" * 70)
summary_text = "\n".join(summary_lines)

print("\n" + summary_text)
with open(os.path.join(OUT_DIR, "sensitivity_summary.txt"), "w") as f:
    f.write(summary_text)
print(f"\nSaved → results/sensitivity_summary.txt")
