"""Reduce the full ACS person file to the columns relevant to the causal
question: does US military service (MIL) raise the probability that a veteran
earns above $50k personal income (PINCP)?

Keeps:
  - Treatment: MIL
  - Outcome:   PINCP (also kept raw; we'll binarize at $50k downstream)
  - Pre-treatment / background confounders: AGEP, SEX, RAC1P, HISP, POBP,
    NATIVITY, CIT, LANX, ENG
  - Outcome-side covariates / mediators: SCHL, MAR, ESR, COW, OCCP, WKHP,
    WAGP, DIS, DEAR, DEYE
  - Veteran-specific descriptors: DRAT, VPS
  - Geography / weighting: ST, PWGTP
"""
from folktables import ACSDataSource

KEEP = [
    "MIL", "PINCP",
    "AGEP", "SEX", "RAC1P", "HISP", "POBP", "NATIVITY", "CIT", "LANX", "ENG",
    "SCHL", "MAR", "ESR", "COW", "OCCP", "WKHP", "WAGP",
    "DIS", "DEAR", "DEYE", "DRAT", "VPS",
    "ST", "PWGTP",
]

src = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person",
                    root_dir="data")
df = src.get_data(download=False)
print(f"Full: {len(df):,} rows x {len(df.columns)} cols")

missing = [c for c in KEEP if c not in df.columns]
assert not missing, f"Missing columns: {missing}"

slim = df[KEEP]
slim.to_parquet("acs_2018_slim.parquet", index=False)
slim.to_csv("acs_2018_slim.csv.gz", index=False, compression="gzip")
print(f"Slim: {len(slim):,} rows x {len(slim.columns)} cols")
print(slim.head())
