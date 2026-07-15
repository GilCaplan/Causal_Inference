"""
One-time extraction script: reads the 5.1 GB fixed-width denominator-plus file
and writes a compact CSV containing only the ~28 columns needed for analysis.

Run once, then delete (or archive) the raw .DUSDENOM file to save disk space.
Output: data/analytic_extract.csv  (~150-200 MB)
"""

import pandas as pd
import os
import time

INPUT  = os.path.join(os.path.dirname(__file__), "../data/VS15LKBC.PublicUse.DUSDENOM")
OUTPUT = os.path.join(os.path.dirname(__file__), "../data/analytic_extract.csv")

# Column specs: (start_0indexed, end_exclusive)  from 1-indexed guide positions
# field_name : (start, end)
FIELDS = [
    # Outcome (appended mortality section of denominator-plus file)
    ("FLGND",      (1345, 1346)),  # 1=died (in both files), blank=survived
    ("AGED",       (1355, 1358)),  # Age at death in days 000-365; blank=survived

    # Filter variable
    ("RESTATUS",   (103, 104)),    # 4=foreign resident → exclude

    # ── Treatment variables ──────────────────────────────────────────────────
    ("CIG0_R",     (260, 261)),    # Pre-pregnancy cigs recode (0=none, 1-5, 6=unknown)
    ("CIG1_R",     (261, 262)),    # 1st trimester cigs recode
    ("CIG2_R",     (262, 263)),    # 2nd trimester cigs recode
    ("CIG3_R",     (263, 264)),    # 3rd trimester cigs recode
    ("F_CIGS_0",   (264, 265)),    # Reporting flag: pre-preg cigs (1=reporting)
    ("F_CIGS_1",   (265, 266)),    # Reporting flag: 1st tri cigs
    ("F_CIGS_2",   (266, 267)),    # Reporting flag: 2nd tri cigs
    ("F_CIGS_3",   (267, 268)),    # Reporting flag: 3rd tri cigs
    ("CIG_REC",    (268, 269)),    # Any smoking recode: Y/N/U
    ("F_TOBACCO",  (269, 270)),    # Master tobacco reporting flag (1=use this record)

    # ── Confounders ──────────────────────────────────────────────────────────
    ("FAGE11",     (148, 150)),    # Father's age recode 11 (01-11, 99=unknown); guide pos 149-150
    ("FEDUC",      (162, 163)),    # Father's education (1-8, 9=unknown); guide pos 163
    ("MAGER",      (74,  76)),     # Mother's age (2-digit recode 41-cat)
    ("MRACE6",     (106, 107)),    # Mother's race recode 6
    ("MHISP_R",    (114, 115)),    # Mother's Hispanic origin recode
    ("MRACEHISP",  (116, 117)),    # Mother's race/Hispanic combined
    ("DMAR",       (119, 120)),    # Marital status (1=married, 2=unmarried)
    ("MEDUC",      (123, 124)),    # Mother's education (1-8, 9=unknown)
    ("PRIORLIVE",  (170, 172)),    # Prior births now living
    ("PRIORDEAD",  (172, 174)),    # Prior births now dead
    ("LBO_REC",    (178, 179)),    # Live birth order recode
    ("F_M_HT",     (281, 282)),    # Reporting flag for mother's height/BMI (0/1)
    ("BMI_R",      (286, 287)),    # Pre-pregnancy BMI recode (1-6, 9=unknown); guide pos 287
    ("WIC",        (250, 251)),    # WIC participation (Y/N/U)
    ("RF_PDIAB",   (312, 313)),    # Pre-pregnancy diabetes (Y/N/U)
    ("RF_PHYPE",   (314, 315)),    # Pre-pregnancy hypertension (Y/N/U)
    ("F_RF_PDIAB", (318, 319)),    # Reporting flag for RF_PDIAB
    ("F_RF_PHYPE", (320, 321)),    # Reporting flag for RF_PHYPE
    ("PAY_REC",    (435, 436)),    # Payment/insurance recode (1=Medicaid…4=Other, 9=Unk)
    ("F_PAY_REC",  (437, 438)),    # Reporting flag for PAY_REC
]

names    = [f[0] for f in FIELDS]
colspecs = [f[1] for f in FIELDS]

CHUNKSIZE = 250_000

print(f"Input : {INPUT}")
print(f"Output: {OUTPUT}")
print(f"Reading in chunks of {CHUNKSIZE:,}...")

t0 = time.time()
chunks = []
total = 0

reader = pd.read_fwf(
    INPUT,
    colspecs=colspecs,
    names=names,
    header=None,
    dtype=str,
    chunksize=CHUNKSIZE,
)

for i, chunk in enumerate(reader):
    total += len(chunk)
    chunks.append(chunk)
    elapsed = time.time() - t0
    print(f"  chunk {i+1:3d} | {total:>9,} records | {elapsed:5.0f}s elapsed")

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal records read: {len(df):,}")

# Strip whitespace so blanks become empty strings (important for FLGND, AGED)
df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

# Add a clean binary outcome column for convenience
df["DIED"] = (df["FLGND"] == "1").astype(int)

print(f"Deaths (DIED=1): {df['DIED'].sum():,}")
print(f"Survivors       : {(df['DIED']==0).sum():,}")

df.to_csv(OUTPUT, index=False)
size_mb = os.path.getsize(OUTPUT) / 1e6
print(f"\nSaved → {OUTPUT}  ({size_mb:.0f} MB)")
print(f"Total time: {time.time()-t0:.0f}s")
