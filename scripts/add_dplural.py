"""
Appends DPLURAL (plurality) to analytic_extract.csv by reading it directly
from the raw fixed-width file for ALL rows (no filtering — filtering happens
downstream in the analysis scripts).

DPLURAL at 0-indexed col 453: 1=singleton, 2=twin, 3=triplet+, 9=unknown.
"""

import pandas as pd
import os, time, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW  = os.path.join(ROOT, "data", "VS15LKBC.PublicUse.DUSDENOM")
CSV  = os.path.join(ROOT, "data", "analytic_extract.csv")

CHUNKSIZE = 250_000

print("Reading DPLURAL from raw file (all rows, no filtering)...")
t0 = time.time()
parts = []
total = 0

reader = pd.read_fwf(
    RAW,
    colspecs=[(453, 454)],
    names=["DPLURAL"],
    header=None,
    dtype=str,
    chunksize=CHUNKSIZE,
)

for i, chunk in enumerate(reader):
    total += len(chunk)
    parts.append(chunk["DPLURAL"])
    if (i + 1) % 4 == 0:
        print(f"  chunk {i+1:3d} | {total:>9,} rows | {time.time()-t0:.0f}s")

dplural = pd.concat(parts, ignore_index=True)
print(f"\nTotal rows from raw file: {len(dplural):,}")

df = pd.read_csv(CSV, dtype=str)
print(f"analytic_extract.csv rows: {len(df):,}")

assert len(dplural) == len(df), \
    f"Mismatch: raw {len(dplural)} vs CSV {len(df)}"

df["DPLURAL"] = dplural.values
df.to_csv(CSV, index=False)

print(f"\nDPLURAL counts (full extract):")
print(dplural.value_counts().sort_index())
print(f"\nSaved → {CSV}  ({time.time()-t0:.0f}s total)")
