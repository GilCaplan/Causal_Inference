# Dataset: CDC 2015 Cohort Linked Birth/Infant Death

## Download

**ZIP (225 MB):**
```
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/cohortlinkedus/LinkCO15US.zip
```

**Documentation (PDF):**
```
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/cohortlinkedus/LinkCO15Guide.pdf
```

**CDC landing page:**
```
https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm
```
(Navigate to: Linked Birth / Infant Death Data → 2015 Cohort)

## Contents of ZIP

| File | Size | Description |
|---|---|---|
| `VS15LKBC.PublicUse.DUSDENOM` | 5.1 GB | **Main analysis file** — all ~3.98M births, with death variables appended |
| `VS15LKBC.PublicUse.USNUMPUB_2019_11_25` | 39 MB | Numerator — only the 23,357 infant deaths (full death certificate) |

## Re-extracting the analytic CSV

If `data/analytic_extract.csv` is lost, re-download the ZIP and run:
```bash
cd "Project/"
unzip data/LinkCO15US.zip -d data/
python scripts/extract_analytic_dataset.py
```

This reads the 5.1 GB file once and writes a ~150-200 MB CSV with only the
28 columns needed for the causal analysis. Delete the raw `.DUSDENOM` file
afterward to save disk space.

## Key variable positions (1-indexed, record length = 1743 bytes)

| Variable | Position | Description |
|---|---|---|
| `FLGND` | 1346 | Death indicator: `1`=died, blank=survived |
| `AGED` | 1356–1358 | Age at death in days (000–365); blank=survived |
| `CIG0_R` | 261 | Pre-pregnancy smoking recode (0=none, 1–5 category, 6=unknown) |
| `CIG1_R` | 262 | 1st trimester smoking recode |
| `CIG2_R` | 263 | 2nd trimester smoking recode |
| `CIG3_R` | 264 | 3rd trimester smoking recode |
| `F_TOBACCO` | 270 | Tobacco reporting flag — **must be 1** to use smoking vars |
| `MEDUC` | 124 | Mother's education (1–8, 9=unknown) |
| `MAGER` | 75–76 | Mother's age recode |
| `MRACEHISP` | 117 | Mother's race/Hispanic origin combined |
| `DMAR` | 120 | Marital status (1=married, 2=unmarried) |
| `MEDUC` | 124 | Mother's education |
| `PRIORLIVE` | 171–172 | Prior births now living (parity proxy) |
| `WIC` | 251 | WIC participation (Y/N/U) |
| `PAY_REC` | 436 | Insurance recode (1=Medicaid, 2=Private, 3=Self-pay, 4=Other) |
| `RF_PDIAB` | 313 | Pre-pregnancy diabetes (Y/N/U; use with `F_RF_PDIAB`) |
| `RF_PHYPE` | 315 | Pre-pregnancy hypertension (Y/N/U; use with `F_RF_PHYPE`) |
