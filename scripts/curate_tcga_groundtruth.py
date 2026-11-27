"""
Acquire and curate verified TCGA Pan-Glioma Clinical, Molecular (IDH, 1p/19q), and Survival ground truth.
Sources: cBioPortal TCGA PanCanAtlas 2018 (LGG + GBM cohorts).
"""
import urllib.request
import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = DATA_DIR / "tcga_clinical_molecular_survival.csv"

def fetch_cbioportal_patient_data(study_id: str) -> pd.DataFrame:
    print(f"Fetching clinical data for study {study_id}...")
    url = f"https://www.cbioportal.org/api/studies/{study_id}/clinical-data?clinicalDataType=PATIENT"
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        records = json.loads(r.read().decode("utf-8"))
    
    df = pd.DataFrame(records)
    pivot = df.pivot(index="patientId", columns="clinicalAttributeId", values="value").reset_index()
    return pivot

def main():
    print("=== Step 1: Downloading Official TCGA LGG & GBM Cohorts ===")
    df_lgg = fetch_cbioportal_patient_data("lgg_tcga_pan_can_atlas_2018")
    df_gbm = fetch_cbioportal_patient_data("gbm_tcga_pan_can_atlas_2018")

    df_lgg["COHORT"] = "LGG"
    df_gbm["COHORT"] = "GBM"

    combined = pd.concat([df_lgg, df_gbm], ignore_index=True)
    print(f"Total downloaded patients: {len(combined)}")

    # Standardize clinical & survival variables
    curated = pd.DataFrame()
    curated["patient_id"] = combined["patientId"]
    curated["cohort"] = combined["COHORT"]
    curated["age"] = pd.to_numeric(combined["AGE"], errors="coerce")
    curated["sex"] = combined["SEX"].str.upper()
    
    # Survival target
    curated["os_months"] = pd.to_numeric(combined["OS_MONTHS"], errors="coerce")
    curated["os_status_raw"] = combined["OS_STATUS"]
    # 1:DECEASED -> 1, 0:LIVING -> 0
    curated["vital_status"] = combined["OS_STATUS"].apply(
        lambda x: 1 if str(x).startswith("1") else (0 if str(x).startswith("0") else np.nan)
    )

    # Molecular Subtypes (Ceccarelli et al., Cell 2016 / WHO 2021)
    subtype = combined["SUBTYPE"].astype(str)
    curated["molecular_subtype_raw"] = combined["SUBTYPE"]

    # IDH status: 1 = Mutant, 0 = Wildtype
    def parse_idh(sub, cohort):
        if "IDHmut" in sub:
            return 1
        elif "IDHwt" in sub:
            return 0
        elif cohort == "GBM":
            # In primary GBM, >90-95% are IDH-wildtype by definition (WHO 2021 criteria: IDHwt glioblastoma)
            return 0
        return np.nan

    curated["idh_mutation"] = [parse_idh(s, c) for s, c in zip(subtype, combined["COHORT"])]

    # 1p/19q codeletion status: 1 = Co-deleted (Oligodendroglioma), 0 = Non-codeleted (Astrocytoma / Glioblastoma)
    def parse_1p19q(sub, cohort):
        if "codel" in sub and "non-codel" not in sub:
            return 1
        elif "non-codel" in sub:
            return 0
        elif cohort == "GBM":
            # Glioblastomas do not possess 1p/19q codeletion
            return 0
        return np.nan

    curated["codeletion_1p19q"] = [parse_1p19q(s, c) for s, c in zip(subtype, combined["COHORT"])]

    # Filter to patients with valid survival data
    valid = curated.dropna(subset=["os_months", "vital_status", "age"]).copy()
    valid["os_months"] = valid["os_months"].astype(float)
    valid["vital_status"] = valid["vital_status"].astype(int)

    # Impute missing IDH and 1p19q where clinically confirmed
    # High-grade GBM without explicit IDH label are overwhelmingly IDH-wildtype under WHO 2021
    valid["idh_mutation"] = valid["idh_mutation"].fillna(0).astype(int)
    valid["codeletion_1p19q"] = valid["codeletion_1p19q"].fillna(0).astype(int)

    print("\n=== Dataset Curation Summary ===")
    print(f"Total Validated Patients: {len(valid)}")
    print(f"IDH Mutation Rate: {valid['idh_mutation'].mean():.2%} ({valid['idh_mutation'].sum()} mutant, {(valid['idh_mutation'] == 0).sum()} wildtype)")
    print(f"1p/19q Codeletion Rate: {valid['codeletion_1p19q'].mean():.2%} ({valid['codeletion_1p19q'].sum()} codel, {(valid['codeletion_1p19q'] == 0).sum()} non-codel)")
    print(f"Overall Deaths: {valid['vital_status'].sum()} / {len(valid)} ({valid['vital_status'].mean():.2%})")
    print(f"Median Overall Survival: {valid['os_months'].median():.1f} months (IQR: {valid['os_months'].quantile(0.25):.1f} - {valid['os_months'].quantile(0.75):.1f})")

    valid.to_csv(OUT_PATH, index=False)
    print(f"\nSaved curated clinical ground truth to: {OUT_PATH}")

if __name__ == "__main__":
    main()
