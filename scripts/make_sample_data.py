#!/usr/bin/env python3
"""Stage synthetic bronze-layer metadata for local runs and CI.

Generates three CSVs whose annotation conventions disagree the way real cohorts
do. Synthetic, but structurally faithful: replace with real GEO series matrices
once you have confirmed the actual ``characteristics_ch1`` labels.

Usage::

    python scripts/make_sample_data.py --root ./data
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

BRAIN_REGIONS = ["entorhinal cortex", "hippocampus", "temporal cortex", "prefrontal cortex"]


def gse5281(rng: random.Random, n: int = 40) -> pd.DataFrame:
    """Affymetrix study: plain integer ages, M/F sex, no neuropathology."""
    return pd.DataFrame(
        {
            "geo_accession": [f"GSM{119600 + i}" for i in range(n)],
            "subject": [f"S{i:03d}" for i in range(n)],
            "age": [str(rng.randint(62, 95)) for _ in range(n)],
            "sex": [rng.choice(["M", "F", "male", "female"]) for _ in range(n)],
            "disease_state": [rng.choice(["Alzheimer's Disease", "normal", "Alzheimer's Disease"]) for _ in range(n)],
            "region": [rng.choice(BRAIN_REGIONS[:2]) for _ in range(n)],
            "apoe": [rng.choice(["33", "34", "44", "23", "NA"]) for _ in range(n)],
        }
    )


def gse33000(rng: random.Random, n: int = 40) -> pd.DataFrame:
    """Numeric sex, '90+' age ceiling, Roman-numeral Braak, some ranges."""
    return pd.DataFrame(
        {
            "sample": [f"PFC_{i:03d}" for i in range(n)],
            "donor_id": [f"D{100 + i}" for i in range(n)],
            "age_at_death": [
                rng.choice(["90+", str(rng.randint(60, 89)), f"{rng.randint(60, 89)} years"]) for _ in range(n)
            ],
            "gender": [rng.choice(["1", "2"]) for _ in range(n)],
            "dx": [rng.choice(["AD", "non-demented", "MCI", "control"]) for _ in range(n)],
            "braak": [rng.choice(["0", "I", "II", "III", "IV", "V", "VI", "III-IV", "NA"]) for _ in range(n)],
            "tissue": ["prefrontal cortex"] * n,
            "pmi": [f"{rng.uniform(2, 30):.1f}" for _ in range(n)],
            "scan_batch": [rng.choice(["B1", "B2", "B3"]) for _ in range(n)],
        }
    )


def gse125583(rng: random.Random, n: int = 40) -> pd.DataFrame:
    """RNA-seq: semantic CERAD, bare-digit APOE, RIN reported."""
    return pd.DataFrame(
        {
            "title": [f"TCX_{i:03d}" for i in range(n)],
            "patient": [f"P{i}" for i in range(n)],
            "age": [str(rng.randint(58, 97)) for _ in range(n)],
            "Sex": [rng.choice(["male", "female", "M", "F"]) for _ in range(n)],
            "diagnosis": [rng.choice(["Alzheimer disease", "control", "MCI", "probable AD"]) for _ in range(n)],
            "braak_stage": [rng.choice(["Braak V", "0", "III", "stage 4", "VI"]) for _ in range(n)],
            "cerad": [rng.choice(["frequent", "none", "sparse", "moderate", "2"]) for _ in range(n)],
            "apoe_genotype": [rng.choice(["34", "33", "23", "44", "43"]) for _ in range(n)],
            "brain_region": ["temporal cortex"] * n,
            "rin": [f"{rng.uniform(3.5, 9.5):.1f}" for _ in range(n)],
            "pmi_hrs": [rng.choice([f"{rng.uniform(2, 20):.1f} hours", f"{rng.uniform(2, 20):.1f}"]) for _ in range(n)],
            "seq_batch": [rng.choice(["R1", "R2"]) for _ in range(n)],
        }
    )


BUILDERS = {"GSE5281": gse5281, "GSE33000": gse33000, "GSE125583": gse125583}


def main() -> None:
    """Write one CSV per study into ``<root>/bronze``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="./data", help="Local lake root (default ./data).")
    parser.add_argument("--rows", type=int, default=40, help="Rows per study (default 40).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    args = parser.parse_args()

    bronze = Path(args.root) / "bronze"
    bronze.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for study_id, builder in BUILDERS.items():
        target = bronze / f"{study_id}_metadata.csv"
        builder(rng, args.rows).to_csv(target, index=False)
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
