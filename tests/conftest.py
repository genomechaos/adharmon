"""Shared fixtures.

The three raw frames deliberately disagree with each other in every way real
cohorts do: different column names, different sex encodings, Roman vs. Arabic
Braak staging, semantic vs. numeric CERAD, and a Safe Harbor age ceiling in one
study but not the others.
"""

from __future__ import annotations

import pandas as pd
import pytest

from adharmon.mappings import MAPPINGS
from adharmon.storage import LocalLakeStore


@pytest.fixture
def gse5281_raw() -> pd.DataFrame:
    """Affymetrix study: plain ages, M/F sex, no neuropathology."""
    return pd.DataFrame(
        {
            "geo_accession": ["GSM119615", "GSM119616", "GSM119617"],
            "subject": ["S01", "S02", "S03"],
            "age": ["79", "84", "73"],
            "sex": ["M", "F", "male"],
            "disease_state": ["Alzheimer's Disease", "normal", "Alzheimer's Disease"],
            "region": ["entorhinal cortex", "entorhinal cortex", "hippocampus"],
            "apoe": ["34", "33", "44"],
        }
    )


@pytest.fixture
def gse33000_raw() -> pd.DataFrame:
    """Numeric sex, '90+' age ceiling, Roman-numeral Braak."""
    return pd.DataFrame(
        {
            "sample": ["PFC_001", "PFC_002", "PFC_003"],
            "donor_id": ["D100", "D101", "D102"],
            "age_at_death": ["90+", "68", "77 years"],
            "gender": ["1", "2", "1"],
            "dx": ["AD", "non-demented", "AD"],
            "braak": ["VI", "I", "IV"],
            "tissue": ["prefrontal cortex"] * 3,
            "pmi": ["8.5", "12", "6"],
            "scan_batch": ["B1", "B1", "B2"],
        }
    )


@pytest.fixture
def gse125583_raw() -> pd.DataFrame:
    """RNA-seq: semantic CERAD, bare-digit APOE, RIN reported."""
    return pd.DataFrame(
        {
            "title": ["TCX_01", "TCX_02", "TCX_03"],
            "patient": ["P1", "P2", "P3"],
            "age": ["81", "94", "66"],
            "Sex": ["female", "male", "F"],
            "diagnosis": ["Alzheimer disease", "control", "MCI"],
            "braak_stage": ["Braak V", "0", "III"],
            "cerad": ["frequent", "none", "sparse"],
            "apoe_genotype": ["34", "33", "23"],
            "brain_region": ["temporal cortex"] * 3,
            "rin": ["7.8", "6.2", "4.1"],
            "pmi_hrs": ["3.5 hours", "5", "9"],
            "seq_batch": ["R1", "R1", "R2"],
        }
    )


@pytest.fixture
def raw_frames(gse5281_raw, gse33000_raw, gse125583_raw) -> dict[str, pd.DataFrame]:
    return {
        "GSE5281": gse5281_raw,
        "GSE33000": gse33000_raw,
        "GSE125583": gse125583_raw,
    }


@pytest.fixture
def mappings():
    return MAPPINGS


@pytest.fixture
def local_lake(tmp_path, raw_frames) -> LocalLakeStore:
    """A local lake with the three raw studies staged in the bronze layer."""
    store = LocalLakeStore(tmp_path)
    bronze = tmp_path / "bronze"
    bronze.mkdir(parents=True, exist_ok=True)
    for study_id, frame in raw_frames.items():
        frame.to_csv(bronze / f"{study_id}_metadata.csv", index=False)
    return store
