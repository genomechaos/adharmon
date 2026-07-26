"""Canonical schema for harmonized Alzheimer's disease sample metadata.

This module is the single source of truth for the target schema. Every source
dataset is mapped onto :data:`CANONICAL_COLUMNS` before it is allowed into the
silver layer, so downstream consumers can rely on a stable contract regardless
of how the upstream study chose to label its columns.
"""

from __future__ import annotations

from enum import Enum


class Sex(str, Enum):
    """Harmonized sex assigned at birth, as reported by the source study."""

    MALE = "M"
    FEMALE = "F"
    UNKNOWN = "unknown"


class Diagnosis(str, Enum):
    """Harmonized clinical / neuropathological diagnosis grouping.

    Deliberately coarse. Source studies use incompatible diagnostic criteria
    (clinical vs. neuropathological, CERAD vs. NIA-Reagan), so collapsing to
    these five buckets is the widest grouping that remains defensible across
    cohorts. Finer-grained source labels are preserved in the bronze layer.
    """

    AD = "AD"
    MCI = "MCI"
    CONTROL = "CONTROL"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


#: Ages above this are censored per HIPAA Safe Harbor de-identification.
HIPAA_AGE_CAP = 90.0

#: Braak neurofibrillary staging is defined on 0-VI.
BRAAK_MIN, BRAAK_MAX = 0, 6

#: CERAD neuritic plaque score, 0 (none) to 3 (frequent).
CERAD_MIN, CERAD_MAX = 0, 3

#: RNA Integrity Number, as reported by Agilent Bioanalyzer.
RIN_MIN, RIN_MAX = 1.0, 10.0

#: Plausible bounds for post-mortem interval in hours. Values outside this
#: range are almost always unit errors (minutes or days recorded as hours).
PMI_MIN, PMI_MAX = 0.0, 120.0

#: Plausible bounds for age in years, before Safe Harbor capping.
AGE_MIN, AGE_MAX = 0.0, 120.0


#: Canonical column order for the silver layer.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "source_study",
    "subject_id",
    "age_years",
    "age_censored",
    "sex",
    "diagnosis",
    "braak_stage",
    "cerad_score",
    "apoe_genotype",
    "brain_region",
    "platform",
    "rin",
    "pmi_hours",
    "batch",
)

#: Columns that must be non-null for a record to reach the silver layer.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "source_study",
    "diagnosis",
)

#: Target pandas dtypes for the silver layer. Nullable extension dtypes are
#: used throughout so that "missing" survives a round trip to Parquet without
#: silently becoming NaN-as-float or the string "nan".
CANONICAL_DTYPES: dict[str, str] = {
    "sample_id": "string",
    "source_study": "string",
    "subject_id": "string",
    "age_years": "Float64",
    "age_censored": "boolean",
    "sex": "string",
    "diagnosis": "string",
    "braak_stage": "Int64",
    "cerad_score": "Int64",
    "apoe_genotype": "string",
    "brain_region": "string",
    "platform": "string",
    "rin": "Float64",
    "pmi_hours": "Float64",
    "batch": "string",
}
