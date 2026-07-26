"""Per-study column mappings from source metadata onto the canonical schema.

Each :class:`SourceMapping` is a declarative crosswalk: it says which source
column feeds which canonical field, and which canonical fields are constant for
the whole study. Keeping the crosswalk as data rather than as per-study
transform code means adding a fourth cohort is a config change with a test,
not a new branch in the pipeline.

.. warning::
   The column names below are illustrative. GEO series encode sample metadata
   in free-form ``characteristics_ch1`` fields whose labels differ per
   submission, so confirm the real labels against the series matrix before
   trusting a mapping. :func:`adharmon.harmonize.audit_mapping` reports which
   mapped columns were actually found in an input frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["SourceMapping", "MAPPINGS", "get_mapping"]


@dataclass(frozen=True)
class SourceMapping:
    """Crosswalk from one source study's columns onto the canonical schema.

    Attributes:
        study_id: Accession used as ``source_study`` in the output.
        field_map: Canonical field name -> source column name.
        constants: Canonical field name -> fixed value for every row.
        notes: Provenance and caveats, surfaced in the lineage report.
    """

    study_id: str
    field_map: dict[str, str]
    constants: dict[str, object] = field(default_factory=dict)
    notes: str = ""

    def source_columns(self) -> set[str]:
        """Source column names this mapping expects to find."""
        return set(self.field_map.values())


MAPPINGS: dict[str, SourceMapping] = {
    # Affymetrix microarray across six brain regions, laser-capture
    # microdissected. Reports age as a plain integer and sex as M/F.
    "GSE5281": SourceMapping(
        study_id="GSE5281",
        field_map={
            "sample_id": "geo_accession",
            "subject_id": "subject",
            "age_years": "age",
            "sex": "sex",
            "diagnosis": "disease_state",
            "brain_region": "region",
            "apoe_genotype": "apoe",
        },
        constants={"platform": "Affymetrix HG-U133 Plus 2.0"},
        notes=(
            "Diagnosis column mixes 'Alzheimer's Disease' and 'normal'. "
            "No neuropathology staging reported; braak_stage/cerad_score "
            "will be null for this cohort."
        ),
    ),
    # Prefrontal cortex microarray. Encodes sex numerically (1/2) and age with
    # a '90+' ceiling, and reports Braak stage in Roman numerals.
    "GSE33000": SourceMapping(
        study_id="GSE33000",
        field_map={
            "sample_id": "sample",
            "subject_id": "donor_id",
            "age_years": "age_at_death",
            "sex": "gender",
            "diagnosis": "dx",
            "braak_stage": "braak",
            "brain_region": "tissue",
            "pmi_hours": "pmi",
            "batch": "scan_batch",
        },
        constants={"platform": "Rosetta/Merck Human 44k"},
        notes=(
            "age_at_death uses a '90+' ceiling per Safe Harbor; those rows are "
            "capped at 90 and flagged via age_censored. Braak in Roman numerals."
        ),
    ),
    # Bulk RNA-seq of temporal cortex. Richest neuropathology annotation of the
    # three, and the only one reporting RIN.
    "GSE125583": SourceMapping(
        study_id="GSE125583",
        field_map={
            "sample_id": "title",
            "subject_id": "patient",
            "age_years": "age",
            "sex": "Sex",
            "diagnosis": "diagnosis",
            "braak_stage": "braak_stage",
            "cerad_score": "cerad",
            "apoe_genotype": "apoe_genotype",
            "brain_region": "brain_region",
            "rin": "rin",
            "pmi_hours": "pmi_hrs",
            "batch": "seq_batch",
        },
        constants={"platform": "Illumina HiSeq 2500"},
        notes=(
            "CERAD reported semantically ('sparse', 'frequent') rather than "
            "numerically. APOE given as two digits without the E prefix."
        ),
    ),
}


def get_mapping(study_id: str) -> SourceMapping:
    """Look up a mapping by accession.

    Raises:
        KeyError: With the list of known studies, so a typo in an ADF pipeline
            parameter produces an actionable message rather than a bare KeyError.
    """
    try:
        return MAPPINGS[study_id]
    except KeyError:
        known = ", ".join(sorted(MAPPINGS))
        raise KeyError(f"No mapping registered for {study_id!r}. Known studies: {known}") from None
