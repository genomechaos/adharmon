"""Harmonize per-study sample metadata onto the canonical schema."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from adharmon import normalizers as nz
from adharmon.mappings import SourceMapping
from adharmon.schema import CANONICAL_COLUMNS, CANONICAL_DTYPES

__all__ = ["MappingAudit", "audit_mapping", "harmonize", "harmonize_many"]

# Canonical field -> normalizer. Fields absent here are passed through
# normalize_text. age_years is special-cased because it returns two values.
_NORMALIZERS = {
    "sex": nz.normalize_sex,
    "diagnosis": nz.normalize_diagnosis,
    "braak_stage": nz.normalize_braak,
    "cerad_score": nz.normalize_cerad,
    "apoe_genotype": nz.normalize_apoe,
    "rin": nz.normalize_float,
    "pmi_hours": nz.normalize_float,
}


@dataclass(frozen=True)
class MappingAudit:
    """Result of checking a mapping against an actual input frame."""

    study_id: str
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    unmapped: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        """``True`` when every column the mapping expects was present."""
        return not self.missing

    def summary(self) -> str:
        """One-line human-readable summary for logs and lineage reports."""
        return (
            f"{self.study_id}: {len(self.matched)} mapped, "
            f"{len(self.missing)} missing {list(self.missing)}, "
            f"{len(self.unmapped)} source columns unused"
        )


def audit_mapping(frame: pd.DataFrame, mapping: SourceMapping) -> MappingAudit:
    """Compare a mapping against a real frame before transforming it.

    An upstream partner renaming or dropping a column is the single most common
    way a working ingestion pipeline breaks, and it is silent: the transform
    still runs, the column is just null from then on. Auditing up front turns
    that into a visible, alertable condition.
    """
    present = set(frame.columns)
    expected = mapping.source_columns()
    matched = expected & present
    return MappingAudit(
        study_id=mapping.study_id,
        matched=tuple(sorted(matched)),
        missing=tuple(sorted(expected - present)),
        unmapped=tuple(sorted(present - expected)),
    )


def harmonize(frame: pd.DataFrame, mapping: SourceMapping) -> pd.DataFrame:
    """Map one study's metadata onto the canonical schema.

    Pure and side-effect free: no I/O, no mutation of ``frame``. All storage
    concerns live in :mod:`adharmon.storage`, which is what makes this the part
    that is cheap to unit test.

    Args:
        frame: Raw source metadata, one row per sample.
        mapping: Crosswalk for this study.

    Returns:
        A frame with exactly :data:`~adharmon.schema.CANONICAL_COLUMNS`, in
        order, with canonical dtypes. Unmapped canonical fields are all-null.
    """
    out = pd.DataFrame(index=frame.index)

    for canonical_field in CANONICAL_COLUMNS:
        if canonical_field in ("source_study", "age_censored"):
            continue  # handled below

        if canonical_field in mapping.constants:
            out[canonical_field] = mapping.constants[canonical_field]
            continue

        source_column = mapping.field_map.get(canonical_field)
        if source_column is None or source_column not in frame.columns:
            out[canonical_field] = None
            continue

        raw = frame[source_column]

        if canonical_field == "age_years":
            parsed = raw.map(nz.normalize_age)
            out["age_years"] = [value for value, _ in parsed]
            out["age_censored"] = [censored for _, censored in parsed]
            continue

        normalizer = _NORMALIZERS.get(canonical_field, nz.normalize_text)
        out[canonical_field] = raw.map(normalizer)

    out["source_study"] = mapping.study_id
    if "age_censored" not in out.columns:
        out["age_censored"] = False
    out["age_censored"] = out["age_censored"].fillna(False)

    out = out.reindex(columns=list(CANONICAL_COLUMNS))
    return out.astype(CANONICAL_DTYPES)


def harmonize_many(frames: dict[str, pd.DataFrame], mappings: dict[str, SourceMapping]) -> pd.DataFrame:
    """Harmonize and concatenate several studies into one silver-layer frame.

    Args:
        frames: Study accession -> raw metadata frame.
        mappings: Study accession -> crosswalk. Must cover every key in ``frames``.

    Returns:
        Concatenated canonical frame with a reset index.

    Raises:
        KeyError: If a frame has no corresponding mapping. Failing here rather
            than skipping the study prevents a silent partial harmonization,
            which is far more expensive to detect downstream than a crash.
    """
    missing_mappings = sorted(set(frames) - set(mappings))
    if missing_mappings:
        raise KeyError(f"No mapping supplied for: {missing_mappings}")

    if not frames:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS)).astype(CANONICAL_DTYPES)

    harmonized = [harmonize(frame, mappings[study]) for study, frame in sorted(frames.items())]
    return pd.concat(harmonized, ignore_index=True)
