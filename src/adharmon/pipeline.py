"""Pipeline orchestration: bronze -> harmonize -> validate -> silver/quarantine.

This is the unit an Azure Data Factory activity invokes. It is intentionally
thin -- all the logic worth testing lives in :mod:`adharmon.harmonize` and
:mod:`adharmon.validate`, and everything here is wiring plus the lineage record.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import pandas as pd

from adharmon.harmonize import audit_mapping, harmonize
from adharmon.mappings import MAPPINGS, SourceMapping, get_mapping
from adharmon.storage import LakeStore, Layer
from adharmon.validate import validate

__all__ = ["RunManifest", "run_study", "run_all"]

logger = logging.getLogger(__name__)

#: ``datetime.UTC`` only exists on 3.11+; this keeps the 3.10 floor.
UTC = timezone.utc


@dataclass
class RunManifest:
    """Lineage record for one pipeline run.

    Written next to the output as JSON. This is the artifact that answers "where
    did this row come from and what was done to it" months later, and it is the
    difference between a reproducible dataset and a mystery Parquet file.
    """

    run_id: str
    started_at: str
    studies: list[str] = field(default_factory=list)
    source_paths: dict[str, str] = field(default_factory=dict)
    mapping_audit: dict[str, str] = field(default_factory=dict)
    mapping_notes: dict[str, str] = field(default_factory=dict)
    rule_counts: dict[str, int] = field(default_factory=dict)
    rows_in: int = 0
    rows_clean: int = 0
    rows_quarantined: int = 0
    silver_path: str | None = None
    quarantine_path: str | None = None
    finished_at: str | None = None

    def to_json(self) -> str:
        """Serialize the manifest deterministically for storage and diffing."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_study(
    store: LakeStore,
    study_id: str,
    source_path: str,
    mapping: SourceMapping | None = None,
) -> tuple[pd.DataFrame, str]:
    """Read and harmonize one study from the bronze layer.

    Args:
        store: Lake backend.
        study_id: Accession, used to look up the mapping when not supplied.
        source_path: Path within the bronze container.
        mapping: Optional explicit crosswalk, overriding the registry.

    Returns:
        ``(harmonized_frame, audit_summary)``.
    """
    mapping = mapping or get_mapping(study_id)
    raw = store.read_csv(Layer.BRONZE, source_path)

    audit = audit_mapping(raw, mapping)
    if not audit.is_complete:
        # Warn rather than raise: a study missing an optional column should
        # still load. Genuinely required fields are enforced in validate().
        logger.warning("Incomplete mapping -- %s", audit.summary())
    else:
        logger.info("Mapping complete -- %s", audit.summary())

    return harmonize(raw, mapping), audit.summary()


def run_all(
    store: LakeStore,
    sources: dict[str, str],
    run_id: str | None = None,
) -> RunManifest:
    """Run the full pipeline across several studies.

    Args:
        store: Lake backend.
        sources: Study accession -> path within the bronze container.
        run_id: Optional identifier; defaults to a UTC timestamp.

    Returns:
        The completed :class:`RunManifest`, also written to the silver layer.
    """
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest = RunManifest(run_id=run_id, started_at=_now(), source_paths=dict(sources))

    harmonized: list[pd.DataFrame] = []
    for study_id, source_path in sorted(sources.items()):
        frame, audit_summary = run_study(store, study_id, source_path)
        harmonized.append(frame)
        manifest.studies.append(study_id)
        manifest.mapping_audit[study_id] = audit_summary
        manifest.mapping_notes[study_id] = MAPPINGS[study_id].notes if study_id in MAPPINGS else ""

    combined = pd.concat(harmonized, ignore_index=True) if harmonized else pd.DataFrame()
    manifest.rows_in = len(combined)

    report = validate(combined)
    manifest.rule_counts = report.rule_counts
    manifest.rows_clean = len(report.clean)
    manifest.rows_quarantined = len(report.quarantined)
    logger.info("Validation -- %s", report.summary())

    manifest.silver_path = store.write_parquet(report.clean, Layer.SILVER, f"samples/run_id={run_id}/samples.parquet")
    if not report.quarantined.empty:
        manifest.quarantine_path = store.write_parquet(
            report.quarantined, Layer.QUARANTINE, f"samples/run_id={run_id}/rejected.parquet"
        )

    manifest.finished_at = _now()
    store.write_parquet(
        pd.DataFrame([{"run_id": run_id, "manifest_json": manifest.to_json()}]),
        Layer.SILVER,
        f"_manifests/run_id={run_id}/manifest.parquet",
    )
    return manifest
