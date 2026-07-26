"""Tests for study-level harmonization and cross-study concatenation."""

from __future__ import annotations

import pandas as pd
import pytest

from adharmon.harmonize import audit_mapping, harmonize, harmonize_many
from adharmon.mappings import SourceMapping, get_mapping
from adharmon.schema import CANONICAL_COLUMNS, CANONICAL_DTYPES


class TestSchemaContract:
    def test_output_has_exact_canonical_columns_in_order(self, gse5281_raw) -> None:
        out = harmonize(gse5281_raw, get_mapping("GSE5281"))
        assert tuple(out.columns) == CANONICAL_COLUMNS

    def test_output_dtypes_match_contract(self, gse5281_raw) -> None:
        out = harmonize(gse5281_raw, get_mapping("GSE5281"))
        for column, expected in CANONICAL_DTYPES.items():
            assert str(out[column].dtype) == expected, column

    def test_unmapped_fields_are_null_not_absent(self, gse5281_raw) -> None:
        """GSE5281 reports no neuropathology; those columns must exist and be null."""
        out = harmonize(gse5281_raw, get_mapping("GSE5281"))
        assert out["braak_stage"].isna().all()
        assert out["cerad_score"].isna().all()

    def test_input_is_not_mutated(self, gse5281_raw) -> None:
        before = gse5281_raw.copy()
        harmonize(gse5281_raw, get_mapping("GSE5281"))
        pd.testing.assert_frame_equal(gse5281_raw, before)

    def test_row_count_preserved(self, gse33000_raw) -> None:
        out = harmonize(gse33000_raw, get_mapping("GSE33000"))
        assert len(out) == len(gse33000_raw)


class TestFieldHarmonization:
    def test_source_study_is_stamped(self, gse33000_raw) -> None:
        out = harmonize(gse33000_raw, get_mapping("GSE33000"))
        assert (out["source_study"] == "GSE33000").all()

    def test_constant_platform_applied(self, gse5281_raw) -> None:
        out = harmonize(gse5281_raw, get_mapping("GSE5281"))
        assert (out["platform"] == "Affymetrix HG-U133 Plus 2.0").all()

    def test_numeric_sex_harmonized(self, gse33000_raw) -> None:
        out = harmonize(gse33000_raw, get_mapping("GSE33000"))
        assert list(out["sex"]) == ["M", "F", "M"]

    def test_roman_braak_harmonized(self, gse33000_raw) -> None:
        out = harmonize(gse33000_raw, get_mapping("GSE33000"))
        assert list(out["braak_stage"]) == [6, 1, 4]

    def test_semantic_cerad_harmonized(self, gse125583_raw) -> None:
        out = harmonize(gse125583_raw, get_mapping("GSE125583"))
        assert list(out["cerad_score"]) == [3, 0, 1]

    def test_age_ceiling_flagged(self, gse33000_raw) -> None:
        out = harmonize(gse33000_raw, get_mapping("GSE33000"))
        assert out.loc[0, "age_years"] == 90.0
        assert bool(out.loc[0, "age_censored"]) is True
        assert bool(out.loc[1, "age_censored"]) is False

    def test_age_above_cap_censored(self, gse125583_raw) -> None:
        """The raw 94 must be capped to 90 and flagged."""
        out = harmonize(gse125583_raw, get_mapping("GSE125583"))
        assert out.loc[1, "age_years"] == 90.0
        assert bool(out.loc[1, "age_censored"]) is True

    def test_apoe_canonicalized(self, gse5281_raw) -> None:
        out = harmonize(gse5281_raw, get_mapping("GSE5281"))
        assert list(out["apoe_genotype"]) == ["E3/E4", "E3/E3", "E4/E4"]

    def test_diagnosis_vocabulary_agrees_across_studies(self, raw_frames, mappings) -> None:
        """The point of the exercise: five spellings of AD become one token.

        The fixtures spell it ``Alzheimer's Disease`` (x2), ``AD`` (x2), and
        ``Alzheimer disease`` (x1); controls arrive as ``normal``,
        ``non-demented``, and ``control``.
        """
        combined = harmonize_many(raw_frames, mappings)
        assert set(combined["diagnosis"]) <= {"AD", "MCI", "CONTROL", "OTHER", "UNKNOWN"}
        counts = combined["diagnosis"].value_counts().to_dict()
        assert counts == {"AD": 5, "CONTROL": 3, "MCI": 1}


class TestAuditMapping:
    def test_complete_mapping(self, gse5281_raw) -> None:
        audit = audit_mapping(gse5281_raw, get_mapping("GSE5281"))
        assert audit.is_complete
        assert audit.missing == ()

    def test_detects_renamed_upstream_column(self, gse5281_raw) -> None:
        """The most common silent pipeline break: a partner renames a column."""
        renamed = gse5281_raw.rename(columns={"age": "age_yrs"})
        audit = audit_mapping(renamed, get_mapping("GSE5281"))
        assert not audit.is_complete
        assert "age" in audit.missing

    def test_reports_unused_source_columns(self, gse5281_raw) -> None:
        extra = gse5281_raw.assign(new_upstream_field=["x", "y", "z"])
        audit = audit_mapping(extra, get_mapping("GSE5281"))
        assert "new_upstream_field" in audit.unmapped

    def test_missing_column_yields_nulls_not_crash(self, gse5281_raw) -> None:
        renamed = gse5281_raw.rename(columns={"age": "age_yrs"})
        out = harmonize(renamed, get_mapping("GSE5281"))
        assert out["age_years"].isna().all()
        assert len(out) == len(renamed)


class TestHarmonizeMany:
    def test_concatenates_all_studies(self, raw_frames, mappings) -> None:
        combined = harmonize_many(raw_frames, mappings)
        assert len(combined) == sum(len(f) for f in raw_frames.values())
        assert set(combined["source_study"]) == set(raw_frames)

    def test_schema_stable_across_concatenation(self, raw_frames, mappings) -> None:
        combined = harmonize_many(raw_frames, mappings)
        assert tuple(combined.columns) == CANONICAL_COLUMNS

    def test_missing_mapping_raises(self, raw_frames, mappings) -> None:
        """Silently skipping an unmapped study is worse than crashing."""
        with pytest.raises(KeyError, match="GSE999999"):
            harmonize_many({**raw_frames, "GSE999999": pd.DataFrame()}, mappings)

    def test_empty_input_returns_typed_empty_frame(self, mappings) -> None:
        out = harmonize_many({}, mappings)
        assert out.empty
        assert tuple(out.columns) == CANONICAL_COLUMNS


class TestGetMapping:
    def test_unknown_study_error_lists_known_studies(self) -> None:
        with pytest.raises(KeyError, match="Known studies"):
            get_mapping("GSE000000")

    def test_source_columns(self) -> None:
        mapping = SourceMapping(study_id="X", field_map={"sample_id": "a", "age_years": "b"})
        assert mapping.source_columns() == {"a", "b"}
