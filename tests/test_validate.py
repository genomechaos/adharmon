"""Tests for quality validation, quarantine, and flagging."""

from __future__ import annotations

import pandas as pd
import pytest

from adharmon.harmonize import harmonize_many
from adharmon.schema import CANONICAL_COLUMNS, CANONICAL_DTYPES
from adharmon.validate import RULES, Severity, validate


def make_frame(**overrides) -> pd.DataFrame:
    """Build a single-row canonical frame, overriding selected fields."""
    row = dict.fromkeys(CANONICAL_COLUMNS)
    row.update(
        {
            "sample_id": "S1",
            "source_study": "GSE1",
            "diagnosis": "AD",
            "age_censored": False,
        }
    )
    row.update(overrides)
    return pd.DataFrame([row]).astype(CANONICAL_DTYPES)


class TestRejectRules:
    @pytest.mark.parametrize("column", ["sample_id", "source_study", "diagnosis"])
    def test_missing_required_field_quarantines(self, column: str) -> None:
        report = validate(make_frame(**{column: None}))
        assert len(report.clean) == 0
        assert len(report.quarantined) == 1
        assert f"missing_{column}" in report.quarantined.loc[0, "quarantine_reason"]

    def test_empty_string_counts_as_missing(self) -> None:
        report = validate(make_frame(sample_id="   "))
        assert len(report.quarantined) == 1

    def test_duplicate_sample_id_quarantines_all_copies(self) -> None:
        frame = pd.concat([make_frame(sample_id="DUP"), make_frame(sample_id="DUP")], ignore_index=True)
        report = validate(frame)
        assert len(report.quarantined) == 2, "keeping an arbitrary winner would depend on row order"
        assert len(report.clean) == 0

    def test_distinct_ids_pass(self) -> None:
        frame = pd.concat([make_frame(sample_id="A"), make_frame(sample_id="B")], ignore_index=True)
        assert len(validate(frame).clean) == 2

    @pytest.mark.parametrize(
        ("column", "bad"),
        [
            ("braak_stage", 7),
            ("cerad_score", 5),
            ("rin", 11.0),
            ("pmi_hours", 500.0),  # hours recorded as minutes
            ("age_years", 200.0),
        ],
    )
    def test_out_of_range_quarantines(self, column: str, bad: float) -> None:
        report = validate(make_frame(**{column: bad}))
        assert len(report.quarantined) == 1

    @pytest.mark.parametrize(
        ("column", "ok"),
        [("braak_stage", 6), ("cerad_score", 3), ("rin", 10.0), ("pmi_hours", 0.0), ("age_years", 0.0)],
    )
    def test_range_bounds_are_inclusive(self, column: str, ok: float) -> None:
        assert len(validate(make_frame(**{column: ok})).clean) == 1

    def test_unrecognized_diagnosis_quarantines(self) -> None:
        frame = make_frame()
        frame["diagnosis"] = pd.Series(["not-a-real-dx"], dtype="string")
        report = validate(frame)
        assert "diagnosis_not_recognized" in report.quarantined.loc[0, "quarantine_reason"]

    def test_multiple_reasons_are_joined(self) -> None:
        report = validate(make_frame(sample_id=None, braak_stage=9))
        reason = report.quarantined.loc[0, "quarantine_reason"]
        assert "missing_sample_id" in reason
        assert "braak_out_of_range" in reason
        assert ";" in reason

    def test_nulls_in_optional_fields_do_not_quarantine(self) -> None:
        """Absent optional data is normal, not a quality failure."""
        assert len(validate(make_frame(braak_stage=None, rin=None, pmi_hours=None)).clean) == 1


class TestFlagRules:
    def test_ad_without_pathology_is_flagged_not_rejected(self) -> None:
        """Clinical/neuropath discordance is real signal -- keep it, annotate it."""
        report = validate(make_frame(diagnosis="AD", braak_stage=0))
        assert len(report.clean) == 1
        assert "ad_without_pathology" in report.clean.loc[0, "quality_flags"]

    def test_control_with_advanced_pathology_flagged(self) -> None:
        report = validate(make_frame(diagnosis="CONTROL", braak_stage=6))
        assert len(report.clean) == 1
        assert "control_with_pathology" in report.clean.loc[0, "quality_flags"]

    def test_degraded_rna_flagged(self) -> None:
        report = validate(make_frame(rin=3.2))
        assert len(report.clean) == 1
        assert "degraded_rna" in report.clean.loc[0, "quality_flags"]

    def test_good_rin_not_flagged(self) -> None:
        report = validate(make_frame(rin=8.0))
        assert "degraded_rna" not in report.clean.loc[0, "quality_flags"]

    def test_implausibly_young_flagged(self) -> None:
        report = validate(make_frame(age_years=12.0))
        assert "implausibly_young" in report.clean.loc[0, "quality_flags"]

    def test_missing_age_flagged_but_kept(self) -> None:
        report = validate(make_frame(age_years=None))
        assert len(report.clean) == 1
        assert "missing_age" in report.clean.loc[0, "quality_flags"]

    def test_clean_record_has_empty_flags(self) -> None:
        report = validate(make_frame(age_years=80.0, braak_stage=5, rin=8.0))
        assert report.clean.loc[0, "quality_flags"] == ""


class TestReport:
    def test_counts_and_pass_rate(self) -> None:
        frame = pd.concat([make_frame(sample_id="A", age_years=80.0), make_frame(sample_id=None)], ignore_index=True)
        report = validate(frame)
        assert report.total == 2
        assert len(report.clean) == 1
        assert report.pass_rate == 0.5
        assert report.rule_counts["missing_sample_id"] == 1

    def test_summary_mentions_triggered_rules(self) -> None:
        report = validate(make_frame(sample_id=None))
        assert "missing_sample_id" in report.summary()

    def test_empty_input(self) -> None:
        empty = pd.DataFrame(columns=list(CANONICAL_COLUMNS)).astype(CANONICAL_DTYPES)
        report = validate(empty)
        assert report.total == 0
        assert report.pass_rate == 1.0

    def test_input_not_mutated(self) -> None:
        frame = make_frame()
        before = frame.copy()
        validate(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_clean_frame_keeps_canonical_columns(self) -> None:
        report = validate(make_frame(age_years=80.0))
        assert list(report.clean.columns) == [*CANONICAL_COLUMNS, "quality_flags"]


class TestRuleRegistry:
    def test_rule_names_unique(self) -> None:
        names = [rule.name for rule in RULES]
        assert len(names) == len(set(names))

    def test_every_rule_documented(self) -> None:
        assert all(rule.description for rule in RULES)

    def test_both_severities_present(self) -> None:
        severities = {rule.severity for rule in RULES}
        assert severities == {Severity.REJECT, Severity.FLAG}

    def test_predicates_tolerate_all_null_frame(self) -> None:
        """A rule must never raise on missing columns or null values."""
        sparse = pd.DataFrame([dict.fromkeys(CANONICAL_COLUMNS)]).astype(CANONICAL_DTYPES)
        for rule in RULES:
            result = rule.predicate(sparse)
            assert len(result) == 1, rule.name


class TestEndToEndOnFixtures:
    def test_realistic_batch_mostly_passes(self, raw_frames, mappings) -> None:
        report = validate(harmonize_many(raw_frames, mappings))
        assert report.total == 9
        assert len(report.quarantined) == 0, report.quarantined.to_dict("records")

    def test_degraded_sample_is_flagged_in_realistic_batch(self, raw_frames, mappings) -> None:
        """GSE125583 TCX_03 has RIN 4.1 -- below threshold, kept with a flag."""
        report = validate(harmonize_many(raw_frames, mappings))
        flagged = report.clean[report.clean["quality_flags"].str.contains("degraded_rna", na=False)]
        assert list(flagged["sample_id"]) == ["TCX_03"]
