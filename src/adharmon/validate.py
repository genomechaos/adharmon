"""Quality validation and quarantine for the harmonized silver layer.

Two severities, deliberately separated:

``REJECT``
    The record violates the schema contract. It is diverted to a quarantine
    path with a machine-readable reason rather than dropped, so an upstream
    partner can be sent a concrete list of what failed and why.

``FLAG``
    The record is schema-valid but biologically or statistically suspicious --
    a clinical diagnosis that disagrees with its neuropathology, RNA too
    degraded to trust, an age that implies a unit error. These stay in the
    silver layer with a flag column, because dropping them is a scientific
    judgement that belongs to the analyst, not to the pipeline.

That split is the whole point of the module. Rejecting on suspicion loses real
signal; passing everything through silently ships artifacts into model training.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from adharmon.schema import (
    AGE_MAX,
    AGE_MIN,
    BRAAK_MAX,
    BRAAK_MIN,
    CERAD_MAX,
    CERAD_MIN,
    PMI_MAX,
    PMI_MIN,
    REQUIRED_COLUMNS,
    RIN_MAX,
    RIN_MIN,
    Diagnosis,
)

__all__ = ["Severity", "Rule", "ValidationReport", "RULES", "validate"]

#: RIN below this is widely treated as too degraded for reliable expression
#: quantification. Not a hard error -- some cohorts have no better material.
RIN_DEGRADED_THRESHOLD = 5.0

#: Below this age, an AD brain cohort entry is more likely a unit or data-entry
#: error than a genuine early-onset case.
IMPLAUSIBLY_YOUNG = 40.0


class Severity(str, Enum):
    """What to do with a record that fails a rule."""

    REJECT = "reject"
    FLAG = "flag"


@dataclass(frozen=True)
class Rule:
    """A single named validation rule.

    Attributes:
        name: Stable identifier, used as the quarantine reason / flag label.
        severity: Whether a failure quarantines or merely annotates the record.
        description: Human-readable rationale, surfaced in the data dictionary.
        predicate: Takes the frame, returns a boolean Series that is ``True``
            for *failing* rows. Must not raise on null input.
    """

    name: str
    severity: Severity
    description: str
    predicate: Callable[[pd.DataFrame], pd.Series]


def _falsy(series: pd.Series, index: pd.Index) -> pd.Series:
    """Coerce a possibly-NA boolean Series to a plain bool mask.

    Comparisons against pandas nullable dtypes propagate NA, which cannot be
    used as a mask. Nulls are handled by explicit ``.isna()`` rules, so any NA
    left in a comparison result means "not a failure" here.
    """
    return series.reindex(index).fillna(False).astype(bool)


def _out_of_range(column: str, low: float, high: float) -> Callable[[pd.DataFrame], pd.Series]:
    """Build a predicate flagging non-null values outside ``[low, high]``."""

    def predicate(frame: pd.DataFrame) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(False, index=frame.index)
        values = frame[column]
        return _falsy((values < low) | (values > high), frame.index)

    return predicate


def _missing_required(column: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Build a predicate flagging null or empty values in a required column."""

    def predicate(frame: pd.DataFrame) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(True, index=frame.index)
        values = frame[column]
        blank = values.isna()
        if values.dtype == "string":
            blank = blank | (values.fillna("").str.strip() == "")
        return _falsy(blank, frame.index)

    return predicate


def _duplicate_sample_id(frame: pd.DataFrame) -> pd.Series:
    """Flag every member of a duplicated ``sample_id`` group.

    All copies are quarantined, not just the later ones. Keeping an arbitrary
    winner would make the pipeline's output depend on input row order, which is
    exactly the kind of irreproducibility that is impossible to debug later.
    """
    if "sample_id" not in frame.columns:
        return pd.Series(False, index=frame.index)
    ids = frame["sample_id"]
    return _falsy(ids.duplicated(keep=False) & ids.notna(), frame.index)


def _diagnosis_not_recognized(frame: pd.DataFrame) -> pd.Series:
    valid = {member.value for member in Diagnosis}
    if "diagnosis" not in frame.columns:
        return pd.Series(True, index=frame.index)
    values = frame["diagnosis"]
    return _falsy(values.notna() & ~values.isin(valid), frame.index)


def _ad_without_pathology(frame: pd.DataFrame) -> pd.Series:
    """AD diagnosis with Braak 0-I: clinical and neuropathological disagreement."""
    if not {"diagnosis", "braak_stage"} <= set(frame.columns):
        return pd.Series(False, index=frame.index)
    return _falsy((frame["diagnosis"] == Diagnosis.AD.value) & (frame["braak_stage"] <= 1), frame.index)


def _control_with_pathology(frame: pd.DataFrame) -> pd.Series:
    """Control with Braak V-VI: advanced pathology in a nominal control."""
    if not {"diagnosis", "braak_stage"} <= set(frame.columns):
        return pd.Series(False, index=frame.index)
    return _falsy((frame["diagnosis"] == Diagnosis.CONTROL.value) & (frame["braak_stage"] >= 5), frame.index)


def _degraded_rna(frame: pd.DataFrame) -> pd.Series:
    if "rin" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return _falsy(frame["rin"] < RIN_DEGRADED_THRESHOLD, frame.index)


def _implausibly_young(frame: pd.DataFrame) -> pd.Series:
    if "age_years" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return _falsy(frame["age_years"] < IMPLAUSIBLY_YOUNG, frame.index)


def _missing_age(frame: pd.DataFrame) -> pd.Series:
    if "age_years" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return _falsy(frame["age_years"].isna(), frame.index)


RULES: tuple[Rule, ...] = (
    *(
        Rule(
            name=f"missing_{column}",
            severity=Severity.REJECT,
            description=f"Required field {column!r} is null or empty.",
            predicate=_missing_required(column),
        )
        for column in REQUIRED_COLUMNS
    ),
    Rule(
        name="duplicate_sample_id",
        severity=Severity.REJECT,
        description="sample_id appears more than once in this batch.",
        predicate=_duplicate_sample_id,
    ),
    Rule(
        name="diagnosis_not_recognized",
        severity=Severity.REJECT,
        description="diagnosis is outside the controlled vocabulary.",
        predicate=_diagnosis_not_recognized,
    ),
    Rule(
        name="age_out_of_range",
        severity=Severity.REJECT,
        description=f"age_years outside [{AGE_MIN}, {AGE_MAX}].",
        predicate=_out_of_range("age_years", AGE_MIN, AGE_MAX),
    ),
    Rule(
        name="braak_out_of_range",
        severity=Severity.REJECT,
        description=f"braak_stage outside [{BRAAK_MIN}, {BRAAK_MAX}].",
        predicate=_out_of_range("braak_stage", BRAAK_MIN, BRAAK_MAX),
    ),
    Rule(
        name="cerad_out_of_range",
        severity=Severity.REJECT,
        description=f"cerad_score outside [{CERAD_MIN}, {CERAD_MAX}].",
        predicate=_out_of_range("cerad_score", CERAD_MIN, CERAD_MAX),
    ),
    Rule(
        name="rin_out_of_range",
        severity=Severity.REJECT,
        description=f"rin outside [{RIN_MIN}, {RIN_MAX}].",
        predicate=_out_of_range("rin", RIN_MIN, RIN_MAX),
    ),
    Rule(
        name="pmi_out_of_range",
        severity=Severity.REJECT,
        description=f"pmi_hours outside [{PMI_MIN}, {PMI_MAX}] -- usually a unit error.",
        predicate=_out_of_range("pmi_hours", PMI_MIN, PMI_MAX),
    ),
    Rule(
        name="ad_without_pathology",
        severity=Severity.FLAG,
        description="AD diagnosis with Braak stage 0-I; clinical/neuropath discordance.",
        predicate=_ad_without_pathology,
    ),
    Rule(
        name="control_with_pathology",
        severity=Severity.FLAG,
        description="Control with Braak stage V-VI; likely preclinical AD.",
        predicate=_control_with_pathology,
    ),
    Rule(
        name="degraded_rna",
        severity=Severity.FLAG,
        description=f"RIN below {RIN_DEGRADED_THRESHOLD}; expression estimates unreliable.",
        predicate=_degraded_rna,
    ),
    Rule(
        name="implausibly_young",
        severity=Severity.FLAG,
        description=f"age_years below {IMPLAUSIBLY_YOUNG}; check for a unit error.",
        predicate=_implausibly_young,
    ),
    Rule(
        name="missing_age",
        severity=Severity.FLAG,
        description="age_years is null; record is usable but not age-adjustable.",
        predicate=_missing_age,
    ),
)


@dataclass
class ValidationReport:
    """Outcome of validating a harmonized frame.

    Attributes:
        clean: Records that passed every REJECT rule, with a ``quality_flags``
            column holding a semicolon-joined list of any FLAG rules hit.
        quarantined: Failing records, with a ``quarantine_reason`` column.
        rule_counts: Rule name -> number of records that failed it.
    """

    clean: pd.DataFrame
    quarantined: pd.DataFrame
    rule_counts: dict[str, int]

    @property
    def total(self) -> int:
        """Records seen, clean plus quarantined."""
        return len(self.clean) + len(self.quarantined)

    @property
    def pass_rate(self) -> float:
        """Fraction of records reaching the silver layer, 1.0 for empty input."""
        return 1.0 if self.total == 0 else len(self.clean) / self.total

    def summary(self) -> str:
        """One-line summary suitable for a pipeline log or an alert body."""
        triggered = {name: count for name, count in self.rule_counts.items() if count}
        return (
            f"{len(self.clean)}/{self.total} records passed "
            f"({self.pass_rate:.1%}); {len(self.quarantined)} quarantined; "
            f"rules triggered: {triggered or 'none'}"
        )


def _join_reasons(masks: dict[str, pd.Series], index: pd.Index) -> pd.Series:
    """Collapse per-rule failure masks into one ``;``-joined reason per row."""
    frame = pd.DataFrame(masks, index=index)
    if frame.empty or not masks:
        return pd.Series([""] * len(index), index=index, dtype="string")
    return frame.apply(lambda row: ";".join(sorted(frame.columns[row.to_numpy(dtype=bool)])), axis=1).astype("string")


def validate(frame: pd.DataFrame, rules: tuple[Rule, ...] = RULES) -> ValidationReport:
    """Split a harmonized frame into clean and quarantined records.

    Args:
        frame: Canonical-schema frame, typically from
            :func:`adharmon.harmonize.harmonize_many`.
        rules: Rules to apply. Defaults to :data:`RULES`; pass a subset to
            relax validation for an exploratory load.

    Returns:
        A :class:`ValidationReport`. Input is never mutated.
    """
    if frame.empty:
        empty_clean = frame.copy()
        empty_clean["quality_flags"] = pd.Series(dtype="string")
        empty_quarantine = frame.copy()
        empty_quarantine["quarantine_reason"] = pd.Series(dtype="string")
        return ValidationReport(empty_clean, empty_quarantine, {rule.name: 0 for rule in rules})

    reject_masks: dict[str, pd.Series] = {}
    flag_masks: dict[str, pd.Series] = {}
    counts: dict[str, int] = {}

    for rule in rules:
        mask = _falsy(rule.predicate(frame), frame.index)
        counts[rule.name] = int(mask.sum())
        if rule.severity is Severity.REJECT:
            reject_masks[rule.name] = mask
        else:
            flag_masks[rule.name] = mask

    any_reject = (
        pd.DataFrame(reject_masks, index=frame.index).any(axis=1)
        if reject_masks
        else pd.Series(False, index=frame.index)
    )

    quarantined = frame.loc[any_reject].copy()
    quarantined["quarantine_reason"] = _join_reasons(reject_masks, frame.index).loc[any_reject]

    clean = frame.loc[~any_reject].copy()
    clean["quality_flags"] = _join_reasons(flag_masks, frame.index).loc[~any_reject]

    return ValidationReport(
        clean=clean.reset_index(drop=True),
        quarantined=quarantined.reset_index(drop=True),
        rule_counts=counts,
    )
