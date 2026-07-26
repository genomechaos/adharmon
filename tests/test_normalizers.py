"""Unit tests for field-level normalizers.

These are the tests worth having. Every case below is a real formatting
convention observed in public AD metadata, so the suite doubles as executable
documentation of what the upstream data actually looks like.
"""

from __future__ import annotations

import math

import pytest

from adharmon.normalizers import (
    is_blank,
    normalize_age,
    normalize_apoe,
    normalize_braak,
    normalize_cerad,
    normalize_diagnosis,
    normalize_float,
    normalize_sex,
    normalize_text,
)
from adharmon.schema import HIPAA_AGE_CAP


class TestIsBlank:
    @pytest.mark.parametrize(
        "value",
        [None, float("nan"), "", "   ", "NA", "n/a", "NaN", "none", "NULL", "-", "not reported", "unknown"],
    )
    def test_recognizes_missing_sentinels(self, value: object) -> None:
        assert is_blank(value) is True

    @pytest.mark.parametrize("value", ["0", 0, "AD", "control", 3.5, "III"])
    def test_keeps_real_values(self, value: object) -> None:
        assert is_blank(value) is False

    def test_zero_is_not_blank(self) -> None:
        """Braak 0 and CERAD 0 are meaningful scores, not missing data."""
        assert is_blank(0) is False


class TestNormalizeSex:
    @pytest.mark.parametrize("value", ["M", "m", "male", "Male", "MALE", "1", "man"])
    def test_male_variants(self, value: str) -> None:
        assert normalize_sex(value) == "M"

    @pytest.mark.parametrize("value", ["F", "f", "female", "Female", "2", "woman"])
    def test_female_variants(self, value: str) -> None:
        assert normalize_sex(value) == "F"

    @pytest.mark.parametrize("value", [None, "", "NA", "other", "intersex", "9"])
    def test_unknown_is_never_null(self, value: object) -> None:
        assert normalize_sex(value) == "unknown"


class TestNormalizeDiagnosis:
    @pytest.mark.parametrize(
        "value",
        ["AD", "ad", "Alzheimer's disease", "Alzheimer Disease", "ALZHEIMERS", "probable AD", "LOAD"],
    )
    def test_ad_variants(self, value: str) -> None:
        assert normalize_diagnosis(value) == "AD"

    @pytest.mark.parametrize("value", ["MCI", "mci", "mild cognitive impairment", "Mild Cognitive Impairment"])
    def test_mci_variants(self, value: str) -> None:
        assert normalize_diagnosis(value) == "MCI"

    @pytest.mark.parametrize(
        "value",
        ["control", "Control", "CTRL", "normal", "non-demented", "cognitively normal", "healthy control", "NC"],
    )
    def test_control_variants(self, value: str) -> None:
        assert normalize_diagnosis(value) == "CONTROL"

    @pytest.mark.parametrize(
        "value",
        [
            "frontotemporal dementia",
            "Lewy body dementia",
            "vascular dementia",
            "PSP progressive supranuclear palsy",
        ],
    )
    def test_other_dementias_are_not_folded_into_ad(self, value: str) -> None:
        """A non-AD dementia must never be silently grouped as AD."""
        assert normalize_diagnosis(value) == "OTHER"

    def test_missing_is_unknown_not_other(self) -> None:
        assert normalize_diagnosis(None) == "UNKNOWN"
        assert normalize_diagnosis("NA") == "UNKNOWN"

    def test_unparseable_is_other_not_ad(self) -> None:
        assert normalize_diagnosis("subject withdrew") == "OTHER"

    def test_short_token_not_substring_matched(self) -> None:
        """'ad' must match exactly -- never as a substring of another word."""
        assert normalize_diagnosis("advanced pathology, no dementia") == "CONTROL"


class TestNormalizeAge:
    def test_plain_integer(self) -> None:
        assert normalize_age("78") == (78.0, False)

    def test_decimal(self) -> None:
        assert normalize_age("78.5") == (78.5, False)

    @pytest.mark.parametrize("value", ["78 years", "78 yrs", "78 y/o", "78yo"])
    def test_strips_units(self, value: str) -> None:
        age, censored = normalize_age(value)
        assert age == 78.0
        assert censored is False

    @pytest.mark.parametrize("value", ["90+", "90 or older", "90 or above", ">89", ">90", "89+"])
    def test_safe_harbor_ceiling_is_flagged(self, value: str) -> None:
        assert normalize_age(value) == (HIPAA_AGE_CAP, True)

    def test_true_age_above_cap_is_censored_by_us(self) -> None:
        """A study reporting 94 must be capped before reaching a shared layer."""
        assert normalize_age("94") == (HIPAA_AGE_CAP, True)

    def test_exactly_ninety_is_not_censored(self) -> None:
        age, censored = normalize_age("90")
        assert age == 90.0
        assert censored is False

    @pytest.mark.parametrize("value", [None, "", "NA", "unknown", "adult"])
    def test_unparseable(self, value: object) -> None:
        assert normalize_age(value) == (None, False)


class TestNormalizeBraak:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("0", 0),
            ("I", 1),
            ("ii", 2),
            ("III", 3),
            ("IV", 4),
            ("V", 5),
            ("VI", 6),
            ("3", 3),
            ("6", 6),
            ("Braak III", 3),
            ("braak stage IV", 4),
            ("stage 5", 5),
            ("Braak NFT stage II", 2),
            ("B2", 2),
        ],
    )
    def test_accepted_forms(self, value: str, expected: int) -> None:
        assert normalize_braak(value) == expected

    @pytest.mark.parametrize("value", ["III-IV", "3-4", "III/IV", "iii to iv", "III or IV"])
    def test_range_takes_lower_bound(self, value: str) -> None:
        assert normalize_braak(value) == 3

    @pytest.mark.parametrize("value", [None, "", "NA", "not assessed", "n/a"])
    def test_missing(self, value: object) -> None:
        assert normalize_braak(value) is None

    def test_zero_is_preserved(self) -> None:
        """Braak 0 means 'no tangles', which is a finding, not missing data."""
        assert normalize_braak("0") == 0


class TestNormalizeCerad:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("0", 0), ("1", 1), ("2", 2), ("3", 3), ("none", 0), ("sparse", 1), ("moderate", 2), ("frequent", 3)],
    )
    def test_numeric_and_semantic_forms_agree(self, value: str, expected: int) -> None:
        assert normalize_cerad(value) == expected

    def test_none_is_score_zero_not_missing(self) -> None:
        """Regression: 'none' is both a valid CERAD score and a null sentinel.

        Reading it as missing would silently erase every plaque-free control --
        exactly the group a case/control comparison leans on hardest.
        """
        assert normalize_cerad("none") == 0
        assert normalize_cerad("CERAD none") == 0

    @pytest.mark.parametrize("value", ["NA", "n/a", "not assessed", "not reported", "", None])
    def test_missing(self, value: object) -> None:
        assert normalize_cerad(value) is None


class TestNormalizeApoe:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("33", "E3/E3"),
            ("3/3", "E3/E3"),
            ("E3/E3", "E3/E3"),
            ("e3e3", "E3/E3"),
            ("34", "E3/E4"),
            ("4/3", "E3/E4"),
            ("E4/E3", "E3/E4"),
            ("apoe 3/4", "E3/E4"),
            ("23", "E2/E3"),
            ("44", "E4/E4"),
        ],
    )
    def test_sorted_canonical_form(self, value: str, expected: str) -> None:
        assert normalize_apoe(value) == expected

    def test_allele_order_collapses(self) -> None:
        """'34' and '43' are one genotype; splitting them halves an effect size."""
        assert normalize_apoe("34") == normalize_apoe("43")

    @pytest.mark.parametrize("value", [None, "NA", "E4 carrier", "unknown", "3"])
    def test_unparseable(self, value: object) -> None:
        assert normalize_apoe(value) is None


class TestNormalizeFloat:
    @pytest.mark.parametrize(("value", "expected"), [("7.2", 7.2), ("7", 7.0), ("4.5 hours", 4.5), ("12 hrs", 12.0)])
    def test_parses_measurements(self, value: str, expected: float) -> None:
        assert normalize_float(value) == expected

    def test_missing(self) -> None:
        assert normalize_float("NA") is None

    def test_nan_input(self) -> None:
        assert normalize_float(math.nan) is None


class TestNormalizeText:
    def test_collapses_whitespace(self) -> None:
        assert normalize_text("  entorhinal   cortex \n") == "entorhinal cortex"

    def test_missing(self) -> None:
        assert normalize_text("  ") is None
