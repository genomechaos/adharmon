"""Field-level normalizers for AD sample metadata.

Each public function takes one raw cell value of unknown type and returns a
value conforming to the canonical schema, or ``None`` when the input carries no
usable information. Normalizers never raise on bad input -- an unparseable
value becomes ``None`` and is caught downstream by :mod:`adharmon.validate`,
which records *why* the record was rejected. Failing loudly at the validation
boundary rather than mid-transform is what keeps a partial upstream delivery
from taking down the whole pipeline run.
"""

from __future__ import annotations

import math
import re

from adharmon.schema import HIPAA_AGE_CAP, Diagnosis, Sex

__all__ = [
    "is_blank",
    "normalize_age",
    "normalize_apoe",
    "normalize_braak",
    "normalize_cerad",
    "normalize_diagnosis",
    "normalize_float",
    "normalize_sex",
    "normalize_text",
]

_BLANK_TOKENS = {
    "",
    "-",
    "--",
    ".",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "missing",
    "unknown",
    "unk",
    "not reported",
    "not available",
    "not applicable",
    "no data",
}

_ROMAN_NUMERALS = {"0": 0, "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}

_SEX_MALE = {"m", "male", "1", "man", "xy"}
_SEX_FEMALE = {"f", "female", "2", "woman", "xx"}

# Exact-match diagnosis lookups. Short, ambiguous tokens ("ad", "nc") are only
# ever matched exactly -- substring matching on "ad" would catch "adjacent",
# "advanced", and half the free-text in a clinical annotation column.
_DIAGNOSIS_EXACT = {
    "ad": Diagnosis.AD,
    "load": Diagnosis.AD,
    "eoad": Diagnosis.AD,
    "mci": Diagnosis.MCI,
    "ci": Diagnosis.MCI,
    "control": Diagnosis.CONTROL,
    "controls": Diagnosis.CONTROL,
    "ctrl": Diagnosis.CONTROL,
    "ctl": Diagnosis.CONTROL,
    "nc": Diagnosis.CONTROL,
    "hc": Diagnosis.CONTROL,
    "normal": Diagnosis.CONTROL,
    "healthy": Diagnosis.CONTROL,
    "unaffected": Diagnosis.CONTROL,
}

# Ordered substring rules, applied only after exact matching fails. Order
# matters: "mild cognitive impairment" must be tested before "impair" style
# fallbacks, and the dementia catch-all must come last so that more specific
# non-AD dementias are not silently folded into AD.
_DIAGNOSIS_SUBSTRINGS: tuple[tuple[str, Diagnosis], ...] = (
    ("mild cognitive impairment", Diagnosis.MCI),
    ("cognitive impairment", Diagnosis.MCI),
    ("alzheimer", Diagnosis.AD),
    ("non-demented", Diagnosis.CONTROL),
    ("nondemented", Diagnosis.CONTROL),
    ("no dementia", Diagnosis.CONTROL),
    ("cognitively normal", Diagnosis.CONTROL),
    ("cognitively unimpaired", Diagnosis.CONTROL),
    ("healthy control", Diagnosis.CONTROL),
    ("frontotemporal", Diagnosis.OTHER),
    ("lewy", Diagnosis.OTHER),
    ("vascular dementia", Diagnosis.OTHER),
    ("parkinson", Diagnosis.OTHER),
    ("progressive supranuclear", Diagnosis.OTHER),
    ("dementia", Diagnosis.OTHER),
)


def is_blank(value: object) -> bool:
    """Return ``True`` for values that carry no information.

    Covers ``None``, float NaN, pandas NA, empty strings, and the long tail of
    sentinel strings that upstream studies use to mean "missing".
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    # pandas.NA and numpy masked values are not None and not float NaN.
    if value is not value:  # noqa: PLR0124 - NA-propagating self-comparison
        return True
    try:
        text = str(value).strip().lower()
    except Exception:  # pragma: no cover - defensive
        return True
    return text in _BLANK_TOKENS


def normalize_text(value: object) -> str | None:
    """Collapse whitespace and strip a free-text field, or return ``None``."""
    if is_blank(value):
        return None
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_sex(value: object) -> str:
    """Map a source sex field onto :class:`~adharmon.schema.Sex`.

    Returns ``Sex.UNKNOWN`` rather than ``None`` so that the column is never
    null: "we asked and the study did not say" and "we never asked" are the
    same fact downstream, and a non-null categorical is easier to group by.
    """
    if is_blank(value):
        return Sex.UNKNOWN.value
    text = str(value).strip().lower().rstrip(".")
    if text in _SEX_MALE:
        return Sex.MALE.value
    if text in _SEX_FEMALE:
        return Sex.FEMALE.value
    return Sex.UNKNOWN.value


def normalize_diagnosis(value: object) -> str:
    """Map a free-text diagnosis onto :class:`~adharmon.schema.Diagnosis`."""
    if is_blank(value):
        return Diagnosis.UNKNOWN.value
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    text = text.replace("'", "").replace("’", "")

    if text in _DIAGNOSIS_EXACT:
        return _DIAGNOSIS_EXACT[text].value

    # "probable AD", "definite AD", "possible AD" -> AD
    stripped = re.sub(r"^(probable|definite|possible|suspected)\s+", "", text)
    if stripped in _DIAGNOSIS_EXACT:
        return _DIAGNOSIS_EXACT[stripped].value

    for needle, diagnosis in _DIAGNOSIS_SUBSTRINGS:
        if needle in text:
            return diagnosis.value

    return Diagnosis.OTHER.value


def normalize_age(value: object) -> tuple[float | None, bool]:
    """Parse an age field, returning ``(age_years, was_censored)``.

    Ages above :data:`~adharmon.schema.HIPAA_AGE_CAP` are capped and flagged.
    Studies express this ceiling inconsistently -- ``"90+"``, ``"90 or older"``,
    ``">89"`` -- and some report a true age above 90 that we are obliged to
    censor ourselves before it reaches a shared layer. Both paths set the flag,
    so a downstream survival analysis can exclude censored ages instead of
    treating a wall of exactly-90.0 values as real data.
    """
    if is_blank(value):
        return None, False

    text = str(value).strip().lower()

    if re.search(r"(90\s*\+|90\s*or\s*(older|above|more)|>\s*(89|90)|89\+)", text):
        return HIPAA_AGE_CAP, True

    cleaned = re.sub(r"\b(years?|yrs?|y/?o)\b", "", text)
    cleaned = cleaned.replace(",", ".").strip()

    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if match is None:
        return None, False

    age = float(match.group())
    if age > HIPAA_AGE_CAP:
        return HIPAA_AGE_CAP, True
    return age, False


def normalize_braak(value: object) -> int | None:
    """Parse a Braak stage, accepting Roman numerals, digits, and prefixes.

    Handles ``"III"``, ``"3"``, ``"Braak III"``, ``"stage 3"``, ``"B2"``, and
    ranges such as ``"III-IV"`` (which take the lower bound, the conservative
    reading of an uncertain stage).
    """
    if is_blank(value):
        return None

    text = str(value).strip().lower()
    text = re.sub(r"\b(braak|stage|nft|score)\b", " ", text)
    text = text.replace("&", "-").replace("/", "-").replace("to", "-")
    text = re.sub(r"\s+", " ", text).strip(" .-")

    # Take the lower bound of a range such as "iii-iv" or "3-4".
    first = re.split(r"\s*-\s*|\s+or\s+", text)[0].strip()

    if first in _ROMAN_NUMERALS:
        return _ROMAN_NUMERALS[first]

    # "b2" is the ABC-score shorthand for Braak stage pairs; keep the digit.
    digits = re.search(r"\d+", first)
    if digits is not None:
        return int(digits.group())

    return None


def normalize_cerad(value: object) -> int | None:
    """Parse a CERAD neuritic plaque score onto 0-3.

    Accepts numeric scores and the semantic labels used interchangeably with
    them (``"none"``, ``"sparse"``, ``"moderate"``, ``"frequent"``).

    Note the deliberate ordering: ``"none"`` is both a valid CERAD score (no
    neuritic plaques) and one of the most common missing-data sentinels. Inside
    a CERAD column the score reading wins, so the semantic vocabulary is
    consulted *before* the blank check. Getting this backwards silently converts
    every plaque-free control into missing data -- which is the group a
    case/control comparison depends on most.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    text = str(value).strip().lower()
    text = re.sub(r"\b(cerad|score|plaques?|neuritic)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .-")

    labels = {
        "none": 0,
        "zero": 0,
        "sparse": 1,
        "mild": 1,
        "moderate": 2,
        "frequent": 3,
        "severe": 3,
    }
    if text in labels:
        return labels[text]

    if is_blank(text):
        return None

    digits = re.search(r"\d+", text)
    if digits is not None:
        return int(digits.group())
    return None


def normalize_apoe(value: object) -> str | None:
    """Normalize an APOE genotype to sorted ``E<n>/E<n>`` form.

    Accepts ``"33"``, ``"3/3"``, ``"E3/E3"``, ``"e3e3"``, ``"apoe 3/4"``.
    Alleles are sorted so that ``"43"`` and ``"34"`` collapse to one genotype --
    without this, an APOE4-carrier stratification silently splits into two
    groups and the effect size halves.
    """
    if is_blank(value):
        return None

    alleles = re.findall(r"[234]", str(value))
    if len(alleles) != 2:
        return None
    first, second = sorted(alleles)
    return f"E{first}/E{second}"


def normalize_float(value: object) -> float | None:
    """Parse a numeric measurement (RIN, PMI) tolerating units and symbols."""
    if is_blank(value):
        return None
    text = str(value).strip().lower()
    text = re.sub(r"\b(hours?|hrs?|h)\b", "", text).replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    return float(match.group())
