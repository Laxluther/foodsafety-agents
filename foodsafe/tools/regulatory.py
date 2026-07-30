"""Regulatory comparison against published limits, and real recall history.

This replaces the predecessor's "risk score (0-10)", which was assembled from
heuristics and random noise and then presented as a compliance verdict. A
measured concentration compared against a published maximum level is a
checkable statement; a synthesised risk score is not.

The bundled limits table is a curated subset -- see data/mycotoxin_limits.json.
Regulations are amended, so `verify_url` travels with every comparison.
"""

import json
from functools import lru_cache
from pathlib import Path

from ..provenance import OPENFDA, sourced
from ._http import ApiError, get_json

_DATA = Path(__file__).resolve().parent.parent / "data" / "mycotoxin_limits.json"
_OPENFDA = "https://api.fda.gov/food/enforcement.json"

# Phrases that mean "no filter". Callers -- including agents reading a templated
# session value -- naturally write these instead of leaving the field empty, and
# passing them through as a literal filter silently matches nothing.
_ANY_JURISDICTION = {"", "all", "all jurisdictions", "any", "none", "not provided", "null"}


class UnknownJurisdiction(ValueError):
    """A jurisdiction filter that matches nothing, rather than silently doing so."""


@lru_cache(maxsize=1)
def _load() -> dict:
    return json.loads(_DATA.read_text(encoding="utf-8"))


def normalise_jurisdiction(jurisdiction: str | None) -> str | None:
    """Map a jurisdiction argument to a known code, or None meaning all.

    An unrecognised code raises rather than quietly returning no limits, so a
    typo surfaces as an error instead of an apparent absence of regulation.
    """
    if jurisdiction is None:
        return None
    cleaned = jurisdiction.strip()
    if cleaned.lower() in _ANY_JURISDICTION:
        return None

    code = cleaned.upper()
    known = set(_load()["jurisdictions"])
    if code not in known:
        raise UnknownJurisdiction(
            f"unknown jurisdiction {jurisdiction!r}; known codes are {sorted(known)}"
        )
    return code


def _to_limit(row: dict, jurisdictions: dict) -> dict:
    meta = jurisdictions[row["jurisdiction"]]
    return {
        "toxin": row["toxin"],
        "commodity": row["commodity"],
        "jurisdiction": row["jurisdiction"],
        "jurisdiction_name": meta["name"],
        "max_level_ug_per_kg": row["max_level_ug_per_kg"],
        "instrument": meta["instrument"],
        "verify_url": meta["url"],
    }


def find_limits(toxin: str, jurisdiction: str | None = None) -> list[dict]:
    """Published limits matching a toxin name, optionally filtered by jurisdiction."""
    data = _load()
    code = normalise_jurisdiction(jurisdiction)
    needle = toxin.strip().lower()

    return [
        _to_limit(row, data["jurisdictions"])
        for row in data["limits"]
        if needle in row["toxin"].lower() and (not code or row["jurisdiction"] == code)
    ]


def statement(comparison: dict) -> str:
    """Render a comparison as a sentence that names the instrument behind it."""
    limit = comparison["limit"]
    verb = "exceeds" if comparison["exceeds"] else "is within"
    return (
        f"{comparison['measured_ug_per_kg']} ug/kg {verb} the {limit['jurisdiction_name']} "
        f"maximum of {limit['max_level_ug_per_kg']} ug/kg for "
        f"{limit['toxin']} in {limit['commodity']} ({limit['instrument']})."
    )


def compare(
    measured_ug_per_kg: float, toxin: str, jurisdiction: str | None = None
) -> list[dict]:
    """Compare a measured concentration against every applicable published limit."""
    if measured_ug_per_kg < 0:
        raise ValueError("measured concentration cannot be negative")

    comparisons = []
    for limit in find_limits(toxin, jurisdiction):
        maximum = limit["max_level_ug_per_kg"]
        comparisons.append(
            {
                "measured_ug_per_kg": measured_ug_per_kg,
                "limit": limit,
                "exceeds": measured_ug_per_kg > maximum,
                "ratio_of_limit": round(measured_ug_per_kg / maximum, 3),
            }
        )
    return comparisons


def recent_recalls(term: str, limit: int = 10) -> dict:
    """Real FDA food enforcement records mentioning a term.

    openFDA returns 404 when a search has no matches, which is a legitimate
    empty result rather than an error.
    """
    url = f"{_OPENFDA}?search=reason_for_recall:%22{term}%22&limit={limit}"

    try:
        payload = get_json(
            _OPENFDA, params={"search": f'reason_for_recall:"{term}"', "limit": limit}
        )
    except ApiError:
        return sourced([], OPENFDA, url)

    recalls = [
        {
            "recall_number": r.get("recall_number"),
            "reason": r.get("reason_for_recall"),
            "product": (r.get("product_description") or "")[:300],
            "classification": r.get("classification"),
            "status": r.get("status"),
            "recall_initiation_date": r.get("recall_initiation_date"),
            "distribution_pattern": r.get("distribution_pattern"),
            "recalling_firm": r.get("recalling_firm"),
        }
        for r in payload.get("results", [])
    ]
    return sourced(recalls, OPENFDA, url)
