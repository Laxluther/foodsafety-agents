"""Regulatory comparison against published limits, and real recall history.

This replaces the predecessor's "risk score (0-10)", which was assembled from
heuristics and random noise and then presented as a compliance verdict. A
measured concentration compared against a published maximum level is a
checkable statement; a synthesised risk score is not.

The bundled limits table is a curated subset — see data/mycotoxin_limits.json.
Regulations are amended, so `verify_url` is returned with every comparison.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from ..provenance import OPENFDA, Provenance, Sourced
from ._http import ApiError, get_json

_DATA = Path(__file__).resolve().parent.parent / "data" / "mycotoxin_limits.json"
_OPENFDA = "https://api.fda.gov/food/enforcement.json"


@dataclass(frozen=True)
class Limit:
    toxin: str
    commodity: str
    jurisdiction: str
    jurisdiction_name: str
    max_level_ug_per_kg: float
    instrument: str
    verify_url: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Comparison:
    measured_ug_per_kg: float
    limit: Limit
    exceeds: bool
    ratio_of_limit: float

    def as_dict(self) -> dict:
        data = asdict(self)
        data["limit"] = self.limit.as_dict()
        return data

    def statement(self) -> str:
        verb = "exceeds" if self.exceeds else "is within"
        return (
            f"{self.measured_ug_per_kg} ug/kg {verb} the {self.limit.jurisdiction_name} "
            f"maximum of {self.limit.max_level_ug_per_kg} ug/kg for "
            f"{self.limit.toxin} in {self.limit.commodity} "
            f"({self.limit.instrument})."
        )


@lru_cache(maxsize=1)
def _load() -> dict:
    return json.loads(_DATA.read_text(encoding="utf-8"))


def _to_limit(row: dict, jurisdictions: dict) -> Limit:
    meta = jurisdictions[row["jurisdiction"]]
    return Limit(
        toxin=row["toxin"],
        commodity=row["commodity"],
        jurisdiction=row["jurisdiction"],
        jurisdiction_name=meta["name"],
        max_level_ug_per_kg=row["max_level_ug_per_kg"],
        instrument=meta["instrument"],
        verify_url=meta["url"],
    )


def find_limits(toxin: str, jurisdiction: str | None = None) -> list[Limit]:
    """Published limits matching a toxin name, optionally filtered by jurisdiction."""
    data = _load()
    needle = toxin.strip().lower()
    matches = []
    for row in data["limits"]:
        if needle not in row["toxin"].lower():
            continue
        if jurisdiction and row["jurisdiction"] != jurisdiction.upper():
            continue
        matches.append(_to_limit(row, data["jurisdictions"]))
    return matches


def compare(measured_ug_per_kg: float, toxin: str, jurisdiction: str | None = None) -> list[Comparison]:
    """Compare a measured concentration against every applicable published limit."""
    if measured_ug_per_kg < 0:
        raise ValueError("measured concentration cannot be negative")

    comparisons = []
    for limit in find_limits(toxin, jurisdiction):
        comparisons.append(
            Comparison(
                measured_ug_per_kg=measured_ug_per_kg,
                limit=limit,
                exceeds=measured_ug_per_kg > limit.max_level_ug_per_kg,
                ratio_of_limit=round(measured_ug_per_kg / limit.max_level_ug_per_kg, 3),
            )
        )
    return comparisons


def recent_recalls(term: str, limit: int = 10) -> Sourced[list[dict]]:
    """Real FDA food enforcement records mentioning a term.

    openFDA returns 404 when a search has no matches, which is a legitimate
    empty result rather than an error.
    """
    params = {"search": f'reason_for_recall:"{term}"', "limit": limit}
    url = f"{_OPENFDA}?search=reason_for_recall:%22{term}%22&limit={limit}"

    try:
        payload = get_json(_OPENFDA, params=params)
    except ApiError:
        return Sourced([], Provenance(url=url, **OPENFDA))

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
    return Sourced(recalls, Provenance(url=url, **OPENFDA))
