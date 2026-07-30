"""Every value this system reports must say where it came from.

The predecessor project returned invented binding constants and fabricated
experimental methods alongside real PubMed citations. The fix is structural:
tools return values wrapped in `Sourced`, so a number without a traceable
origin cannot be constructed by accident.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Provenance:
    source: str
    url: str
    license: str | None = None
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Sourced(Generic[T]):
    """A value plus the record of where it came from."""

    value: T
    provenance: Provenance

    def as_dict(self) -> dict[str, Any]:
        value = self.value
        if hasattr(value, "as_dict"):
            value = value.as_dict()
        return {"value": value, "provenance": self.provenance.as_dict()}


ALPHAFOLD = dict(
    source="AlphaFold Protein Structure Database (Google DeepMind / EMBL-EBI)",
    license="CC-BY-4.0",
)
UNIPROT = dict(source="UniProtKB", license="CC-BY-4.0")
PUBCHEM = dict(source="PubChem (NCBI)", license="Public domain (US Government work)")
PUBMED = dict(source="PubMed (NCBI E-utilities)", license="See NCBI usage policy")
OPENFDA = dict(source="openFDA Food Enforcement", license="https://open.fda.gov/license/")
