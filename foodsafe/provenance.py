"""Every value this system reports must say where it came from.

The predecessor project returned invented binding constants and fabricated
experimental methods alongside real PubMed citations. The fix is structural:
tools return their value wrapped by `sourced()`, so a number without a traceable
origin cannot be constructed by accident.

Plain dicts, not objects: the whole point of these values is to be serialised to
JSON for the API, written into a report, and walked by the grounding checker.
Wrapping them in classes only to unwrap them again adds nothing.
"""

from datetime import datetime, timezone

ALPHAFOLD = (
    "AlphaFold Protein Structure Database (Google DeepMind / EMBL-EBI)",
    "CC-BY-4.0",
)
UNIPROT = ("UniProtKB", "CC-BY-4.0")
PUBCHEM = ("PubChem (NCBI)", "Public domain (US Government work)")
PUBMED = ("PubMed (NCBI E-utilities)", "See NCBI usage policy")
OPENFDA = ("openFDA Food Enforcement", "https://open.fda.gov/license/")
RDKIT = ("RDKit", "BSD-3-Clause")


def provenance(origin: tuple[str, str | None], url: str) -> dict:
    """Build a provenance record. `origin` is one of the constants above."""
    source, licence = origin
    return {
        "source": source,
        "url": url,
        "license": licence,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def sourced(value, origin: tuple[str, str | None], url: str) -> dict:
    """Pair a value with the record of where it came from."""
    return {"value": value, "provenance": provenance(origin, url)}


def computed(value, tool: str, url: str, licence: str) -> dict:
    """As `sourced`, for values calculated locally rather than fetched."""
    return {"value": value, "provenance": provenance((tool, licence), url)}
