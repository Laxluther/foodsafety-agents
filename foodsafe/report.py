"""Deterministic evidence gathering.

This stage runs before any language model touches the data. It collects facts
from the public APIs and nothing else -- no interpretation, no scoring, no
narrative. If a source is unavailable the gap is recorded in `warnings` rather
than filled in, because a missing value is information and a plausible
substitute is not.
"""

from .tools import alphafold, chem, literature, pubchem, regulatory, uniprot

_SOURCED_KEYS = ("protein", "structure", "compound", "descriptors", "citations", "recalls")


def query(
    protein: str | None = None,
    contaminant: str | None = None,
    organism: str | None = None,
    commodity: str | None = None,
    measured_ug_per_kg: float | None = None,
    jurisdiction: str | None = None,
) -> dict:
    """Build the request the gatherer works from."""
    return {
        "protein": protein,
        "contaminant": contaminant,
        "organism": organism,
        "commodity": commodity,
        "measured_ug_per_kg": measured_ug_per_kg,
        "jurisdiction": jurisdiction,
    }


def sources(report: dict) -> list[dict]:
    """Every distinct source consulted, for the citations panel."""
    seen = {}
    for key in _SOURCED_KEYS:
        entry = report.get(key)
        if entry:
            seen[entry["provenance"]["url"]] = entry["provenance"]

    for comparison in report.get("comparisons", []):
        limit = comparison["limit"]
        seen[limit["verify_url"]] = {
            "source": limit["instrument"],
            "url": limit["verify_url"],
            "license": None,
        }
    return list(seen.values())


def _try(report: dict, key: str, description: str, fetch):
    """Run one lookup, recording a failure as a warning instead of aborting."""
    try:
        report[key] = fetch()
    except Exception as exc:  # noqa: BLE001 - one dead source must not end the run
        report["warnings"].append(f"{description}: {exc}")
    return report[key]


def assemble(request: dict) -> dict:
    """Gather everything the public sources can say about this request."""
    report = {"query": request, "comparisons": [], "warnings": []}
    for key in _SOURCED_KEYS:
        report[key] = None

    protein = request.get("protein")
    contaminant = request.get("contaminant")
    measured = request.get("measured_ug_per_kg")

    if protein:
        found = _try(
            report, "protein", f"UniProt lookup failed for {protein!r}",
            lambda: uniprot.resolve_protein(protein, organism=request.get("organism")),
        )
        if found:
            accession = found["value"]["accession"]
            _try(
                report, "structure", f"AlphaFold has no structure for {accession}",
                lambda: alphafold.fetch_structure(accession),
            )

    if contaminant:
        found = _try(
            report, "compound", f"PubChem lookup failed for {contaminant!r}",
            lambda: pubchem.resolve_compound(contaminant),
        )
        if found:
            _try(
                report, "descriptors", "RDKit could not parse the PubChem structure",
                lambda: chem.describe(found["value"]["smiles"]),
            )

    if protein and contaminant:
        found = _try(
            report, "citations", "PubMed search failed",
            lambda: literature.interaction_evidence(protein, contaminant),
        )
        if found is not None and not found["value"]:
            report["warnings"].append(
                "No indexed literature found for this protein/contaminant pair. "
                "Absence of evidence is reported as such."
            )

    if contaminant and measured is not None:
        report["comparisons"] = regulatory.compare(
            measured, contaminant, jurisdiction=request.get("jurisdiction")
        )
        if not report["comparisons"]:
            report["warnings"].append(
                f"No published limit for {contaminant!r} in the bundled table; "
                "no compliance statement can be made."
            )

    if contaminant:
        _try(
            report, "recalls", "openFDA recall lookup failed",
            lambda: regulatory.recent_recalls(contaminant, limit=5),
        )

    return report
