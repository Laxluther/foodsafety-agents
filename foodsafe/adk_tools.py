"""The evidence tools, exposed as Google ADK function tools.

Each returns a plain JSON-serialisable dict carrying its provenance, so a tool
result an agent quotes can always be traced back to the source it came from.
Failures are returned as `status: "error"` rather than raised, so one dead
endpoint does not abort a run — and so an agent is told the fact is missing
instead of being left free to guess it.
"""

from __future__ import annotations

from .tools import alphafold as _alphafold
from .tools import chem as _chem
from .tools import literature as _literature
from .tools import pubchem as _pubchem
from .tools import regulatory as _regulatory
from .tools import uniprot as _uniprot


def _ok(payload: dict, provenance: dict | None = None) -> dict:
    result = {"status": "success", **payload}
    if provenance is not None:
        result["provenance"] = provenance
    return result


def _err(message: str) -> dict:
    return {"status": "error", "error_message": message}


def resolve_protein(name: str, organism: str) -> dict:
    """Look up a protein in UniProtKB and return its accession and sequence.

    Args:
        name: Protein name, for example "ovalbumin".
        organism: Scientific name or NCBI taxon id. Pass "" to search all organisms.

    Returns:
        The UniProt accession, protein name, organism, length and sequence.
    """
    try:
        found = _uniprot.resolve_protein(name, organism=organism or None)
    except Exception as exc:  # noqa: BLE001
        return _err(f"UniProt lookup failed for {name!r}: {exc}")
    return _ok({"protein": found["value"]}, found["provenance"])


def fetch_protein_structure(uniprot_accession: str) -> dict:
    """Fetch the AlphaFold (Google DeepMind) structure for a UniProt accession.

    Args:
        uniprot_accession: A UniProt accession such as "P01012".

    Returns:
        Mean pLDDT confidence, its interpretation band, and structure file URLs.
    """
    try:
        found = _alphafold.fetch_structure(uniprot_accession)
    except Exception as exc:  # noqa: BLE001
        return _err(f"AlphaFold has no structure for {uniprot_accession!r}: {exc}")
    return _ok({"structure": found["value"]}, found["provenance"])


def resolve_compound(name: str) -> dict:
    """Look up a chemical compound in PubChem and return its real structure.

    Args:
        name: Compound name, for example "aflatoxin B1".

    Returns:
        PubChem CID, molecular formula, molecular weight and SMILES.
    """
    try:
        found = _pubchem.resolve_compound(name)
    except Exception as exc:  # noqa: BLE001
        return _err(f"PubChem lookup failed for {name!r}: {exc}")
    return _ok({"compound": found["value"]}, found["provenance"])


def describe_molecule(smiles: str) -> dict:
    """Compute molecular descriptors from a SMILES string using RDKit.

    Args:
        smiles: A SMILES string, normally taken from resolve_compound.

    Returns:
        Molecular weight, logP, TPSA, hydrogen bond donors/acceptors and
        Lipinski rule-of-five violations. Deterministic for a given input.
    """
    try:
        found = _chem.describe(smiles)
    except Exception as exc:  # noqa: BLE001
        return _err(f"RDKit could not parse {smiles!r}: {exc}")
    return _ok({"descriptors": found["value"]}, found["provenance"])


def search_literature(protein: str, contaminant: str) -> dict:
    """Search PubMed for published work on a protein/contaminant pair.

    Returns only what PubMed indexed. An empty list means no evidence was found,
    which is a valid finding and must be reported as such.

    Args:
        protein: Protein name, for example "ovalbumin".
        contaminant: Contaminant name, for example "aflatoxin B1".

    Returns:
        Citations with PMID, title, authors, journal, year and DOI.
    """
    try:
        found = _literature.interaction_evidence(protein, contaminant)
    except Exception as exc:  # noqa: BLE001
        return _err(f"PubMed search failed: {exc}")
    return _ok(
        {"citation_count": len(found["value"]), "citations": found["value"]},
        found["provenance"],
    )


def compare_to_regulatory_limits(contaminant: str, measured_ug_per_kg: float, jurisdiction: str) -> dict:
    """Compare a measured concentration against published regulatory limits.

    Args:
        contaminant: Contaminant name, for example "aflatoxins, total".
        measured_ug_per_kg: Measured concentration in micrograms per kilogram.
        jurisdiction: "EU", "US", "IN", or "" for all jurisdictions.

    Returns:
        One comparison per applicable limit, each naming its legal instrument
        and whether the measurement exceeds it.
    """
    try:
        comparisons = _regulatory.compare(
            measured_ug_per_kg, contaminant, jurisdiction=jurisdiction or None
        )
    except ValueError as exc:
        return _err(str(exc))

    if not comparisons:
        return _err(
            f"No published limit for {contaminant!r} in the bundled table. "
            "No compliance statement can be made."
        )
    return _ok(
        {
            "comparisons": comparisons,
            "statements": [_regulatory.statement(c) for c in comparisons],
        }
    )


def recent_recalls(term: str) -> dict:
    """Search real FDA food enforcement (recall) records.

    Args:
        term: A term appearing in the recall reason, for example "aflatoxin".

    Returns:
        Matching recall records, or an empty list if there are none.
    """
    try:
        found = _regulatory.recent_recalls(term, limit=5)
    except Exception as exc:  # noqa: BLE001
        return _err(f"openFDA lookup failed: {exc}")
    return _ok({"recall_count": len(found["value"]), "recalls": found["value"]}, found["provenance"])


ALL_TOOLS = [
    resolve_protein,
    fetch_protein_structure,
    resolve_compound,
    describe_molecule,
    search_literature,
    compare_to_regulatory_limits,
    recent_recalls,
]
