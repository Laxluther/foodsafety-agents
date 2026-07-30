"""Resolve a food protein to a UniProtKB entry.

Reviewed (Swiss-Prot) entries are strongly preferred: they are manually curated
and carry the stable accessions that AlphaFold DB is keyed on.
"""

from ..provenance import UNIPROT, sourced
from ._http import ApiError, get_json

_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
_ENTRY = "https://www.uniprot.org/uniprotkb"
_FIELDS = "accession,id,protein_name,organism_name,length,sequence,reviewed"


def _protein_name(entry: dict) -> str:
    description = entry.get("proteinDescription", {})
    recommended = description.get("recommendedName", {}).get("fullName", {}).get("value")
    if recommended:
        return recommended
    for submission in description.get("submissionNames", []):
        value = submission.get("fullName", {}).get("value")
        if value:
            return value
    return entry.get("uniProtkbId", "")


def _build_query(name: str, organism: str | None, reviewed_only: bool) -> str:
    query = name
    if organism:
        field = "organism_id" if str(organism).isdigit() else "organism_name"
        query += f' AND {field}:"{organism}"'
    if reviewed_only:
        query += " AND reviewed:true"
    return query


def resolve_protein(
    name: str, organism: str | None = None, reviewed_only: bool = True
) -> dict:
    """Find the best UniProt entry for a protein name.

    `organism` accepts a scientific name ("Gallus gallus") or an NCBI taxon id.
    Falls back to unreviewed entries, since many food allergens only exist in
    TrEMBL. Returns {"value": {accession, protein_name, ...}, "provenance": {...}}.
    """
    payload = get_json(
        _SEARCH,
        params={
            "query": _build_query(name, organism, reviewed_only),
            "fields": _FIELDS,
            "format": "json",
            "size": 1,
        },
    )
    results = payload.get("results") if isinstance(payload, dict) else None

    if not results:
        if reviewed_only:
            return resolve_protein(name, organism, reviewed_only=False)
        raise ApiError(f"UniProt has no entry for {name!r}")

    entry = results[0]
    accession = entry["primaryAccession"]
    protein = {
        "accession": accession,
        "entry_name": entry.get("uniProtkbId", ""),
        "protein_name": _protein_name(entry),
        "organism": entry.get("organism", {}).get("scientificName", ""),
        "length": int(entry.get("sequence", {}).get("length", 0)),
        "sequence": entry.get("sequence", {}).get("value", ""),
        "reviewed": entry.get("entryType", "").startswith("UniProtKB reviewed"),
    }
    return sourced(protein, UNIPROT, f"{_ENTRY}/{accession}")
