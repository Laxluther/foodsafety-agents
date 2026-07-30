"""Resolve a food protein to a reviewed UniProtKB entry.

Reviewed (Swiss-Prot) entries are strongly preferred: they are manually curated
and carry the stable accessions that AlphaFold DB is keyed on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..provenance import UNIPROT, Provenance, Sourced
from ._http import ApiError, get_json

_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
_ENTRY = "https://www.uniprot.org/uniprotkb"
_FIELDS = "accession,id,protein_name,organism_name,length,sequence,reviewed"


@dataclass(frozen=True)
class Protein:
    accession: str
    entry_name: str
    protein_name: str
    organism: str
    length: int
    sequence: str
    reviewed: bool

    def as_dict(self) -> dict:
        return asdict(self)


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


def resolve_protein(name: str, organism: str | None = None, reviewed_only: bool = True) -> Sourced[Protein]:
    """Find the best UniProt entry for a protein name.

    `organism` accepts a scientific name ("Gallus gallus") or NCBI taxon id.
    """
    query = name
    if organism:
        field = "organism_id" if str(organism).isdigit() else "organism_name"
        query += f' AND {field}:"{organism}"'
    if reviewed_only:
        query += " AND reviewed:true"

    payload = get_json(_SEARCH, params={"query": query, "fields": _FIELDS, "format": "json", "size": 1})
    results = payload.get("results") if isinstance(payload, dict) else None

    if not results:
        if reviewed_only:
            # Many food allergens only exist as unreviewed TrEMBL entries.
            return resolve_protein(name, organism, reviewed_only=False)
        raise ApiError(f"UniProt has no entry for {name!r}")

    entry = results[0]
    accession = entry["primaryAccession"]
    protein = Protein(
        accession=accession,
        entry_name=entry.get("uniProtkbId", ""),
        protein_name=_protein_name(entry),
        organism=entry.get("organism", {}).get("scientificName", ""),
        length=int(entry.get("sequence", {}).get("length", 0)),
        sequence=entry.get("sequence", {}).get("value", ""),
        reviewed=entry.get("entryType", "").startswith("UniProtKB reviewed"),
    )
    return Sourced(protein, Provenance(url=f"{_ENTRY}/{accession}", **UNIPROT))
