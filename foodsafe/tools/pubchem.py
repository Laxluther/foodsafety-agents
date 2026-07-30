"""Resolve a compound name to a real structure via PubChem PUG REST.

This replaces the predecessor's `_mock_molecular_properties`, which invented a
molecular weight with `np.random.uniform(200, 800)`. A toxin's structure is a
matter of public record; there is no reason to guess it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..provenance import PUBCHEM, Provenance, Sourced
from ._http import ApiError, get_json

_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_WEB = "https://pubchem.ncbi.nlm.nih.gov/compound"

# PubChem renamed CanonicalSMILES to ConnectivitySMILES; older deployments still
# return the former, so both are requested and whichever arrives is used.
_SMILES_KEYS = ("ConnectivitySMILES", "CanonicalSMILES", "IsomericSMILES", "SMILES")
_PROPERTIES = "ConnectivitySMILES,CanonicalSMILES,MolecularFormula,MolecularWeight,IUPACName"


@dataclass(frozen=True)
class Compound:
    cid: int
    name: str
    smiles: str
    formula: str
    molecular_weight: float
    iupac_name: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_compound(name: str) -> Sourced[Compound]:
    """Look up a compound by name. Raises ApiError if PubChem has no match."""
    url = f"{_BASE}/compound/name/{name}/property/{_PROPERTIES}/JSON"
    payload = get_json(url)

    try:
        record = payload["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ApiError(f"unexpected PubChem response shape for {name!r}") from exc

    smiles = next((record[k] for k in _SMILES_KEYS if record.get(k)), None)
    if not smiles:
        raise ApiError(f"PubChem returned no SMILES for {name!r}")

    cid = int(record["CID"])
    compound = Compound(
        cid=cid,
        name=name,
        smiles=smiles,
        formula=record.get("MolecularFormula", ""),
        molecular_weight=float(record["MolecularWeight"]),
        iupac_name=record.get("IUPACName"),
    )
    return Sourced(compound, Provenance(url=f"{_WEB}/{cid}", **PUBCHEM))
