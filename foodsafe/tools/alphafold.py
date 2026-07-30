"""Google DeepMind AlphaFold structures, looked up rather than predicted.

AlphaFold DB holds over 200 million precomputed structures. Fetching the real
one for a UniProt accession is better science than running a local fold — and
it fits on a laptop, which running ESMFold does not.

The predecessor's `mock_structure_prediction` returned `np.random.randn(n, 3)`
as atomic coordinates. These are the actual deposited coordinates, with the
real per-residue pLDDT confidence attached.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..provenance import ALPHAFOLD, Provenance, Sourced
from ._http import ApiError, get_json, get_text

_API = "https://alphafold.ebi.ac.uk/api/prediction"
_ENTRY = "https://alphafold.ebi.ac.uk/entry"


@dataclass(frozen=True)
class Structure:
    uniprot_accession: str
    description: str
    organism: str
    model_version: int
    mean_plddt: float
    confidence_band: str
    fraction_very_high: float
    fraction_confident: float
    fraction_low: float
    fraction_very_low: float
    pdb_url: str
    cif_url: str
    pae_image_url: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def _confidence_band(plddt: float) -> str:
    """AlphaFold's published pLDDT interpretation bands."""
    if plddt >= 90:
        return "very high (backbone and side chains reliable)"
    if plddt >= 70:
        return "confident (backbone reliable)"
    if plddt >= 50:
        return "low (treat with caution)"
    return "very low (likely disordered)"


def fetch_structure(accession: str) -> Sourced[Structure]:
    """Fetch the AlphaFold prediction for a UniProt accession."""
    payload = get_json(f"{_API}/{accession}")
    if not payload:
        raise ApiError(f"AlphaFold DB has no structure for {accession!r}")

    record = payload[0]
    plddt = float(record.get("globalMetricValue", 0.0))

    structure = Structure(
        uniprot_accession=record.get("uniprotAccession", accession),
        description=record.get("uniprotDescription", ""),
        organism=record.get("organismScientificName", ""),
        model_version=int(record.get("latestVersion", 0)),
        mean_plddt=round(plddt, 2),
        confidence_band=_confidence_band(plddt),
        fraction_very_high=record.get("fractionPlddtVeryHigh", 0.0),
        fraction_confident=record.get("fractionPlddtConfident", 0.0),
        fraction_low=record.get("fractionPlddtLow", 0.0),
        fraction_very_low=record.get("fractionPlddtVeryLow", 0.0),
        pdb_url=record.get("pdbUrl", ""),
        cif_url=record.get("cifUrl", ""),
        pae_image_url=record.get("paeImageUrl"),
    )
    return Sourced(structure, Provenance(url=f"{_ENTRY}/{accession}", **ALPHAFOLD))


def fetch_pdb(structure: Structure) -> str:
    """Download the PDB coordinate file, for the 3D viewer in the frontend."""
    if not structure.pdb_url:
        raise ApiError(f"no PDB file listed for {structure.uniprot_accession}")
    return get_text(structure.pdb_url)


def per_residue_plddt(pdb_text: str) -> list[float]:
    """Extract per-residue pLDDT, which AlphaFold stores in the B-factor column.

    Used to colour the structure by confidence instead of showing a flat model.
    """
    seen: dict[int, float] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            residue_number = int(line[22:26])
            b_factor = float(line[60:66])
        except ValueError:
            continue
        seen.setdefault(residue_number, b_factor)
    return [seen[k] for k in sorted(seen)]
