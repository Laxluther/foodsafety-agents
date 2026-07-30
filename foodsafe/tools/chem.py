"""Molecular descriptors computed with RDKit.

Every value here is deterministic and reproducible from the SMILES string —
run it twice, get the same answer. The predecessor returned these same fields
from `np.random.uniform`, which is why RDKit is a hard dependency now rather
than an optional one with a random fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from ..provenance import Provenance, Sourced

RDLogger.DisableLog("rdApp.*")


class InvalidStructure(ValueError):
    """SMILES that RDKit could not parse into a molecule."""


@dataclass(frozen=True)
class Descriptors_:
    molecular_weight: float
    exact_mass: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    aromatic_rings: int
    heavy_atoms: int
    formal_charge: int
    lipinski_violations: int
    formula: str

    def as_dict(self) -> dict:
        return asdict(self)


def _lipinski_violations(mw: float, logp: float, hbd: int, hba: int) -> int:
    """Lipinski's rule of five — a count of the thresholds exceeded."""
    return sum([mw > 500, logp > 5, hbd > 5, hba > 10])


def describe(smiles: str) -> Sourced[Descriptors_]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidStructure(f"RDKit could not parse SMILES: {smiles!r}")

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    descriptors = Descriptors_(
        molecular_weight=round(mw, 2),
        exact_mass=round(Descriptors.ExactMolWt(mol), 4),
        logp=round(logp, 2),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        hbd=hbd,
        hba=hba,
        rotatable_bonds=Lipinski.NumRotatableBonds(mol),
        aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
        heavy_atoms=mol.GetNumHeavyAtoms(),
        formal_charge=Chem.GetFormalCharge(mol),
        lipinski_violations=_lipinski_violations(mw, logp, hbd, hba),
        formula=rdMolDescriptors.CalcMolFormula(mol),
    )

    from rdkit import __version__ as rdkit_version

    return Sourced(
        descriptors,
        Provenance(
            source=f"Computed with RDKit {rdkit_version}",
            url="https://www.rdkit.org/docs/GettingStartedInPython.html#descriptor-calculation",
            license="BSD-3-Clause",
        ),
    )


def to_svg(smiles: str, width: int = 420, height: int = 320) -> str:
    """2D depiction for the frontend, drawn from the real structure."""
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidStructure(f"RDKit could not parse SMILES: {smiles!r}")

    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
