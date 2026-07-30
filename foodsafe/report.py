"""Deterministic evidence gathering.

This stage runs before any language model touches the data. It collects facts
from the public APIs and nothing else — no interpretation, no scoring, no
narrative. If a source is unavailable the gap is recorded in `warnings` rather
than filled in, because a missing value is information and a plausible
substitute is not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .provenance import Sourced
from .tools import alphafold, chem, literature, pubchem, regulatory, uniprot
from .tools._http import ApiError


@dataclass(frozen=True)
class Query:
    protein: str | None = None
    contaminant: str | None = None
    organism: str | None = None
    commodity: str | None = None
    measured_ug_per_kg: float | None = None
    jurisdiction: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceReport:
    query: Query
    protein: Sourced | None = None
    structure: Sourced | None = None
    compound: Sourced | None = None
    descriptors: Sourced | None = None
    citations: Sourced | None = None
    comparisons: list = field(default_factory=list)
    recalls: Sourced | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.as_dict(),
            "protein": self.protein.as_dict() if self.protein else None,
            "structure": self.structure.as_dict() if self.structure else None,
            "compound": self.compound.as_dict() if self.compound else None,
            "descriptors": self.descriptors.as_dict() if self.descriptors else None,
            "citations": self.citations.as_dict() if self.citations else None,
            "comparisons": [c.as_dict() for c in self.comparisons],
            "recalls": self.recalls.as_dict() if self.recalls else None,
            "warnings": list(self.warnings),
        }

    def sources(self) -> list[dict]:
        """Every distinct source consulted, for the citations panel."""
        seen: dict[str, dict] = {}
        for sourced in (self.protein, self.structure, self.compound,
                        self.descriptors, self.citations, self.recalls):
            if sourced is not None:
                p = sourced.provenance
                seen[p.url] = p.as_dict()
        for comparison in self.comparisons:
            seen[comparison.limit.verify_url] = {
                "source": comparison.limit.instrument,
                "url": comparison.limit.verify_url,
                "license": None,
            }
        return list(seen.values())


def assemble(query: Query) -> EvidenceReport:
    """Gather everything the public sources can say about this query."""
    report = EvidenceReport(query=query)

    if query.protein:
        try:
            report.protein = uniprot.resolve_protein(query.protein, organism=query.organism)
        except (ApiError, Exception) as exc:  # noqa: BLE001 - a bad source must not abort the run
            report.warnings.append(f"UniProt lookup failed for {query.protein!r}: {exc}")

        if report.protein is not None:
            try:
                report.structure = alphafold.fetch_structure(report.protein.value.accession)
            except Exception as exc:  # noqa: BLE001
                report.warnings.append(
                    f"AlphaFold has no structure for {report.protein.value.accession}: {exc}"
                )

    if query.contaminant:
        try:
            report.compound = pubchem.resolve_compound(query.contaminant)
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"PubChem lookup failed for {query.contaminant!r}: {exc}")

        if report.compound is not None:
            try:
                report.descriptors = chem.describe(report.compound.value.smiles)
            except chem.InvalidStructure as exc:
                report.warnings.append(f"RDKit could not parse the PubChem structure: {exc}")

    if query.protein and query.contaminant:
        try:
            report.citations = literature.interaction_evidence(query.protein, query.contaminant)
            if not report.citations.value:
                report.warnings.append(
                    "No indexed literature found for this protein/contaminant pair. "
                    "Absence of evidence is reported as such."
                )
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"PubMed search failed: {exc}")

    if query.contaminant and query.measured_ug_per_kg is not None:
        report.comparisons = regulatory.compare(
            query.measured_ug_per_kg, query.contaminant, jurisdiction=query.jurisdiction
        )
        if not report.comparisons:
            report.warnings.append(
                f"No published limit for {query.contaminant!r} in the bundled table; "
                "no compliance statement can be made."
            )

    if query.contaminant:
        try:
            report.recalls = regulatory.recent_recalls(query.contaminant, limit=5)
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"openFDA recall lookup failed: {exc}")

    return report
