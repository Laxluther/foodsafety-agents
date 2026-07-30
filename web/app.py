"""FastAPI server for the evidence viewer.

The deterministic endpoints are fast and always available. The agent endpoint is
slow because it runs a local model, so the UI can render real evidence long
before any narrative exists — which is the right order anyway: facts first,
interpretation second.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodsafe import report as evidence_report
from foodsafe.tools import alphafold, chem
from foodsafe.tools._http import ApiError

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="foodsafety-agents", version="0.1.0")


class AnalysisRequest(BaseModel):
    protein: str | None = None
    contaminant: str | None = None
    organism: str | None = None
    commodity: str | None = None
    measured_ug_per_kg: float | None = None
    jurisdiction: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/evidence")
def gather_evidence(request: AnalysisRequest) -> dict:
    """Deterministic evidence gathering — no language model involved."""
    query = evidence_report.Query(**request.model_dump())
    result = evidence_report.assemble(query)
    return {**result.as_dict(), "sources": result.sources()}


@app.get("/api/limits")
def limits(
    toxin: str = Query(..., min_length=1),
    measured_ug_per_kg: float | None = None,
    jurisdiction: str | None = None,
) -> dict:
    """Regulatory comparison only.

    Limits are indexed by toxin group ("aflatoxins, total") while the compound
    lookup wants the specific molecule ("aflatoxin B1"), so the UI needs these
    separately. Keeping it off /api/evidence avoids re-querying UniProt,
    AlphaFold, PubMed and openFDA just to re-read a local JSON table.
    """
    from foodsafe.tools import regulatory

    if measured_ug_per_kg is None:
        found = regulatory.find_limits(toxin, jurisdiction)
        return {"comparisons": [], "limits": [limit.as_dict() for limit in found]}

    try:
        comparisons = regulatory.compare(measured_ug_per_kg, toxin, jurisdiction)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"comparisons": [c.as_dict() for c in comparisons]}


@app.get("/api/structure/{accession}.pdb", response_class=PlainTextResponse)
def structure_pdb(accession: str) -> str:
    """Proxy the AlphaFold PDB file so the browser viewer avoids a cross-origin fetch."""
    try:
        structure = alphafold.fetch_structure(accession)
        return alphafold.fetch_pdb(structure.value)
    except ApiError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/molecule.svg")
def molecule_svg(smiles: str = Query(..., min_length=1)) -> Response:
    """2D depiction drawn by RDKit from the real structure."""
    try:
        svg = chem.to_svg(smiles)
    except chem.InvalidStructure as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=svg, media_type="image/svg+xml")


class AgentRequest(BaseModel):
    request: str
    model: str | None = None


@app.post("/api/agent-report")
def agent_report(body: AgentRequest) -> dict:
    """Run the ADK pipeline. Slow: a local model generates through five agents."""
    from foodsafe.agents import analyse
    from foodsafe.llm import DEFAULT_MODEL

    try:
        result = analyse(body.request, model_name=body.model or DEFAULT_MODEL)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"agent run failed: {exc}") from exc
    return result.as_dict()


@app.get("/api/models")
def models() -> dict:
    from foodsafe.llm import Ollama

    client = Ollama()
    return {"installed": client.installed_models(), "default": client.model}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
