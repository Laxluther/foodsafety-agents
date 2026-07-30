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
    gathered = evidence_report.assemble(evidence_report.query(**request.model_dump()))
    return {**gathered, "sources": evidence_report.sources(gathered)}


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

    try:
        if measured_ug_per_kg is None:
            return {"comparisons": [], "limits": regulatory.find_limits(toxin, jurisdiction)}
        return {"comparisons": regulatory.compare(measured_ug_per_kg, toxin, jurisdiction)}
    except (ValueError, regulatory.UnknownJurisdiction) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/structure/{accession}.pdb", response_class=PlainTextResponse)
def structure_pdb(accession: str) -> str:
    """Proxy the AlphaFold PDB file so the browser viewer avoids a cross-origin fetch."""
    try:
        structure = alphafold.fetch_structure(accession)
        return alphafold.fetch_pdb(structure["value"])
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
        return analyse(body.request, model_name=body.model or DEFAULT_MODEL)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"agent run failed: {exc}") from exc


class Question(BaseModel):
    question: str
    model: str | None = None


@app.post("/api/ask")
def ask(body: Question) -> dict:
    """Answer a food safety question, or refuse it.

    Refusals are decided before any model runs, so an out-of-scope question
    returns immediately and costs nothing.
    """
    from foodsafe import assistant
    from foodsafe.llm import DEFAULT_MODEL

    try:
        return assistant.ask(body.question, model_name=body.model or DEFAULT_MODEL)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"assistant unavailable: {exc}") from exc


@app.get("/api/models")
def models() -> dict:
    from foodsafe import llm

    return {"installed": llm.installed_models(), "default": llm.DEFAULT_MODEL}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
