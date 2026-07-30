"""Multi-agent pipeline built on Google's Agent Development Kit.

Five specialists run in sequence, each with only the tools its role needs, and
a reporter that synthesises what they found. Every agent operates under one
standing instruction: state only what the tools returned. Whatever the reporter
writes is then checked by `grounding.check`, which flags any number that does
not trace back to gathered evidence.

The model runs locally through Ollama via LiteLLM, so no data leaves the machine.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm

from . import adk_tools
from .llm import DEFAULT_BASE_URL, DEFAULT_MODEL

_HOUSE_RULES = """
You are part of a food safety evidence system. Absolute rules:

1. State only what the tools returned. Never estimate, extrapolate or invent a
   number. If you did not receive a value, say it is unavailable.
2. Never invent binding affinities, Kd values, experimental methods, or
   confidence scores. If a tool did not supply it, it does not exist.
3. If a tool returns status "error" or an empty result, report that plainly.
   Absence of evidence is a finding, not a gap to fill.
4. Attribute every figure to the tool or source that produced it.
5. Be concise. No preamble, no restating the question.
"""


# LiteLLM defaults to a 600s timeout, which a large model partially offloaded to
# CPU can exceed on a single generation - the reporter in particular, since it is
# handed all four specialists' findings at once.
LLM_TIMEOUT_SECONDS = int(os.environ.get("FOODSAFE_LLM_TIMEOUT", "3600"))

# Keeping the model resident avoids paying the load cost again at every step of
# the sequence; an 18GB model takes ~30s just to come back off disk.
OLLAMA_KEEP_ALIVE = os.environ.get("FOODSAFE_KEEP_ALIVE", "30m")


def _model(model_name: str) -> LiteLlm:
    """Route ADK through LiteLLM to a local Ollama model."""
    return LiteLlm(
        model=f"ollama_chat/{model_name}",
        api_base=DEFAULT_BASE_URL,
        timeout=LLM_TIMEOUT_SECONDS,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )


def build_agents(model_name: str = DEFAULT_MODEL) -> SequentialAgent:
    model = _model(model_name)

    structural_biologist = LlmAgent(
        name="structural_biologist",
        model=model,
        description="Identifies the protein and retrieves its AlphaFold structure.",
        instruction=_HOUSE_RULES + """
Your job: resolve the protein named in the request with resolve_protein, then
call fetch_protein_structure with the accession you got back.

Report the accession, sequence length, mean pLDDT and what that confidence band
means. pLDDT is a prediction confidence, not an experimental measurement - say
so. If confidence is below 70, warn that the model is unreliable.
""",
        tools=adk_tools.ALL_TOOLS[:2],
        output_key="structure_findings",
    )

    chemist = LlmAgent(
        name="chemist",
        model=model,
        description="Retrieves the contaminant structure and computes descriptors.",
        instruction=_HOUSE_RULES + """
Your job: call resolve_compound for the contaminant, then describe_molecule with
the SMILES it returned.

Report formula, molecular weight, logP, TPSA and Lipinski violations. Note that
RDKit's computed molecular weight can be cross-checked against PubChem's - if
they disagree by more than a rounding difference, flag it.

Do not predict toxicity, binding, or absorption. You describe the molecule only.
""",
        tools=adk_tools.ALL_TOOLS[2:4],
        output_key="chemistry_findings",
    )

    literature_analyst = LlmAgent(
        name="literature_analyst",
        model=model,
        description="Finds published evidence for the protein/contaminant pair.",
        instruction=_HOUSE_RULES + """
Your job: call search_literature for the protein and contaminant.

Summarise what the papers are about, citing PMIDs. If no papers were found, say
exactly that - do not substitute general knowledge for indexed evidence, and do
not attach experimental values to any citation.
""",
        tools=[adk_tools.search_literature],
        output_key="literature_findings",
    )

    compliance_officer = LlmAgent(
        name="compliance_officer",
        model=model,
        description="Compares measurements against published regulatory limits.",
        instruction=_HOUSE_RULES + """
The sample details were supplied as structured fields, not prose. Use them exactly:

  contaminant for limit lookup: {limit_toxin}
  measured concentration:       {measured_ug_per_kg} ug/kg
  jurisdiction:                 {jurisdiction}

If the measured concentration is "not provided", skip the comparison and say so.
Otherwise call compare_to_regulatory_limits with exactly those values - do not
re-read them from the conversation text.

Also call recent_recalls for the contaminant.

Report each jurisdiction separately, naming the legal instrument, and state
plainly whether the measurement exceeds the limit. Do not produce an overall
risk score - report the comparison, which is checkable.

If no published limit exists in the table, say no compliance statement can be made.
""",
        tools=adk_tools.ALL_TOOLS[5:],
        output_key="compliance_findings",
    )

    reporter = LlmAgent(
        name="reporter",
        model=model,
        description="Synthesises the specialists' findings into one report.",
        instruction=_HOUSE_RULES + """
Write a food safety evidence report from these findings:

Structure: {structure_findings}
Chemistry: {chemistry_findings}
Literature: {literature_findings}
Compliance: {compliance_findings}

Sections: Summary, Protein, Contaminant, Published Evidence, Regulatory Position,
Limitations.

Every number you write must already appear above. Introducing a figure that is
not in the findings is the one unacceptable error. In Limitations, state what
could not be determined.
""",
        output_key="final_report",
    )

    return SequentialAgent(
        name="food_safety_pipeline",
        description="Evidence-based food safety analysis over public scientific sources.",
        sub_agents=[
            structural_biologist,
            chemist,
            literature_analyst,
            compliance_officer,
            reporter,
        ],
    )


def _result(request, report, outputs, tool_results, numbers, identifiers, by_agent) -> dict:
    """Assemble the run result. `is_grounded` is false if anything was unsupported."""
    return {
        "request": request,
        "report": report,
        "agent_outputs": outputs,
        "tool_results": tool_results,
        "ungrounded_numbers": numbers,
        "ungrounded_identifiers": identifiers,
        "ungrounded_by_agent": by_agent,
        "is_grounded": not numbers and not identifiers and not any(by_agent.values()),
    }


def analyse(
    request: str,
    model_name: str = DEFAULT_MODEL,
    user_id: str = "local",
    limit_toxin: str | None = None,
    measured_ug_per_kg: float | None = None,
    jurisdiction: str | None = None,
) -> dict:
    """Run the pipeline and verify the report against the evidence it collected.

    Sample measurements are passed as structured arguments rather than left to be
    parsed out of the request text. A small model asked to extract "25 ug/kg" from
    prose will sometimes report that no measurement was given, so the values are
    seeded into session state and read from there.
    """
    import asyncio

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from . import grounding

    runner = InMemoryRunner(agent=build_agents(model_name), app_name="foodsafety-agents")

    tool_results: list = []

    async def _run() -> dict:
        session = await runner.session_service.create_session(
            app_name="foodsafety-agents",
            user_id=user_id,
            state={
                "limit_toxin": limit_toxin or "not provided",
                "measured_ug_per_kg": (
                    "not provided" if measured_ug_per_kg is None else measured_ug_per_kg
                ),
                "jurisdiction": jurisdiction or "all jurisdictions",
            },
        )
        message = types.Content(role="user", parts=[types.Part(text=request)])
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=message
        ):
            # Capture what the tools actually returned. This, not the agents'
            # prose, is the ground truth every generated number is checked against.
            if event.content and event.content.parts:
                for part in event.content.parts:
                    response = getattr(part, "function_response", None)
                    if response is not None:
                        tool_results.append(
                            {"tool": response.name, "result": response.response}
                        )
        finished = await runner.session_service.get_session(
            app_name="foodsafety-agents", user_id=user_id, session_id=session.id
        )
        return finished.state

    state = asyncio.run(_run())

    outputs = {
        key: state.get(key, "")
        for key in (
            "structure_findings",
            "chemistry_findings",
            "literature_findings",
            "compliance_findings",
        )
    }
    report = state.get("final_report", "")

    # Ground against the tool results, not against the agents' own prose.
    # Checking the report against the specialists' text only verifies the last
    # hop: if a specialist invents a value the reporter copies it faithfully and
    # the check passes. Every agent is therefore checked against what the tools
    # actually returned, which is the only real evidence in the run.
    evidence = {"tool_results": tool_results, "request": request}

    return _result(
        request=request,
        report=report,
        outputs=outputs,
        tool_results=tool_results,
        numbers=grounding.check(report, evidence),
        identifiers=grounding.check_identifiers(report, evidence),
        by_agent={
            name: grounding.check(text, evidence) + grounding.check_identifiers(text, evidence)
            for name, text in outputs.items()
            if text
        },
    )
