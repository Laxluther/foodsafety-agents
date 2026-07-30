"""Multi-agent pipeline built on Google's Agent Development Kit.

Five specialists run in sequence, each with only the tools its role needs, and
a reporter that synthesises what they found. Every agent operates under one
standing instruction: state only what the tools returned. Whatever the reporter
writes is then checked by `grounding.check`, which flags any number that does
not trace back to gathered evidence.

The model runs locally through Ollama via LiteLLM, so no data leaves the machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


def _model(model_name: str) -> LiteLlm:
    """Route ADK through LiteLLM to a local Ollama model."""
    return LiteLlm(model=f"ollama_chat/{model_name}", api_base=DEFAULT_BASE_URL)


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
Your job: if the request contains a measured concentration, call
compare_to_regulatory_limits. Also call recent_recalls for the contaminant.

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


@dataclass
class AnalysisResult:
    request: str
    report: str
    agent_outputs: dict = field(default_factory=dict)
    ungrounded_numbers: list = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return not self.ungrounded_numbers

    def as_dict(self) -> dict:
        return {
            "request": self.request,
            "report": self.report,
            "agent_outputs": self.agent_outputs,
            "ungrounded_numbers": [u.as_dict() for u in self.ungrounded_numbers],
            "is_grounded": self.is_grounded,
        }


def analyse(request: str, model_name: str = DEFAULT_MODEL, user_id: str = "local") -> AnalysisResult:
    """Run the pipeline and verify the report against the evidence it collected."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from . import grounding

    runner = InMemoryRunner(agent=build_agents(model_name), app_name="foodsafety-agents")
    session = runner.session_service.create_session_sync(
        app_name="foodsafety-agents", user_id=user_id
    )

    message = types.Content(role="user", parts=[types.Part(text=request)])
    for _ in runner.run(user_id=user_id, session_id=session.id, new_message=message):
        pass

    state = runner.session_service.get_session_sync(
        app_name="foodsafety-agents", user_id=user_id, session_id=session.id
    ).state

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

    # The specialists' outputs are the evidence the reporter was given, so the
    # report is checked against exactly what it was allowed to draw from.
    findings = grounding.check(report, {"agent_outputs": outputs})

    return AnalysisResult(
        request=request, report=report, agent_outputs=outputs, ungrounded_numbers=findings
    )
