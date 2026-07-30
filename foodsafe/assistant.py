"""Question answering restricted to food safety, over the same sourced tools.

Two gates, because a prompt instruction alone is not a control. A model told to
stay on topic will still drift, and a jailbreak only has to work once.

  1. in_scope()  runs before any model call. Deterministic, testable, and cheap:
                 an off-topic question is refused without the model ever seeing it.
  2. grounding   runs after. The answer's numbers and identifiers must trace back
                 to what the tools returned.

The first gate stops misuse. The second stops invention. Neither depends on the
model choosing to behave.
"""

import re

from . import adk_tools, grounding
from .llm import DEFAULT_BASE_URL, DEFAULT_MODEL

# Vocabulary that marks a question as belonging to this domain.
_IN_SCOPE_TERMS = {
    # contaminants
    "aflatoxin", "mycotoxin", "ochratoxin", "patulin", "deoxynivalenol", "fumonisin",
    "zearalenone", "toxin", "contaminant", "adulterant", "residue", "pesticide",
    "heavy metal", "melamine", "histamine", "acrylamide",
    # proteins and biology
    "protein", "allergen", "enzyme", "peptide", "amino acid", "sequence",
    "structure", "uniprot", "alphafold", "plddt", "ovalbumin", "casein", "gluten",
    "gliadin", "lactoglobulin", "albumin", "patatin", "tropomyosin",
    # chemistry
    "molecule", "compound", "smiles", "molecular weight", "logp", "tpsa",
    "lipinski", "pubchem", "descriptor", "solubility",
    # food and safety
    "food", "feed", "grain", "cereal", "groundnut", "peanut", "milk", "dairy",
    "coffee", "spice", "maize", "corn", "wheat", "rice", "nut", "juice", "apple",
    "safety", "haccp", "recall", "contamination", "spoilage", "shelf life",
    # regulatory
    "limit", "regulation", "regulatory", "fda", "efsa", "fssai", "codex",
    "compliance", "maximum level", "action level", "ppb", "ug/kg", "µg/kg",
    "jurisdiction", "legal", "permitted",
    # evidence
    "pubmed", "citation", "paper", "study", "literature", "pmid", "evidence",
    "source", "reference",
}

# Requests that are off-task even when they mention a food safety word, and
# attempts to talk the assistant out of its remit.
_REFUSAL_PATTERNS = [
    (r"\b(ignore|disregard|forget)\b.{0,30}\b(instruction|rule|prompt|guardrail|above)\b",
     "That asks the assistant to drop its restrictions."),
    (r"\b(system prompt|your prompt|your instructions|jailbreak|developer mode)\b",
     "That asks about the assistant's configuration rather than food safety."),
    (r"\b(write|generate|create)\b.{0,25}\b(code|script|poem|story|essay|email|song)\b",
     "This assistant only answers food safety questions."),
    (r"\b(diagnos|prescrib|treat)\w*\b.{0,30}\b(me|my|patient|symptom|illness|disease)\b",
     "This is not a medical service and cannot give medical advice."),
    (r"\bhow (do|can) i\b.{0,40}\b(poison|harm|hurt|kill|contaminate deliberately)\b",
     "That asks how to cause harm."),
]

_MEDICAL_ADVICE = re.compile(
    r"\b(should i (eat|drink|take)|is it safe for me|am i (sick|poisoned)|my (symptom|doctor))\b",
    re.IGNORECASE,
)

_SYSTEM = """
You answer questions about food safety, food proteins, contaminants and the
regulations covering them. You have tools that look up real records.

Rules, without exception:
1. Call a tool. Do not answer a factual question from memory.
2. Report only what the tool returned. Never estimate a value or fill a gap.
3. If a tool returns an error or an empty result, say so plainly. Finding no
   evidence is a real answer.
4. Cite what you used: the accession, PMID, CID or recall number.
5. Never give medical advice or tell someone whether a food is safe for them
   personally. Report what the regulations and the published record say.
6. If the question is not about food safety, say that is outside what you cover.
Be brief. No preamble.
"""


def in_scope(question: str) -> tuple[bool, str]:
    """Decide whether a question belongs to this assistant, before any model call.

    Returns (allowed, reason). The reason is shown to the user when refused, so
    it explains rather than stonewalls.
    """
    text = (question or "").strip()
    if len(text) < 3:
        return False, "Ask a question about a food, a contaminant or a regulatory limit."

    lowered = text.lower()

    for pattern, reason in _REFUSAL_PATTERNS:
        if re.search(pattern, lowered):
            return False, reason

    if _MEDICAL_ADVICE.search(lowered):
        return False, (
            "This is not a medical service. It can tell you what the regulations "
            "and published research say, but not whether something is safe for you."
        )

    if not any(term in lowered for term in _IN_SCOPE_TERMS):
        return False, (
            "That looks outside food safety. Try asking about a contaminant, a food "
            "protein, a published limit, or what the research says about a pairing."
        )

    return True, ""


def build_agent(model_name: str = DEFAULT_MODEL):
    """One assistant holding every lookup tool."""
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm

    return LlmAgent(
        name="toxitrace_assistant",
        model=LiteLlm(
            model=f"ollama_chat/{model_name}",
            api_base=DEFAULT_BASE_URL,
            timeout=3600,
            keep_alive="30m",
        ),
        description="Answers food safety questions using public scientific records.",
        instruction=_SYSTEM,
        tools=adk_tools.ALL_TOOLS,
    )


def ask(question: str, model_name: str = DEFAULT_MODEL, user_id: str = "local") -> dict:
    """Answer a question, or refuse it. Never reaches the model when refused."""
    allowed, reason = in_scope(question)
    if not allowed:
        return {
            "question": question,
            "answer": reason,
            "refused": True,
            "tool_results": [],
            "ungrounded": [],
            "is_grounded": True,
        }

    import asyncio

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=build_agent(model_name), app_name="toxitrace")
    tool_results: list = []
    answer = ""

    async def _run() -> str:
        nonlocal answer
        session = await runner.session_service.create_session(
            app_name="toxitrace", user_id=user_id
        )
        message = types.Content(role="user", parts=[types.Part(text=question)])
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=message
        ):
            if not (event.content and event.content.parts):
                continue
            for part in event.content.parts:
                response = getattr(part, "function_response", None)
                if response is not None:
                    tool_results.append({"tool": response.name, "result": response.response})
                if getattr(part, "text", None):
                    answer = part.text
        return answer

    asyncio.run(_run())

    evidence = {"tool_results": tool_results}
    unsupported = grounding.check(answer, evidence) + grounding.check_identifiers(answer, evidence)

    return {
        "question": question,
        "answer": answer,
        "refused": False,
        "tool_results": tool_results,
        "ungrounded": unsupported,
        "is_grounded": not unsupported,
    }
