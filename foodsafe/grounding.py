"""Check that a generated narrative invented no numbers.

The failure this project exists to prevent is a plausible-looking figure with no
source behind it. The deterministic layer cannot produce one. A language model
can, so anything it writes is checked back against the evidence: every number in
the narrative must appear in the gathered facts.

This is a guardrail, not a proof. It catches invented quantities, which is the
specific failure mode that made the predecessor untrustworthy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

# Numbers, including decimals and thousands separators, but not the digits
# embedded in identifiers like AF-P01012-F1.
_NUMBER = re.compile(r"(?<![\w.-])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(?![\w.-])")

_URL_KEYS = ("url", "pdb_url", "cif_url", "pae_image_url", "verify_url")

_RELATIVE_TOLERANCE = 0.005


@dataclass(frozen=True)
class Ungrounded:
    text: str
    value: float
    context: str

    def as_dict(self) -> dict:
        return {"text": self.text, "value": self.value, "context": self.context}


def extract_numbers(text: str) -> list[tuple[str, float]]:
    found = []
    for match in _NUMBER.finditer(text):
        raw = match.group(1)
        try:
            found.append((raw, float(raw.replace(",", ""))))
        except ValueError:
            continue
    return found


def _walk(node: Any, key: str | None = None) -> Iterable[float]:
    """Collect every number reachable in the evidence, ignoring URLs."""
    if key is not None and key in _URL_KEYS:
        return
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        yield float(node)
    elif isinstance(node, str):
        if node.startswith("http"):
            return
        for _, value in extract_numbers(node):
            yield value
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, k)
    elif isinstance(node, (list, tuple)):
        yield len(node)  # "3 citations" is a grounded statement about the evidence
        for item in node:
            yield from _walk(item, key)


def evidence_numbers(report) -> set[float]:
    data = report.as_dict() if hasattr(report, "as_dict") else report
    return set(_walk(data))


def _is_grounded(value: float, allowed: set[float]) -> bool:
    if value in allowed:
        return True
    for candidate in allowed:
        if candidate == 0:
            continue
        if abs(value - candidate) <= abs(candidate) * _RELATIVE_TOLERANCE:
            return True
        # A narrative may round 92.12 to 92.1, or 312.28 to 312.
        if round(candidate, 1) == value or round(candidate) == value:
            return True
    return False


def check(narrative: str, report, allow_small_integers: bool = True) -> list[Ungrounded]:
    """Return the numbers in `narrative` that the evidence does not support.

    `allow_small_integers` permits ordinary prose counting ("the three sources")
    without treating it as fabrication. Decimals and larger values are always
    checked, which is where invented measurements live.
    """
    allowed = evidence_numbers(report)
    findings = []

    for raw, value in extract_numbers(narrative):
        if allow_small_integers and value.is_integer() and 0 <= value <= 10:
            continue
        if _is_grounded(value, allowed):
            continue
        index = narrative.find(raw)
        context = narrative[max(0, index - 45): index + len(raw) + 45].replace("\n", " ")
        findings.append(Ungrounded(text=raw, value=value, context=context.strip()))

    return findings


# Identifiers are quoted, not computed, so a mangled one is a distinct failure
# from an invented number and the numeric check cannot see it. Observed in a real
# run: "UniProt Accession: P0  012" where the tool returned P01012.
_LABELLED_ID = re.compile(
    r"(?P<label>accession|PMID|CID|recall(?:\s+number)?)"
    r"[:\s#]*"
    r"(?P<value>[A-Za-z]?[A-Za-z0-9][A-Za-z0-9 \-]{2,18}?)"
    r"(?=[\s,.;)\]]|$)",
    re.IGNORECASE,
)

_ID_FIELDS = ("accession", "uniprot_accession", "pmid", "cid", "recall_number")


def _squash(value: str) -> str:
    return re.sub(r"[\s\-]+", "", str(value)).upper()


def evidence_identifiers(evidence) -> set[str]:
    """Every identifier the tools actually returned, whitespace-insensitive."""
    data = evidence.as_dict() if hasattr(evidence, "as_dict") else evidence
    found: set[str] = set()

    def walk(node, key=None):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item, key)
        elif node is not None and not isinstance(node, bool):
            if key in _ID_FIELDS:
                found.add(_squash(node))

    walk(data)
    return found


def check_identifiers(narrative: str, evidence) -> list[Ungrounded]:
    """Flag accessions, PMIDs and CIDs that the tools never returned.

    A transcription error in an identifier is as misleading as an invented
    number - it points a reader at the wrong record - but it carries no
    arithmetic for the numeric check to catch.
    """
    known = evidence_identifiers(evidence)
    if not known:
        return []

    findings = []
    for match in _LABELLED_ID.finditer(narrative):
        raw = match.group("value").strip()
        squashed = _squash(raw)
        if not squashed or squashed in known:
            continue
        start = match.start()
        context = narrative[max(0, start - 30): match.end() + 30].replace("\n", " ")
        findings.append(Ungrounded(text=raw, value=float("nan"), context=context.strip()))

    return findings


def format_findings(findings: list[Ungrounded]) -> str:
    if not findings:
        return "All numbers in the narrative trace back to the gathered evidence."
    lines = [f"{len(findings)} unsupported number(s) found in the generated text:"]
    for f in findings:
        lines.append(f"  {f.text!r} -> ...{f.context}...")
    return "\n".join(lines)
