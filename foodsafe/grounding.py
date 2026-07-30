"""Check that a generated narrative invented no numbers or identifiers.

The failure this project exists to prevent is a plausible-looking figure with no
source behind it. The deterministic layer cannot produce one. A language model
can, so anything it writes is checked back against the evidence.

Two checks, because two things go wrong differently:

  check()             numbers must appear in the tool results
  check_identifiers() accessions, PMIDs and CIDs must match what was returned

These are guardrails, not proofs. They catch invented quantities and misquoted
records, which are the specific failures that made the predecessor untrustworthy.
"""

import re

# Numbers, including decimals and thousands separators, but not the digits
# embedded in identifiers like AF-P01012-F1.
_NUMBER = re.compile(r"(?<![\w.-])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(?![\w.-])")

# Identifiers are quoted, not computed, so a mangled one is a distinct failure
# from an invented number. Observed in a real run: "UniProt Accession: P0  012"
# where the tool returned P01012.
_LABELLED_ID = re.compile(
    r"(?P<label>accession|PMID|CID|recall(?:\s+number)?)"
    r"[:\s#]*"
    r"(?P<value>[A-Za-z]?[A-Za-z0-9][A-Za-z0-9 \-]{2,18}?)"
    r"(?=[\s,.;)\]]|$)",
    re.IGNORECASE,
)

_URL_KEYS = ("url", "pdb_url", "cif_url", "pae_image_url", "verify_url")
_ID_FIELDS = ("accession", "uniprot_accession", "pmid", "cid", "recall_number")

_RELATIVE_TOLERANCE = 0.005


def _finding(text: str, value: float, narrative: str, start: int, end: int) -> dict:
    context = narrative[max(0, start - 40): end + 40].replace("\n", " ")
    return {"text": text, "value": value, "context": context.strip()}


def extract_numbers(text: str) -> list[tuple[str, float]]:
    found = []
    for match in _NUMBER.finditer(text):
        raw = match.group(1)
        try:
            found.append((raw, float(raw.replace(",", ""))))
        except ValueError:
            continue
    return found


def _walk_numbers(node, key=None):
    """Collect every number reachable in the evidence, ignoring URLs."""
    if key in _URL_KEYS or isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        yield float(node)
    elif isinstance(node, str):
        if not node.startswith("http"):
            for _, value in extract_numbers(node):
                yield value
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _walk_numbers(v, k)
    elif isinstance(node, (list, tuple)):
        yield len(node)  # "3 citations" is a grounded statement about the evidence
        for item in node:
            yield from _walk_numbers(item, key)


def evidence_numbers(evidence: dict) -> set[float]:
    return set(_walk_numbers(evidence))


def _squash(value) -> str:
    return re.sub(r"[\s\-]+", "", str(value)).upper()


def _walk_identifiers(node, key=None):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk_identifiers(v, k)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk_identifiers(item, key)
    elif node is not None and not isinstance(node, bool) and key in _ID_FIELDS:
        yield _squash(node)


def evidence_identifiers(evidence: dict) -> set[str]:
    """Every identifier the tools actually returned, whitespace-insensitive."""
    return set(_walk_identifiers(evidence))


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


def check(narrative: str, evidence: dict, allow_small_integers: bool = True) -> list[dict]:
    """Return the numbers in `narrative` that the evidence does not support.

    `allow_small_integers` permits ordinary prose counting ("the three sources")
    without treating it as fabrication. Decimals and larger values are always
    checked, which is where invented measurements live.
    """
    allowed = evidence_numbers(evidence)
    findings = []

    for match in _NUMBER.finditer(narrative):
        raw = match.group(1)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if allow_small_integers and value.is_integer() and 0 <= value <= 10:
            continue
        if _is_grounded(value, allowed):
            continue
        findings.append(_finding(raw, value, narrative, match.start(), match.end()))

    return findings


def check_identifiers(narrative: str, evidence: dict) -> list[dict]:
    """Flag accessions, PMIDs and CIDs that the tools never returned.

    A transcription error in an identifier is as misleading as an invented
    number -- it points a reader at the wrong record -- but it carries no
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
        findings.append(
            _finding(raw, float("nan"), narrative, match.start(), match.end())
        )
    return findings


def format_findings(findings: list[dict]) -> str:
    if not findings:
        return "All numbers in the narrative trace back to the gathered evidence."
    lines = [f"{len(findings)} unsupported value(s) found in the generated text:"]
    for finding in findings:
        lines.append(f"  {finding['text']!r} -> ...{finding['context']}...")
    return "\n".join(lines)
