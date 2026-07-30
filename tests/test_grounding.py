"""Tests for the guardrail that catches invented numbers.

These encode the exact failure the predecessor shipped: a fabricated binding
constant sitting in otherwise-real output.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodsafe import grounding

EVIDENCE = {
    "structure": {"mean_plddt": 92.12, "length": 386, "accession": "P01012"},
    "chemistry": {"molecular_weight": 312.28, "logp": 2.28, "tpsa": 74.97},
    "compliance": {"measured": 25.0, "limit": 20.0},
    "citations": [{"pmid": "37062343"}, {"pmid": "34829017"}],
}


TOOL_RESULTS = {
    "tool_results": [
        {"tool": "describe_molecule",
         "result": {"status": "success",
                    "descriptors": {"molecular_weight": 312.28, "logp": 2.28, "tpsa": 74.97}}},
    ]
}

FABRICATED = "LogP: (Source: PubChem) Approximately 1.2"

ID_EVIDENCE = {
    "tool_results": [
        {"tool": "resolve_protein", "result": {"protein": {"accession": "P01012"}}},
        {"tool": "resolve_compound", "result": {"compound": {"cid": 186907}}},
        {"tool": "search_literature",
         "result": {"citations": [{"pmid": "37062343"}, {"pmid": "31745739"}]}},
    ]
}


def test_finds_decimals_and_integers():
    values = [v for _, v in grounding.extract_numbers("pLDDT 92.12 over 386 residues")]
    assert values == [92.12, 386.0]

def test_ignores_digits_inside_identifiers():
    """AF-P01012-F1 is an accession, not a measurement."""
    assert grounding.extract_numbers("structure AF-P01012-F1") == []

def test_handles_thousands_separators():
    values = [v for _, v in grounding.extract_numbers("1,000 ug/kg")]
    assert values == [1000.0]


def test_accepts_numbers_present_in_evidence():
    text = "Mean pLDDT is 92.12 across 386 residues; molecular weight 312.28."
    assert grounding.check(text, EVIDENCE) == []

def test_catches_fabricated_binding_constant():
    """The predecessor's exact failure mode."""
    text = "Binding affinity was Kd = 4.73 uM by surface plasmon resonance."
    findings = grounding.check(text, EVIDENCE)
    assert [f['value'] for f in findings] == [4.73]

def test_catches_invented_measurement_among_real_ones():
    text = "pLDDT 92.12 and molecular weight 312.28, with a risk score of 7.4 out of 10."
    findings = grounding.check(text, EVIDENCE)
    assert [f['value'] for f in findings] == [7.4]

def test_allows_rounding_of_real_values():
    assert grounding.check("confidence about 92.1", EVIDENCE) == []

def test_allows_prose_counting_by_default():
    assert grounding.check("The two citations agree.", EVIDENCE) == []

def test_strict_mode_flags_bare_integers():
    findings = grounding.check("There are 9 findings.", EVIDENCE, allow_small_integers=False)
    assert [f['value'] for f in findings] == [9.0]

def test_list_lengths_count_as_grounded():
    """'2 citations' is a true statement about the evidence."""
    assert grounding.check("Found 2 citations.", EVIDENCE, allow_small_integers=False) == []

def test_urls_do_not_launder_numbers():
    """A number must not become 'grounded' by appearing in a source URL."""
    evidence = {"provenance": {"url": "https://example.org/files/AF-9999-model_v6.pdb"}}
    findings = grounding.check("The measured level was 9999 ug/kg.", evidence)
    assert [f['value'] for f in findings] == [9999.0]

def test_finding_carries_surrounding_context():
    findings = grounding.check("Observed Kd = 4.73 uM in assay.", EVIDENCE)
    assert "Kd" in findings[0]['context']

def test_format_findings_is_readable():
    assert "trace back" in grounding.format_findings([])
    findings = grounding.check("Kd = 4.73 uM", EVIDENCE)
    assert "4.73" in grounding.format_findings(findings)


def test_collects_nested_values():
    numbers = grounding.evidence_numbers(EVIDENCE)
    assert {92.12, 386.0, 312.28, 25.0, 20.0} <= numbers

def test_booleans_are_not_numbers():
    """True must not ground the number 1."""
    assert grounding.evidence_numbers({"exceeds": True}) == set()


def test_agent_prose_launders_the_invented_value():
    """Reproduces the false negative: checking against prose lets 1.2 through."""
    laundered = {"agent_outputs": {"chemistry_findings": FABRICATED}}
    assert grounding.check(FABRICATED, laundered) == []

def test_tool_results_catch_it():
    findings = grounding.check(FABRICATED, TOOL_RESULTS)
    assert [f['value'] for f in findings] == [1.2]

def test_real_value_still_passes():
    assert grounding.check("LogP is 2.28", TOOL_RESULTS) == []


def test_catches_the_observed_corruption():
    found = grounding.check_identifiers("UniProt Accession: P0  012", ID_EVIDENCE)
    assert [f['text'] for f in found] == ["P0  012"]

def test_accepts_the_real_accession():
    assert grounding.check_identifiers("UniProt Accession: P01012", ID_EVIDENCE) == []

def test_catches_invented_pmid():
    found = grounding.check_identifiers("see PMID: 99999999", ID_EVIDENCE)
    assert [f['text'] for f in found] == ["99999999"]

def test_accepts_real_pmids():
    text = "reported in PMID: 37062343 and PMID 31745739"
    assert grounding.check_identifiers(text, ID_EVIDENCE) == []

def test_accepts_real_cid():
    assert grounding.check_identifiers("PubChem CID 186907", ID_EVIDENCE) == []

def test_numeric_check_misses_a_plausible_wrong_accession():
    """Why this check has to exist separately.

    P01013 is a well-formed accession for a different record. It contains no
    free-standing number, so the numeric check sees nothing wrong with it.
    """
    assert grounding.check("UniProt Accession: P01013", ID_EVIDENCE) == []
    assert [f['text'] for f in grounding.check_identifiers("UniProt Accession: P01013", ID_EVIDENCE)] == ["P01013"]

def test_numeric_check_mislabels_the_spaced_corruption():
    """The space makes "012" parse as a number, so it is caught but misdescribed.

    Reporting a corrupted accession as "the unsupported number 12" points a
    reader at the wrong problem; the identifier check names it correctly.
    """
    numeric = grounding.check("UniProt Accession: P0  012", ID_EVIDENCE)
    assert [f['value'] for f in numeric] == [12.0]
    assert [f['text'] for f in grounding.check_identifiers("UniProt Accession: P0  012", ID_EVIDENCE)] == ["P0  012"]

def test_no_evidence_identifiers_means_no_claims():
    assert grounding.check_identifiers("accession: P01012", {"tool_results": []}) == []
