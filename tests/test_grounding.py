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


class TestExtraction:
    def test_finds_decimals_and_integers(self):
        values = [v for _, v in grounding.extract_numbers("pLDDT 92.12 over 386 residues")]
        assert values == [92.12, 386.0]

    def test_ignores_digits_inside_identifiers(self):
        """AF-P01012-F1 is an accession, not a measurement."""
        assert grounding.extract_numbers("structure AF-P01012-F1") == []

    def test_handles_thousands_separators(self):
        values = [v for _, v in grounding.extract_numbers("1,000 ug/kg")]
        assert values == [1000.0]


class TestGroundingCheck:
    def test_accepts_numbers_present_in_evidence(self):
        text = "Mean pLDDT is 92.12 across 386 residues; molecular weight 312.28."
        assert grounding.check(text, EVIDENCE) == []

    def test_catches_fabricated_binding_constant(self):
        """The predecessor's exact failure mode."""
        text = "Binding affinity was Kd = 4.73 uM by surface plasmon resonance."
        findings = grounding.check(text, EVIDENCE)
        assert [f.value for f in findings] == [4.73]

    def test_catches_invented_measurement_among_real_ones(self):
        text = "pLDDT 92.12 and molecular weight 312.28, with a risk score of 7.4 out of 10."
        findings = grounding.check(text, EVIDENCE)
        assert [f.value for f in findings] == [7.4]

    def test_allows_rounding_of_real_values(self):
        assert grounding.check("confidence about 92.1", EVIDENCE) == []

    def test_allows_prose_counting_by_default(self):
        assert grounding.check("The two citations agree.", EVIDENCE) == []

    def test_strict_mode_flags_bare_integers(self):
        findings = grounding.check("There are 9 findings.", EVIDENCE, allow_small_integers=False)
        assert [f.value for f in findings] == [9.0]

    def test_list_lengths_count_as_grounded(self):
        """'2 citations' is a true statement about the evidence."""
        assert grounding.check("Found 2 citations.", EVIDENCE, allow_small_integers=False) == []

    def test_urls_do_not_launder_numbers(self):
        """A number must not become 'grounded' by appearing in a source URL."""
        evidence = {"provenance": {"url": "https://example.org/files/AF-9999-model_v6.pdb"}}
        findings = grounding.check("The measured level was 9999 ug/kg.", evidence)
        assert [f.value for f in findings] == [9999.0]

    def test_finding_carries_surrounding_context(self):
        findings = grounding.check("Observed Kd = 4.73 uM in assay.", EVIDENCE)
        assert "Kd" in findings[0].context

    def test_report_object_with_as_dict_is_accepted(self):
        class Report:
            def as_dict(self):
                return EVIDENCE

        assert grounding.check("pLDDT 92.12", Report()) == []

    def test_format_findings_is_readable(self):
        assert "trace back" in grounding.format_findings([])
        findings = grounding.check("Kd = 4.73 uM", EVIDENCE)
        assert "4.73" in grounding.format_findings(findings)


class TestEvidenceNumbers:
    def test_collects_nested_values(self):
        numbers = grounding.evidence_numbers(EVIDENCE)
        assert {92.12, 386.0, 312.28, 25.0, 20.0} <= numbers

    def test_booleans_are_not_numbers(self):
        """True must not ground the number 1."""
        assert grounding.evidence_numbers({"exceeds": True}) == set()


class TestUpstreamFabrication:
    """A specialist that invents a value must not be laundered by the reporter.

    Grounding the final report against the agents' own prose only verifies the
    last hop. If the chemist fabricates a logP the reporter copies it verbatim
    and the check passes. Observed in a real run: the report carried
    "LogP ... approximately 1.2" while RDKit had returned 2.28.
    """

    TOOL_RESULTS = {
        "tool_results": [
            {"tool": "describe_molecule",
             "result": {"status": "success",
                        "descriptors": {"molecular_weight": 312.28, "logp": 2.28, "tpsa": 74.97}}},
        ]
    }

    FABRICATED = "LogP: (Source: PubChem) Approximately 1.2"

    def test_agent_prose_launders_the_invented_value(self):
        """Reproduces the false negative: checking against prose lets 1.2 through."""
        laundered = {"agent_outputs": {"chemistry_findings": self.FABRICATED}}
        assert grounding.check(self.FABRICATED, laundered) == []

    def test_tool_results_catch_it(self):
        findings = grounding.check(self.FABRICATED, self.TOOL_RESULTS)
        assert [f.value for f in findings] == [1.2]

    def test_real_value_still_passes(self):
        assert grounding.check("LogP is 2.28", self.TOOL_RESULTS) == []


class TestIdentifierGrounding:
    """Identifiers are quoted, not computed, so the numeric check cannot see them.

    Observed in a real run: the report said "UniProt Accession: P0  012" while
    resolve_protein had returned P01012. A mangled accession points a reader at
    the wrong record, but carries no arithmetic to catch.
    """

    EVIDENCE = {
        "tool_results": [
            {"tool": "resolve_protein", "result": {"protein": {"accession": "P01012"}}},
            {"tool": "resolve_compound", "result": {"compound": {"cid": 186907}}},
            {"tool": "search_literature",
             "result": {"citations": [{"pmid": "37062343"}, {"pmid": "31745739"}]}},
        ]
    }

    def test_catches_the_observed_corruption(self):
        found = grounding.check_identifiers("UniProt Accession: P0  012", self.EVIDENCE)
        assert [f.text for f in found] == ["P0  012"]

    def test_accepts_the_real_accession(self):
        assert grounding.check_identifiers("UniProt Accession: P01012", self.EVIDENCE) == []

    def test_catches_invented_pmid(self):
        found = grounding.check_identifiers("see PMID: 99999999", self.EVIDENCE)
        assert [f.text for f in found] == ["99999999"]

    def test_accepts_real_pmids(self):
        text = "reported in PMID: 37062343 and PMID 31745739"
        assert grounding.check_identifiers(text, self.EVIDENCE) == []

    def test_accepts_real_cid(self):
        assert grounding.check_identifiers("PubChem CID 186907", self.EVIDENCE) == []

    def test_numeric_check_misses_a_plausible_wrong_accession(self):
        """Why this check has to exist separately.

        P01013 is a well-formed accession for a different record. It contains no
        free-standing number, so the numeric check sees nothing wrong with it.
        """
        assert grounding.check("UniProt Accession: P01013", self.EVIDENCE) == []
        assert [f.text for f in grounding.check_identifiers("UniProt Accession: P01013", self.EVIDENCE)] == ["P01013"]

    def test_numeric_check_mislabels_the_spaced_corruption(self):
        """The space makes "012" parse as a number, so it is caught but misdescribed.

        Reporting a corrupted accession as "the unsupported number 12" points a
        reader at the wrong problem; the identifier check names it correctly.
        """
        numeric = grounding.check("UniProt Accession: P0  012", self.EVIDENCE)
        assert [f.value for f in numeric] == [12.0]
        assert [f.text for f in grounding.check_identifiers("UniProt Accession: P0  012", self.EVIDENCE)] == ["P0  012"]

    def test_no_evidence_identifiers_means_no_claims(self):
        assert grounding.check_identifiers("accession: P01012", {"tool_results": []}) == []
