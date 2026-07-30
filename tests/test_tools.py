"""Offline tests for deterministic logic, plus opt-in live API smoke tests.

Run everything:            python -m pytest tests/
Skip the network tests:    python -m pytest tests/ -m "not live"
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodsafe.tools import chem, regulatory

# Structures are stable public facts, so these are safe to pin.
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
AFLATOXIN_B1 = "COC1=C2C3=C(C(=O)CC3)C(=O)OC2=C4C5C=COC5OC4=C1"


def test_caffeine_matches_published_values():
    d = chem.describe(CAFFEINE)["value"]
    assert d["formula"] == "C8H10N4O2"
    assert d["molecular_weight"] == pytest.approx(194.19, abs=0.02)
    assert d["lipinski_violations"] == 0

def test_aflatoxin_b1_matches_pubchem():
    """RDKit computing 312.28 from SMILES independently corroborates PubChem's 312.27."""
    d = chem.describe(AFLATOXIN_B1)["value"]
    assert d["formula"] == "C17H12O6"
    assert d["molecular_weight"] == pytest.approx(312.27, abs=0.05)

def test_descriptors_are_deterministic():
    """The predecessor drew these from np.random; identical input must repeat."""
    first = chem.describe(AFLATOXIN_B1)["value"]
    second = chem.describe(AFLATOXIN_B1)["value"]
    assert first == second

def test_invalid_smiles_raises_rather_than_inventing():
    with pytest.raises(chem.InvalidStructure):
        chem.describe("not-a-molecule")

def test_carries_provenance():
    assert "RDKit" in chem.describe(CAFFEINE)["provenance"]["source"]

def test_svg_depiction_renders():
    svg = chem.to_svg(CAFFEINE)
    assert svg.lstrip().startswith("<?xml") or "<svg" in svg


def test_finds_limits_across_jurisdictions():
    limits = regulatory.find_limits("aflatoxins, total")
    assert {l["jurisdiction"] for l in limits} >= {"EU", "US", "IN"}

def test_comparison_flags_exceedance():
    us = [c for c in regulatory.compare(25.0, "aflatoxins, total") if c["limit"]["jurisdiction"] == "US"][0]
    assert us["exceeds"] is True
    assert us["ratio_of_limit"] == pytest.approx(1.25)

def test_comparison_passes_when_within_limit():
    us = [c for c in regulatory.compare(5.0, "aflatoxins, total") if c["limit"]["jurisdiction"] == "US"][0]
    assert us["exceeds"] is False

def test_boundary_value_is_not_an_exceedance():
    """At exactly the maximum level a sample is compliant, not in breach."""
    us = [c for c in regulatory.compare(20.0, "aflatoxins, total") if c["limit"]["jurisdiction"] == "US"][0]
    assert us["exceeds"] is False

def test_negative_measurement_rejected():
    with pytest.raises(ValueError):
        regulatory.compare(-1.0, "aflatoxin M1")

def test_statement_cites_its_instrument():
    c = regulatory.compare(25.0, "aflatoxins, total", jurisdiction="EU")[0]
    assert "2023/915" in regulatory.statement(c)
    assert c["limit"]["verify_url"].startswith("http")

def test_every_bundled_limit_has_a_citable_source():
    data = json.loads((Path(__file__).resolve().parents[1] / "foodsafe/data/mycotoxin_limits.json").read_text())
    for row in data["limits"]:
        meta = data["jurisdictions"][row["jurisdiction"]]
        assert meta["instrument"] and meta["url"].startswith("http")
        assert row["max_level_ug_per_kg"] > 0



def test_pubchem_resolves_aflatoxin():
    from foodsafe.tools import pubchem

    c = pubchem.resolve_compound("aflatoxin B1")["value"]
    assert c["cid"] == 186907
    assert c["formula"] == "C17H12O6"

def test_uniprot_resolves_ovalbumin():
    from foodsafe.tools import uniprot

    p = uniprot.resolve_protein("ovalbumin", organism="Gallus gallus")["value"]
    assert p["accession"] == "P01012"
    assert p["reviewed"] is True

def test_alphafold_structure_matches_uniprot_length():
    from foodsafe.tools import alphafold

    s = alphafold.fetch_structure("P01012")
    pdb = alphafold.fetch_pdb(s["value"])
    plddt = alphafold.per_residue_plddt(pdb)
    assert len(plddt) == 386, "residue count must match the UniProt sequence"
    assert abs(sum(plddt) / len(plddt) - s["value"]["mean_plddt"]) < 0.5

def test_plddt_bands_match_alphafold_reported_fractions():
    """The viewer colours residues by B-factor; those bands must be the real ones.

    Independently banding the parsed per-residue pLDDT has to reproduce the
    fractions AlphaFold publishes for the same entry, or the structure is
    being coloured by something other than genuine confidence.
    """
    from foodsafe.tools import alphafold

    structure = alphafold.fetch_structure("P01012")
    plddt = alphafold.per_residue_plddt(alphafold.fetch_pdb(structure["value"]))
    total = len(plddt)

    fraction = lambda lo, hi: sum(1 for b in plddt if lo < b <= hi) / total  # noqa: E731
    reported = structure["value"]

    assert fraction(90, 1000) == pytest.approx(reported["fraction_very_high"], abs=0.02)
    assert fraction(70, 90) == pytest.approx(reported["fraction_confident"], abs=0.02)
    assert fraction(50, 70) == pytest.approx(reported["fraction_low"], abs=0.02)
    assert fraction(0, 50) == pytest.approx(reported["fraction_very_low"], abs=0.02)

def test_pubmed_returns_real_citations_only():
    from foodsafe.tools import literature

    cites = literature.interaction_evidence("ovalbumin", "aflatoxin B1", max_results=3)["value"]
    for c in cites:
        assert c["pmid"].isdigit()
        assert c["url"].endswith(f"/{c['pmid']}/")
        # No fabricated experimental fields may appear on a citation.
        assert "binding_affinity" not in c

def test_pubmed_empty_result_is_not_padded():
    """A nonsense query must yield nothing rather than an invented record."""
    from foodsafe.tools import literature

    cites = literature.search("zzzqqqxyzzy nonexistent compound 12345", max_results=5)["value"]
    assert cites == []



def test_placeholder_phrases_mean_no_filter():
    from foodsafe.tools import regulatory

    for phrase in ["all jurisdictions", "all", "", "  ", "none", "not provided", "any"]:
        assert regulatory.normalise_jurisdiction(phrase) is None

def test_known_codes_are_case_insensitive():
    from foodsafe.tools import regulatory

    assert regulatory.normalise_jurisdiction("eu") == "EU"
    assert regulatory.normalise_jurisdiction(" in ") == "IN"

def test_unknown_code_raises_rather_than_matching_nothing():
    """A typo must not look like an absence of regulation."""
    from foodsafe.tools import regulatory

    with pytest.raises(regulatory.UnknownJurisdiction):
        regulatory.normalise_jurisdiction("XX")

def test_the_failing_agent_call_now_succeeds():
    from foodsafe.adk_tools import compare_to_regulatory_limits

    result = compare_to_regulatory_limits("aflatoxins, total", 25.0, "all jurisdictions")
    assert result["status"] == "success"
    assert len(result["comparisons"]) == 3
