"""Literature retrieval from PubMed.

This module deliberately returns *only* what PubMed actually published: title,
authors, journal, year, PMID, DOI. The predecessor took real search results and
attached invented binding constants to them --

    'binding_affinity': f"Kd = {np.random.uniform(0.1, 10.0):.2f} uM",
    'experimental_method': 'Surface plasmon resonance',

-- which attributes fabricated measurements to real, citable papers. Nothing
here invents an experimental value. If a binding constant is wanted, a human
reads the paper.
"""

from ..provenance import PUBMED, sourced
from ._http import get_json

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_ARTICLE = "https://pubmed.ncbi.nlm.nih.gov"


def format_citation(citation: dict) -> str:
    lead = citation["authors"][0] if citation["authors"] else "Unknown"
    et_al = " et al." if len(citation["authors"]) > 1 else ""
    year = citation.get("year") or "n.d."
    return f"{lead}{et_al} ({year}). {citation['title']} {citation['journal']}. PMID:{citation['pmid']}"


def _to_citation(pmid: str, record: dict) -> dict:
    doi = next(
        (a.get("value") for a in record.get("articleids", []) if a.get("idtype") == "doi"),
        None,
    )
    pub_date = record.get("pubdate", "")
    return {
        "pmid": pmid,
        "title": record.get("title", "").rstrip("."),
        "authors": [a.get("name", "") for a in record.get("authors", [])][:6],
        "journal": record.get("source", ""),
        "year": pub_date.split(" ")[0] if pub_date else None,
        "doi": doi,
        "url": f"{_ARTICLE}/{pmid}/",
    }


def search(query: str, max_results: int = 10) -> dict:
    """Search PubMed and return real citations, or an empty list."""
    found = get_json(
        f"{_EUTILS}/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmode": "json", "retmax": max_results},
    )
    pmids = found.get("esearchresult", {}).get("idlist", [])
    search_url = f"{_ARTICLE}/?term={query.replace(' ', '+')}"

    if not pmids:
        return sourced([], PUBMED, search_url)

    summaries = get_json(
        f"{_EUTILS}/esummary.fcgi",
        params={"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
    ).get("result", {})

    citations = [
        _to_citation(pmid, summaries[pmid]) for pmid in pmids if summaries.get(pmid)
    ]
    return sourced(citations, PUBMED, search_url)


def interaction_evidence(protein: str, contaminant: str, max_results: int = 10) -> dict:
    """Find published work on a specific protein/contaminant pair.

    Returns the literature that exists. An empty result means no indexed
    evidence was found -- which is itself a finding, not a prompt to invent one.
    """
    query = f'("{protein}"[All Fields]) AND ("{contaminant}"[All Fields])'
    return search(query, max_results=max_results)
