"""Literature retrieval from PubMed.

This module deliberately returns *only* what PubMed actually published: title,
authors, journal, year, PMID, DOI. The predecessor took real search results and
attached invented binding constants to them —

    'binding_affinity': f"Kd = {np.random.uniform(0.1, 10.0):.2f} uM",
    'experimental_method': 'Surface plasmon resonance',

— which attributes fabricated measurements to real, citable papers. Nothing
here invents an experimental value. If a binding constant is wanted, a human
reads the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..provenance import PUBMED, Provenance, Sourced
from ._http import get_json

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_ARTICLE = "https://pubmed.ncbi.nlm.nih.gov"


@dataclass(frozen=True)
class Citation:
    pmid: str
    title: str
    authors: list[str]
    journal: str
    year: str | None
    doi: str | None
    url: str

    def as_dict(self) -> dict:
        return asdict(self)

    def formatted(self) -> str:
        lead = self.authors[0] if self.authors else "Unknown"
        et_al = " et al." if len(self.authors) > 1 else ""
        return f"{lead}{et_al} ({self.year or 'n.d.'}). {self.title} {self.journal}. PMID:{self.pmid}"


def search(query: str, max_results: int = 10) -> Sourced[list[Citation]]:
    """Search PubMed and return real citations, or an empty list."""
    found = get_json(
        f"{_EUTILS}/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmode": "json", "retmax": max_results},
    )
    pmids = found.get("esearchresult", {}).get("idlist", [])

    search_url = f"{_ARTICLE}/?term={query.replace(' ', '+')}"
    if not pmids:
        return Sourced([], Provenance(url=search_url, **PUBMED))

    summaries = get_json(
        f"{_EUTILS}/esummary.fcgi",
        params={"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
    ).get("result", {})

    citations = []
    for pmid in pmids:
        record = summaries.get(pmid)
        if not record:
            continue
        doi = next(
            (a.get("value") for a in record.get("articleids", []) if a.get("idtype") == "doi"),
            None,
        )
        pub_date = record.get("pubdate", "")
        citations.append(
            Citation(
                pmid=pmid,
                title=record.get("title", "").rstrip("."),
                authors=[a.get("name", "") for a in record.get("authors", [])][:6],
                journal=record.get("source", ""),
                year=pub_date.split(" ")[0] if pub_date else None,
                doi=doi,
                url=f"{_ARTICLE}/{pmid}/",
            )
        )

    return Sourced(citations, Provenance(url=search_url, **PUBMED))


def interaction_evidence(protein: str, contaminant: str, max_results: int = 10) -> Sourced[list[Citation]]:
    """Find published work on a specific protein/contaminant pair.

    Returns the literature that exists. An empty result means no indexed
    evidence was found — which is itself a finding, not a prompt to invent one.
    """
    query = f'("{protein}"[All Fields]) AND ("{contaminant}"[All Fields])'
    return search(query, max_results=max_results)
