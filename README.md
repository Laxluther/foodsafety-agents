# foodsafety-agents

Multi-agent food safety analysis where **every reported value is traceable to a real source**.

Protein structures come from Google DeepMind's AlphaFold database, molecular descriptors
are computed with RDKit, literature comes from PubMed, and contamination levels are
compared against published regulatory limits. Nothing is simulated, estimated, or
generated to look plausible.

## Why this exists

This is a rewrite. The predecessor advertised ESMFold structure prediction, RDKit
docking, and FDA/EFSA compliance checking — but underneath:

```python
# protein structure "prediction"
'coordinates': np.random.randn(length, 3).tolist()

# binding affinity, with noise added to look convincing
noise = np.random.normal(0, 0.5)

# fabricated experimental values attached to real PubMed papers
'binding_affinity': f"Kd = {np.random.uniform(0.1, 10.0):.2f} uM",
'experimental_method': 'Surface plasmon resonance',
```

That last one is the reason for the rewrite. Attaching invented binding constants and
a fake experimental method to real, citable papers is not a shortcut — it is fabricated
evidence, and in a food safety context that matters.

The agent orchestration was always the interesting part. It survives. The invented
science does not.

### What changed

| Predecessor | This version |
|---|---|
| `np.random.randn(n, 3)` as atomic coordinates | Real AlphaFold coordinates, fetched by UniProt accession |
| `np.random.uniform(200, 800)` as molecular weight | RDKit, computed from the PubChem structure |
| Heuristic binding affinity + random noise | Removed — not computable without docking software and validation |
| Random `Kd` values on real citations | Citations only: title, authors, journal, PMID, DOI |
| Synthesised 0–10 "risk score" | Measured value compared against a published legal limit |
| Optional deps with random fallbacks | RDKit is a hard dependency; a missing source raises |

## Design rule

Every value is wrapped in `Sourced`, which carries a `Provenance` record — source name,
URL, licence, retrieval timestamp. A number without a traceable origin cannot be
constructed by accident.

```python
from foodsafe.tools import pubchem, chem

compound = pubchem.resolve_compound("aflatoxin B1")
descriptors = chem.describe(compound.value.smiles)

descriptors.value.molecular_weight   # 312.28
descriptors.provenance.source        # 'Computed with RDKit 2026.03.4'
```

## Data sources

| Source | Used for | Licence |
|---|---|---|
| [AlphaFold DB](https://alphafold.ebi.ac.uk/) (Google DeepMind / EMBL-EBI) | Protein structures, per-residue pLDDT | CC-BY-4.0 |
| [UniProtKB](https://www.uniprot.org/) | Protein identity and sequence | CC-BY-4.0 |
| [PubChem](https://pubchem.ncbi.nlm.nih.gov/) | Compound structures (SMILES) | Public domain |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov/) | Literature | NCBI usage policy |
| [openFDA](https://open.fda.gov/) | Food enforcement / recall records | [openFDA licence](https://open.fda.gov/license/) |
| [RDKit](https://www.rdkit.org/) | Molecular descriptors | BSD-3-Clause |

Structures are looked up rather than folded locally. AlphaFold DB holds 200M+ precomputed
predictions, so fetching the real one is both better science and lighter than running a
fold on a laptop GPU.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Use

```python
from foodsafe.tools import uniprot, alphafold, literature, regulatory

protein = uniprot.resolve_protein("ovalbumin", organism="Gallus gallus")
# P01012, 386 aa, reviewed (Swiss-Prot)

structure = alphafold.fetch_structure(protein.value.accession)
# mean pLDDT 92.12 — "very high (backbone and side chains reliable)"

evidence = literature.interaction_evidence("ovalbumin", "aflatoxin B1")
# real citations, or an empty list if nothing is indexed

for c in regulatory.compare(25.0, "aflatoxins, total"):
    print(c.statement())
# 25.0 ug/kg exceeds the United States maximum of 20.0 ug/kg for
# aflatoxins, total (B1+B2+G1+G2) in all foods except milk (FDA action levels...)
```

## Tests

```bash
python -m pytest tests/               # includes live API checks
python -m pytest tests/ -m "not live" # offline only
```

The live tests assert cross-source consistency: the AlphaFold PDB must contain exactly as
many residues as the UniProt sequence (386 for ovalbumin), and RDKit's molecular weight
computed from SMILES must agree with PubChem's reported value.

## Limits and honest caveats

- **The bundled regulatory table is a curated subset.** Each entry cites its legal
  instrument and links to the primary text. Regulations are amended — verify before any
  operational use. See `foodsafe/data/mycotoxin_limits.json`.
- **No binding affinity prediction.** Credible docking needs proper software and
  experimental validation. An invented number is worse than no number.
- **This is a research and educational tool.** It is not a regulatory instrument and
  produces no compliance certification.
- Structures carry pLDDT confidence for a reason. Low-confidence regions are predictions,
  not measurements.

## Licence

Apache-2.0. See [LICENSE](LICENSE).

Structural data from the AlphaFold Protein Structure Database is CC-BY-4.0 —
Jumper et al., *Highly accurate protein structure prediction with AlphaFold*,
Nature (2021); Varadi et al., *AlphaFold Protein Structure Database*, NAR (2024).
