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

## Agents

Five specialists run as a Google ADK `SequentialAgent`, each holding only the tools its
role needs — structural biologist, chemist, literature analyst, compliance officer, and a
reporter that synthesises their findings. The model runs locally through LiteLLM to
Ollama, so nothing leaves the machine.

```bash
ollama pull qwen3:4b          # any tool-calling model works
export FOODSAFE_MODEL=qwen3:4b
```

```python
from foodsafe.agents import analyse

result = analyse("Assess ovalbumin against aflatoxin B1; a groundnut sample measured 25 ug/kg.")
print(result.report)
print(result.is_grounded)        # False if the model introduced an unsupported number
```

### The grounding check

A language model can invent a figure where the deterministic layer cannot, so every
number the reporter writes is checked back against the evidence the agents actually
gathered:

```python
grounding.check("Binding affinity was Kd = 4.73 uM by surface plasmon resonance.", evidence)
# [Ungrounded(text='4.73', value=4.73, context='Binding affinity was Kd = 4.73 uM by ...')]
```

That is the predecessor's exact failure, caught automatically. Numbers cannot be
laundered into looking supported: booleans do not ground `1`, digits inside identifiers
like `AF-P01012-F1` are not measurements, and a number appearing in a source URL does not
count as evidence for it.

This is a guardrail, not a proof. It catches invented quantities, which is the specific
failure mode that made the predecessor untrustworthy.

### Choosing a local model

Parameter count and marketing copy do not predict whether a model will call a tool with
the right arguments and then report the result without embellishing it. `scripts/bench_models.py`
measures it:

```bash
python scripts/bench_models.py                       # every installed model
python scripts/bench_models.py --models gemma4:26b
```

Measured on an RTX 3050 6GB / 32GB RAM:

| model | size | Ollama tool support | emits tool calls | drives the pipeline | seconds |
|---|---|---|---|---|---|
| `gemma4:26b` | 18GB | accepts | **yes** | **yes** | 33.3 |
| `phi4-mini` | 2.5GB | accepts | no | no | 27.0 |
| `gemma3n:e4b` | 7.5GB | **rejects** | – | no | 3.1 |
| `gemma4:latest` | 9.6GB | rejects `/api/chat` | – | no | – |

Only one of four installed models can actually drive this pipeline, and it is the largest.

The two failures fail differently, which is worth separating:

- `gemma3n:e4b` is refused by Ollama outright — `"does not support tools"`. Its chat
  template has no tool-calling support, so no amount of prompting reaches the model.
- `phi4-mini` **is** accepted by Ollama with tools attached and still emitted no tool
  call, even for an explicit `"Call t with x=1"`. That makes it model behaviour rather
  than a packaging gap — worth stating plainly, since function calling is the headline
  feature it is marketed on for edge devices.

The going-in assumption, that a small model advertised for function calling would be the
fast choice, was wrong in the only way that mattered. Measuring it took less time than
arguing about it would have.

When swapping models, the column to watch in the benchmark output is `invented numbers`:
values in the reply that the tool never returned.

## Web viewer

```bash
python -m uvicorn web.app:app --reload
# http://127.0.0.1:8000
```

The protein is rendered with 3Dmol.js and coloured by the **real per-residue pLDDT**
parsed from the B-factor column of the AlphaFold PDB, using AlphaFold's published
confidence bands. The contaminant is drawn by RDKit from the PubChem structure.
Regulatory comparisons show the measured value against the published maximum with a link
to the primary legal text, and every source consulted is listed at the bottom of the page.

Evidence rendering does not require a language model — facts first, interpretation second.

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
