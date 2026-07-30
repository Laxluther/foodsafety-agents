"use strict";

const $ = (id) => document.getElementById(id);

// AlphaFold's published pLDDT confidence bands and their standard colours.
const PLDDT_BANDS = [
  { min: 90, colour: "#0053D6" },
  { min: 70, colour: "#65CBF3" },
  { min: 50, colour: "#FFDB13" },
  { min: 0, colour: "#FF7D45" },
];

const plddtColour = (b) => PLDDT_BANDS.find((band) => b > band.min).colour;

function show(id) { $(id).hidden = false; }
function hide(id) { $(id).hidden = true; }

function facts(target, rows) {
  $(target).innerHTML = rows
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`)
    .join("");
}

let viewer = null;

async function renderStructure(structure) {
  show("structure-panel");
  const accession = structure.uniprot_accession;

  facts("structure-facts", [
    ["Accession", `<a href="https://www.uniprot.org/uniprotkb/${accession}" target="_blank" rel="noopener">${accession}</a>`],
    ["Description", structure.description],
    ["Organism", `<i>${structure.organism}</i>`],
    ["Mean pLDDT", `<strong>${structure.mean_plddt}</strong> — ${structure.confidence_band}`],
    ["Very high (&gt;90)", `${(structure.fraction_very_high * 100).toFixed(1)}%`],
    ["Confident (70–90)", `${(structure.fraction_confident * 100).toFixed(1)}%`],
    ["Low (50–70)", `${(structure.fraction_low * 100).toFixed(1)}%`],
    ["Very low (&lt;50)", `${(structure.fraction_very_low * 100).toFixed(1)}%`],
    ["Model version", `v${structure.model_version}`],
  ]);

  const pdb = await fetch(`/api/structure/${accession}.pdb`).then((r) => {
    if (!r.ok) throw new Error(`structure fetch failed (${r.status})`);
    return r.text();
  });

  const host = $("viewer");
  host.innerHTML = "";
  viewer = $3Dmol.createViewer(host, { backgroundColor: "white" });
  viewer.addModel(pdb, "pdb");
  viewer.setStyle({}, { cartoon: { colorfunc: (atom) => plddtColour(atom.b) } });
  viewer.zoomTo();
  viewer.render();
}

function renderCompound(compound, descriptors) {
  show("compound-panel");
  $("molecule").innerHTML =
    `<img alt="2D structure of ${compound.name}" src="/api/molecule.svg?smiles=${encodeURIComponent(compound.smiles)}">`;

  const rows = [
    ["PubChem CID", `<a href="https://pubchem.ncbi.nlm.nih.gov/compound/${compound.cid}" target="_blank" rel="noopener">${compound.cid}</a>`],
    ["Formula", compound.formula],
    ["MW (PubChem)", `${compound.molecular_weight}`],
  ];
  if (descriptors) {
    rows.push(
      ["MW (RDKit)", `<strong>${descriptors.molecular_weight}</strong>`],
      ["logP", descriptors.logp],
      ["TPSA", `${descriptors.tpsa} Å²`],
      ["H-bond donors", descriptors.hbd],
      ["H-bond acceptors", descriptors.hba],
      ["Rotatable bonds", descriptors.rotatable_bonds],
      ["Aromatic rings", descriptors.aromatic_rings],
      ["Lipinski violations", descriptors.lipinski_violations],
    );
  }
  facts("compound-facts", rows);
}

function renderRegulatory(comparisons) {
  if (!comparisons.length) return;
  show("regulatory-panel");
  $("regulatory").innerHTML = comparisons
    .map((c) => {
      const l = c.limit;
      const cls = c.exceeds ? "exceeds" : "within";
      const verdict = c.exceeds ? "EXCEEDS LIMIT" : "within limit";
      return `<div class="verdict ${cls}">
        <div class="verdict-head">
          <span class="badge">${l.jurisdiction}</span>
          <strong>${verdict}</strong>
          <span class="ratio">${c.ratio_of_limit}× the limit</span>
        </div>
        <p>${c.measured_ug_per_kg} µg/kg measured against a maximum of
           <strong>${l.max_level_ug_per_kg} µg/kg</strong> for ${l.toxin} in ${l.commodity}.</p>
        <p class="instrument">${l.instrument} —
           <a href="${l.verify_url}" target="_blank" rel="noopener">verify against primary text</a></p>
      </div>`;
    })
    .join("");
}

function renderLiterature(citations) {
  show("literature-panel");
  if (!citations.length) {
    $("literature").innerHTML =
      `<p class="empty">No indexed literature found for this pair. Absence of evidence
       is reported as such — no substitute is generated.</p>`;
    return;
  }
  $("literature").innerHTML = `<ol>${citations
    .map((c) => `<li>
        <a href="${c.url}" target="_blank" rel="noopener">${c.title}</a>
        <div class="cite-meta">${c.authors.slice(0, 3).join(", ")}${c.authors.length > 3 ? " et al." : ""}
        — ${c.journal} ${c.year ? `(${c.year})` : ""} · PMID ${c.pmid}</div>
      </li>`)
    .join("")}</ol>`;
}

function renderRecalls(recalls) {
  if (!recalls || !recalls.length) return;
  show("recalls-panel");
  $("recalls").innerHTML = `<table><thead><tr>
      <th>Recall</th><th>Class</th><th>Date</th><th>Product</th></tr></thead><tbody>${recalls
    .map((r) => `<tr>
        <td>${r.recall_number ?? ""}</td>
        <td>${r.classification ?? ""}</td>
        <td>${r.recall_initiation_date ?? ""}</td>
        <td>${(r.product ?? "").slice(0, 90)}</td>
      </tr>`)
    .join("")}</tbody></table>`;
}

function renderSources(sources) {
  if (!sources.length) return;
  show("sources-panel");
  $("sources").innerHTML = sources
    .map((s) => `<li><a href="${s.url}" target="_blank" rel="noopener">${s.source}</a>${
      s.license ? ` <span class="lic">${s.license}</span>` : ""}</li>`)
    .join("");
}

function renderWarnings(warnings) {
  if (!warnings.length) { hide("warnings"); return; }
  show("warnings");
  $("warnings").innerHTML =
    `<h2>Gaps in the evidence</h2><ul>${warnings.map((w) => `<li>${w}</li>`).join("")}</ul>`;
}

$("query").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const status = $("status");
  const button = $("run");

  button.disabled = true;
  status.textContent = "querying UniProt, AlphaFold, PubChem, PubMed, openFDA…";
  ["structure-panel", "compound-panel", "regulatory-panel",
   "literature-panel", "recalls-panel", "sources-panel", "warnings"].forEach(hide);

  const measured = form.get("measured_ug_per_kg");
  const body = {
    protein: form.get("protein") || null,
    organism: form.get("organism") || null,
    contaminant: form.get("limit_toxin") || form.get("contaminant") || null,
    measured_ug_per_kg: measured ? Number(measured) : null,
    jurisdiction: form.get("jurisdiction") || null,
  };
  // The compound lookup wants the specific contaminant; limits are indexed by group.
  const compoundName = form.get("contaminant");

  try {
    const data = await fetch("/api/evidence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, contaminant: compoundName }),
    }).then((r) => r.json());

    const limits = await fetch("/api/evidence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json());

    renderWarnings(data.warnings);
    if (data.compound) renderCompound(data.compound.value, data.descriptors?.value);
    if (data.citations) renderLiterature(data.citations.value);
    if (data.recalls) renderRecalls(data.recalls.value);
    renderRegulatory(limits.comparisons);
    renderSources(data.sources);
    if (data.structure) await renderStructure(data.structure.value);

    status.textContent = "done";
  } catch (err) {
    status.textContent = `failed: ${err.message}`;
  } finally {
    button.disabled = false;
  }
});
