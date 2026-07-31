"use strict";

const $ = (id) => document.getElementById(id);
const show = (id) => { $(id).hidden = false; };
const hide = (id) => { $(id).hidden = true; };

// AlphaFold's published pLDDT bands. The same four colours are the site palette,
// so the legend on the page and the structure below it speak one language.
const PLDDT_BANDS = [
  { min: 90, colour: "#0053d6" },
  { min: 70, colour: "#65cbf3" },
  { min: 50, colour: "#ffdb13" },
  { min: 0,  colour: "#ff7d45" },
];

const plddtColour = (b) => PLDDT_BANDS.find((band) => b > band.min).colour;

const pct = (fraction) => `${(fraction * 100).toFixed(1)}%`;

function facts(target, rows) {
  $(target).innerHTML = rows
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`)
    .join("");
}

const RESULT_PANELS = [
  "structure-panel", "compound-panel", "regulatory-panel",
  "literature-panel", "recalls-panel", "sources-panel", "warnings",
];

async function renderStructure(structure) {
  show("structure-panel");
  const accession = structure.uniprot_accession;

  facts("structure-facts", [
    ["Accession", `<a href="https://www.uniprot.org/uniprotkb/${accession}" target="_blank" rel="noopener">${accession}</a>`],
    ["Description", structure.description],
    ["Organism", structure.organism],
    ["Mean pLDDT", `<b>${structure.mean_plddt}</b> — ${structure.confidence_band}`],
    ["Very high", pct(structure.fraction_very_high)],
    ["Confident", pct(structure.fraction_confident)],
    ["Low", pct(structure.fraction_low)],
    ["Very low", pct(structure.fraction_very_low)],
    ["Model", `v${structure.model_version}`],
  ]);

  const pdb = await fetch(`/api/structure/${accession}.pdb`).then((r) => {
    if (!r.ok) throw new Error(`structure fetch failed (${r.status})`);
    return r.text();
  });

  const host = $("viewer");
  host.innerHTML = "";
  const dark = document.documentElement.dataset.theme === "dark";
  const viewer = $3Dmol.createViewer(host, { backgroundColor: dark ? "#0f151c" : "#ffffff" });
  window.__toxitraceViewer = viewer;
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
    ["MW (PubChem)", compound.molecular_weight],
  ];
  if (descriptors) {
    rows.push(
      ["MW (RDKit)", `<b>${descriptors.molecular_weight}</b>`],
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

  $("regulatory").innerHTML = comparisons.map((c) => {
    const limit = c.limit;
    const state = c.exceeds ? "exceeds" : "within";
    const word = c.exceeds ? "Exceeds limit" : "Within limit";
    return `<div class="verdict verdict-${state}">
      <div class="verdict-top">
        <span class="jur">${limit.jurisdiction}</span>
        <span class="verdict-word">${word}</span>
        <span class="ratio">${c.ratio_of_limit}× limit</span>
      </div>
      <p>${c.measured_ug_per_kg} µg/kg measured against a maximum of
         <b>${limit.max_level_ug_per_kg} µg/kg</b> for ${limit.toxin} in ${limit.commodity}.</p>
      <p class="instrument">${limit.instrument} —
         <a href="${limit.verify_url}" target="_blank" rel="noopener">verify against primary text</a></p>
    </div>`;
  }).join("");
}

function renderLiterature(citations) {
  show("literature-panel");
  if (!citations.length) {
    $("literature").innerHTML =
      `<p class="empty">No indexed literature found for this pair. Reported as such —
       nothing is substituted when a search returns nothing.</p>`;
    return;
  }
  $("literature").innerHTML = `<ol class="cites">${citations.map((c) => {
    const authors = c.authors.slice(0, 3).join(", ") + (c.authors.length > 3 ? " et al." : "");
    return `<li>
        <a href="${c.url}" target="_blank" rel="noopener">${c.title}</a>
        <div class="cite-meta">${authors} — ${c.journal} ${c.year ? `(${c.year})` : ""} · PMID ${c.pmid}</div>
      </li>`;
  }).join("")}</ol>`;
}

function renderRecalls(recalls) {
  if (!recalls || !recalls.length) return;
  show("recalls-panel");
  $("recalls").innerHTML = `<table><thead><tr>
      <th>Recall</th><th>Class</th><th>Date</th><th>Product</th></tr></thead><tbody>${
    recalls.map((r) => `<tr>
        <td>${r.recall_number ?? ""}</td>
        <td>${(r.classification ?? "").replace("Class ", "")}</td>
        <td>${r.recall_initiation_date ?? ""}</td>
        <td>${(r.product ?? "").slice(0, 80)}</td>
      </tr>`).join("")}</tbody></table>`;
}

function renderSources(sources) {
  if (!sources.length) return;
  show("sources-panel");
  $("sources-used").innerHTML = sources.map((s) =>
    `<li><a href="${s.url}" target="_blank" rel="noopener">${s.source}</a>${
      s.license ? `<span class="lic">${s.license}</span>` : ""}</li>`).join("");
}

function renderWarnings(warnings) {
  if (!warnings.length) { hide("warnings"); return; }
  show("warnings");
  $("warnings").innerHTML =
    `<h3>Gaps in the evidence</h3><ul>${warnings.map((w) => `<li>${w}</li>`).join("")}</ul>`;
}

$("query").addEventListener("submit", async (event) => {
  event.preventDefault();

  const form = new FormData(event.target);
  const status = $("status");
  const button = $("run-btn");
  const measured = form.get("measured_ug_per_kg");
  const jurisdiction = form.get("jurisdiction") || "";

  button.disabled = true;
  status.textContent = "querying UniProt, AlphaFold, PubChem, PubMed, openFDA…";
  RESULT_PANELS.forEach(hide);

  // The compound lookup wants the specific molecule ("aflatoxin B1"); regulatory
  // limits are written per group ("aflatoxins, total"), so they are fetched apart.
  const evidenceBody = {
    protein: form.get("protein") || null,
    organism: form.get("organism") || null,
    contaminant: form.get("contaminant") || null,
  };

  const limitsRequest = () => {
    const toxin = form.get("limit_toxin") || form.get("contaminant");
    if (!toxin || !measured) return Promise.resolve({ comparisons: [] });
    const params = new URLSearchParams({ toxin, measured_ug_per_kg: measured });
    if (jurisdiction) params.set("jurisdiction", jurisdiction);
    return fetch(`/api/limits?${params}`).then((r) => r.json());
  };

  try {
    const [evidence, limits] = await Promise.all([
      fetch("/api/evidence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(evidenceBody),
      }).then((r) => r.json()),
      limitsRequest(),
    ]);

    renderWarnings(evidence.warnings);
    if (evidence.compound) renderCompound(evidence.compound.value, evidence.descriptors?.value);
    if (evidence.citations) renderLiterature(evidence.citations.value);
    if (evidence.recalls) renderRecalls(evidence.recalls.value);
    renderRegulatory(limits.comparisons ?? []);
    renderSources(evidence.sources);
    if (evidence.structure) await renderStructure(evidence.structure.value);

    status.textContent = "done";
  } catch (err) {
    status.textContent = `failed — ${err.message}`;
  } finally {
    button.disabled = false;
  }
});

/* ---------- ask ---------- */

const thread = $("thread");

function addMessage(cls, html) {
  const node = document.createElement("div");
  node.className = `msg ${cls}`;
  node.innerHTML = html;
  thread.appendChild(node);
  thread.scrollTop = thread.scrollHeight;
  return node;
}

const escapeHtml = (text) =>
  text.replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));

async function askQuestion(question) {
  const input = $("ask-input");
  const button = $("ask-btn");

  addMessage("msg-you", escapeHtml(question));
  input.value = "";
  input.disabled = button.disabled = true;
  const pending = addMessage("msg-bot", "<p>Looking it up…</p>");

  try {
    const reply = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }).then((r) => r.json());

    if (reply.detail) throw new Error(reply.detail);

    pending.className = `msg ${reply.refused ? "msg-bot msg-refused" : "msg-bot"}`;
    let meta = "";
    if (!reply.refused) {
      const tools = [...new Set((reply.tool_results || []).map((t) => t.tool))];
      const checked = reply.is_grounded
        ? "<b>every figure traced back to the source</b>"
        : `<span class="bad">${reply.ungrounded.length} figure(s) could not be traced — treat with caution</span>`;
      meta = `<div class="msg-meta">${tools.length ? `looked up: ${tools.join(", ")} · ` : ""}${checked}</div>`;
    }
    pending.innerHTML = `<p>${escapeHtml(reply.answer || "No answer returned.")}</p>${meta}`;
  } catch (err) {
    pending.className = "msg msg-bot msg-refused";
    pending.innerHTML = `<p>Couldn't answer that — ${escapeHtml(err.message)}</p>`;
  } finally {
    input.disabled = button.disabled = false;
    input.focus();
  }
}

$("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = $("ask-input").value.trim();
  if (question) askQuestion(question);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => askQuestion(chip.textContent.trim()));
});

/* ---------- theme ---------- */

const THEME_KEY = "toxitrace-theme";

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const button = $("theme-toggle");
  if (button) {
    button.setAttribute("aria-label",
      theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }
  // The structure viewer draws its own canvas, so it needs telling separately.
  if (window.__toxitraceViewer) {
    window.__toxitraceViewer.setBackgroundColor(theme === "dark" ? "#0f151c" : "#ffffff");
    window.__toxitraceViewer.render();
  }
}

// Saved choice wins; otherwise follow the operating system.
const savedTheme = localStorage.getItem(THEME_KEY);
applyTheme(savedTheme || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));

$("theme-toggle")?.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
});

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (event) => {
  if (!localStorage.getItem(THEME_KEY)) applyTheme(event.matches ? "dark" : "light");
});
