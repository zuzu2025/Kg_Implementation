"""
Stage 11: Visualization
=========================
Input:  outputs_ML/ontology.json          (Stage 6)
        outputs_ML/validated_triples.json (Stage 7)
        outputs_ML/rejected_triples.json  (Stage 7)
        outputs_ML/graph_completed.json   (Stage 8)
        outputs_ML/inferred_links.json    (Stage 8)
        outputs_ML/link_prediction_metrics.json (Stage 8)
        kg_output/evaluation_report.json  (Stage 10)
Output: kg_output/dashboard.html

What this does:
  Builds one self-contained HTML file with:
    1. An interactive force-directed graph explorer (search, entity-type
       filters, degree-based node limiting for performance on large graphs).
    2. Pipeline health charts: entity/relation type distribution, Stage 7
       validation pass rate + rejection reasons, Stage 8 link-prediction
       metrics, Stage 6 induced domain rules table.
  All inputs are optional — missing files are skipped gracefully so this
  can be run after any subset of stages 6-10 has completed.

Run:
    python stage11_visualization.py
    (then open kg_output/dashboard.html in a browser)
"""

import json
import os
from collections import Counter, defaultdict


def _load_json(path, default=None):
    if not os.path.exists(path):
        print(f"  (skipping {path} — not found)")
        return default
    with open(path, "r") as f:
        return json.load(f)


def load_all_data():
    ontology = _load_json("outputs_ML/ontology.json", {})
    validated = _load_json("outputs_ML/validated_triples.json", [])
    rejected = _load_json("outputs_ML/rejected_triples.json", [])
    graph_data = _load_json("outputs_ML/graph_completed.json", None)
    inferred = _load_json("outputs_ML/inferred_links.json", [])
    link_metrics = _load_json("outputs_ML/link_prediction_metrics.json", {})
    evaluation = _load_json("kg_output/evaluation_report.json", {})
    return ontology, validated, rejected, graph_data, inferred, link_metrics, evaluation


def normalize_graph(graph_data, max_nodes=400):
    """
    Convert nx.node_link_data() output (which may use 'links' or 'edges'
    as the key depending on networkx version) into a plain, JS-friendly
    {nodes, edges} structure. Also caps to the top-N nodes by degree for
    initial render performance on large graphs — the full data is still
    available client-side for search, just not all rendered by default.
    """
    if not graph_data:
        return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0}

    raw_nodes = graph_data.get("nodes", [])
    raw_edges = graph_data.get("edges", graph_data.get("links", []))

    degree = Counter()
    for e in raw_edges:
        degree[e.get("source")] += 1
        degree[e.get("target")] += 1

    nodes = [
        {
            "id": str(n.get("id")),
            "type": n.get("type", "UNKNOWN"),
            "degree": degree.get(n.get("id"), 0),
        }
        for n in raw_nodes
    ]
    edges = [
        {
            "source": str(e.get("source")),
            "target": str(e.get("target")),
            "relation": e.get("relation", ""),
            "confidence": e.get("confidence", 0.0),
            "inferred": bool(e.get("inferred", False)),
        }
        for e in raw_edges
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "default_limit": min(max_nodes, len(nodes)),
    }


def summarize(ontology, validated, rejected, graph, inferred, link_metrics, evaluation):
    entity_type_counts = Counter(n["type"] for n in graph["nodes"])
    relation_counts = Counter(e["relation"] for e in graph["edges"])

    reject_reason_buckets = Counter()
    for r in rejected:
        reason = r.get("reject_reason", "")
        if reason.startswith("domain/range"):
            reject_reason_buckets["Domain/range violation"] += 1
        elif reason.startswith("semantic type mismatch"):
            reject_reason_buckets["Semantic type mismatch"] += 1
        elif reason.startswith("confidence"):
            reject_reason_buckets["Low confidence"] += 1
        else:
            reject_reason_buckets["Other"] += 1

    total_considered = len(validated) + len(rejected)
    pass_rate = (len(validated) / total_considered * 100) if total_considered else 0.0

    domain_rules = sorted(
        ontology.get("domain_rules", []),
        key=lambda r: r.get("frequency", 0),
        reverse=True,
    )

    return {
        "entity_type_counts": dict(entity_type_counts.most_common()),
        "relation_counts": dict(relation_counts.most_common()),
        "reject_reason_buckets": dict(reject_reason_buckets),
        "validated_count": len(validated),
        "rejected_count": len(rejected),
        "pass_rate": round(pass_rate, 1),
        "inferred_count": len(inferred),
        "domain_rules": domain_rules,
        "link_metrics": link_metrics,
        "evaluation": evaluation,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Contract Knowledge Graph — Pipeline Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<script>__VIS_NETWORK_JS__</script>
<script>__CHART_JS__</script>
<style>
  :root {
    --ink: #0D1117;
    --panel: #151B23;
    --panel-raised: #1B2330;
    --hairline: #2A323F;
    --text: #E6E8EB;
    --text-muted: #8B94A3;
    --gold: #C9A227;
    --teal: #4FA8A0;
    --rust: #B5533C;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--ink);
    color: var(--text);
    font-family: 'Inter', sans-serif;
  }
  .mono { font-family: 'IBM Plex Mono', monospace; }
  header {
    padding: 40px 48px 28px;
    border-bottom: 1px solid var(--hairline);
  }
  header .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 10px;
  }
  header h1 {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 32px;
    margin: 0 0 6px;
    letter-spacing: -0.01em;
  }
  header p {
    color: var(--text-muted);
    margin: 0;
    font-size: 14.5px;
    max-width: 640px;
    line-height: 1.5;
  }

  .kpi-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1px;
    background: var(--hairline);
    border-top: 1px solid var(--hairline);
    border-bottom: 1px solid var(--hairline);
  }
  .kpi {
    background: var(--panel);
    padding: 22px 24px;
    position: relative;
  }
  .kpi .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 500;
    color: var(--gold);
    border: 1px solid rgba(201,162,39,0.35);
    display: inline-block;
    padding: 2px 10px;
    border-radius: 2px;
    box-shadow: inset 0 0 0 1px rgba(201,162,39,0.08);
  }
  .kpi .label {
    display: block;
    margin-top: 10px;
    font-size: 11.5px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
  }

  main {
    padding: 32px 48px 64px;
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    gap: 24px;
  }
  @media (max-width: 980px) { main { grid-template-columns: 1fr; } }

  .panel {
    background: var(--panel);
    border: 1px solid var(--hairline);
    border-radius: 6px;
    padding: 22px 24px 26px;
  }
  .panel h2 {
    font-family: 'Source Serif 4', serif;
    font-size: 17px;
    font-weight: 600;
    margin: 0 0 4px;
  }
  .panel .sub {
    font-size: 12.5px;
    color: var(--text-muted);
    margin: 0 0 16px;
  }

  #graph-panel { grid-row: span 2; display: flex; flex-direction: column; }
  #graph-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 14px;
    margin-bottom: 12px;
    align-items: center;
  }
  #graph-controls input[type="text"] {
    background: var(--panel-raised);
    border: 1px solid var(--hairline);
    color: var(--text);
    padding: 7px 10px;
    border-radius: 4px;
    font-size: 13px;
    font-family: 'Inter', sans-serif;
    flex: 1;
    min-width: 160px;
  }
  #graph-controls label {
    font-size: 12px;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 5px;
    cursor: pointer;
  }
  .type-swatch {
    width: 9px; height: 9px; border-radius: 50%;
    display: inline-block;
  }
  #network {
    height: 560px;
    background: var(--panel-raised);
    border: 1px solid var(--hairline);
    border-radius: 4px;
  }
  #graph-meta {
    margin-top: 10px;
    font-size: 12px;
    color: var(--text-muted);
    font-family: 'IBM Plex Mono', monospace;
  }

  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--hairline); }
  th { color: var(--text-muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
  td.mono, th.mono { font-family: 'IBM Plex Mono', monospace; }
  .bar-cell { display: flex; align-items: center; gap: 8px; }
  .bar-track { flex: 1; height: 5px; background: var(--hairline); border-radius: 3px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--teal); }

  canvas { max-width: 100%; }

  footer {
    padding: 20px 48px 40px;
    color: var(--text-muted);
    font-size: 11.5px;
    font-family: 'IBM Plex Mono', monospace;
  }
</style>
</head>
<body>

<header>
  <div class="eyebrow">Stage 11 · Pipeline Output</div>
  <h1>Contract Knowledge Graph</h1>
  <p>Entities, relations, and induced domain rules extracted from CUAD legal contracts across a ten-stage classical-ML pipeline. Explore the graph below or review pipeline health metrics to the right.</p>
</header>

<div class="kpi-strip" id="kpi-strip"></div>

<main>
  <section class="panel" id="graph-panel">
    <h2>Graph Explorer</h2>
    <p class="sub">Search an entity, filter by type, or limit by connection count.</p>
    <div id="graph-controls">
      <input type="text" id="search-box" placeholder="Search entity name…">
      <span id="type-filters"></span>
      <label><input type="checkbox" id="hide-inferred"> hide inferred edges</label>
    </div>
    <div id="network"></div>
    <div id="graph-meta"></div>
  </section>

  <section class="panel">
    <h2>Entity Types</h2>
    <p class="sub">Node count per entity type in the completed graph.</p>
    <canvas id="entity-chart" height="160"></canvas>
  </section>

  <section class="panel">
    <h2>Relation Types</h2>
    <p class="sub">Edge count per relation type.</p>
    <canvas id="relation-chart" height="180"></canvas>
  </section>

  <section class="panel">
    <h2>Stage 7 — Triple Validation</h2>
    <p class="sub" id="validation-sub"></p>
    <canvas id="validation-chart" height="150"></canvas>
  </section>

  <section class="panel">
    <h2>Stage 8 — Link Prediction</h2>
    <p class="sub">Best model performance on held-out edge/non-edge pairs.</p>
    <div id="link-metrics"></div>
  </section>

  <section class="panel" style="grid-column: 1 / -1;">
    <h2>Stage 6 — Induced Domain Rules</h2>
    <p class="sub">Data-driven subject→relation→object rules, ranked by frequency.</p>
    <table id="rules-table">
      <thead>
        <tr><th>Subject</th><th>Relation</th><th>Object</th><th class="mono">Freq.</th><th>Support</th><th>Confidence</th></tr>
      </thead>
      <tbody></tbody>
    </table>
  </section>
</main>

<footer>generated by stage11_visualization.py</footer>

<script>
const DATA = __PIPELINE_DATA__;

const TYPE_COLORS = {
  PARTY: "#C9A227",
  CONTRACT: "#4FA8A0",
  JURISDICTION: "#8FB8DE",
  DATE: "#B5533C",
  EFFECTIVE_DATE: "#D98A5F",
  NOTICE: "#9C89C4",
  UNKNOWN: "#5A6270",
};
function colorFor(type) { return TYPE_COLORS[type] || "#5A6270"; }

// ---------- KPI strip ----------
const kpis = [
  ["Entities", DATA.graph.total_nodes],
  ["Relations", DATA.graph.total_edges],
  ["Validated triples", DATA.summary.validated_count],
  ["Rejected triples", DATA.summary.rejected_count],
  ["Validation pass rate", DATA.summary.pass_rate + "%"],
  ["Inferred links added", DATA.summary.inferred_count],
];
document.getElementById("kpi-strip").innerHTML = kpis.map(
  ([label, value]) => `<div class="kpi"><span class="value">${value}</span><span class="label">${label}</span></div>`
).join("");

// ---------- Graph explorer ----------
const allNodes = DATA.graph.nodes;
const allEdges = DATA.graph.edges;
const presentTypes = [...new Set(allNodes.map(n => n.type))];

document.getElementById("type-filters").innerHTML = presentTypes.map(t =>
  `<label><input type="checkbox" class="type-cb" value="${t}" checked>
     <span class="type-swatch" style="background:${colorFor(t)}"></span>${t}
   </label>`
).join("");

const nodesDS = new vis.DataSet([]);
const edgesDS = new vis.DataSet([]);
const network = new vis.Network(
  document.getElementById("network"),
  { nodes: nodesDS, edges: edgesDS },
  {
    nodes: {
      shape: "dot",
      font: { color: "#E6E8EB", size: 11, face: "Inter" },
      borderWidth: 1,
    },
    edges: {
      color: { color: "#2A323F", highlight: "#C9A227" },
      arrows: { to: { enabled: true, scaleFactor: 0.4 } },
      font: { color: "#8B94A3", size: 9, strokeWidth: 0, align: "middle" },
      smooth: { type: "continuous" },
    },
    physics: {
      stabilization: { iterations: 120 },
      barnesHut: { gravitationalConstant: -12000, springLength: 110, springConstant: 0.03 },
    },
    interaction: { hover: true, tooltipDelay: 120 },
  }
);
network.once("stabilizationIterationsDone", () => network.setOptions({ physics: false }));

function renderGraph() {
  const query = document.getElementById("search-box").value.trim().toLowerCase();
  const activeTypes = new Set(
    [...document.querySelectorAll(".type-cb:checked")].map(cb => cb.value)
  );
  const hideInferred = document.getElementById("hide-inferred").checked;
  const limit = DATA.graph.default_limit;

  let candidateNodes = allNodes.filter(n => activeTypes.has(n.type));

  if (query) {
    candidateNodes = candidateNodes.filter(n => n.id.toLowerCase().includes(query));
    // pull in direct neighbors of matches for context
    const matchIds = new Set(candidateNodes.map(n => n.id));
    const neighborIds = new Set();
    allEdges.forEach(e => {
      if (matchIds.has(e.source)) neighborIds.add(e.target);
      if (matchIds.has(e.target)) neighborIds.add(e.source);
    });
    candidateNodes = allNodes.filter(n => matchIds.has(n.id) || neighborIds.has(n.id));
  } else {
    candidateNodes = candidateNodes
      .sort((a, b) => b.degree - a.degree)
      .slice(0, limit);
  }

  const idSet = new Set(candidateNodes.map(n => n.id));
  const visibleEdges = allEdges.filter(e =>
    idSet.has(e.source) && idSet.has(e.target) && (!hideInferred || !e.inferred)
  );

  nodesDS.clear(); edgesDS.clear();
  nodesDS.add(candidateNodes.map(n => ({
    id: n.id,
    label: n.id.length > 22 ? n.id.slice(0, 20) + "…" : n.id,
    title: `${n.id}  [${n.type}]  degree ${n.degree}`,
    color: colorFor(n.type),
    value: Math.max(4, Math.min(24, 4 + n.degree)),
  })));
  edgesDS.add(visibleEdges.map((e, i) => ({
    id: i,
    from: e.source,
    to: e.target,
    label: e.relation,
    dashes: e.inferred,
    title: `confidence ${e.confidence}`,
  })));

  document.getElementById("graph-meta").textContent =
    `showing ${candidateNodes.length} of ${DATA.graph.total_nodes} entities, ` +
    `${visibleEdges.length} of ${DATA.graph.total_edges} relations` +
    (query ? ` — filtered by "${query}"` : ` — top ${limit} by connections`);
}

document.getElementById("search-box").addEventListener("input", renderGraph);
document.getElementById("hide-inferred").addEventListener("change", renderGraph);
document.querySelectorAll(".type-cb").forEach(cb => cb.addEventListener("change", renderGraph));
renderGraph();

// ---------- Charts ----------
const chartDefaults = {
  color: "#8B94A3",
  borderColor: "#2A323F",
};

new Chart(document.getElementById("entity-chart"), {
  type: "bar",
  data: {
    labels: Object.keys(DATA.summary.entity_type_counts),
    datasets: [{
      data: Object.values(DATA.summary.entity_type_counts),
      backgroundColor: Object.keys(DATA.summary.entity_type_counts).map(colorFor),
      borderRadius: 3,
    }],
  },
  options: {
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: "#8B94A3", font: { size: 10 } }, grid: { display: false } },
      y: { ticks: { color: "#8B94A3" }, grid: { color: "#2A323F" } },
    },
  },
});

new Chart(document.getElementById("relation-chart"), {
  type: "bar",
  data: {
    labels: Object.keys(DATA.summary.relation_counts),
    datasets: [{
      data: Object.values(DATA.summary.relation_counts),
      backgroundColor: "#4FA8A0",
      borderRadius: 3,
    }],
  },
  options: {
    indexAxis: "y",
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: "#8B94A3" }, grid: { color: "#2A323F" } },
      y: { ticks: { color: "#8B94A3", font: { size: 10 } }, grid: { display: false } },
    },
  },
});

document.getElementById("validation-sub").textContent =
  `${DATA.summary.validated_count} validated / ${DATA.summary.rejected_count} rejected — ${DATA.summary.pass_rate}% pass rate`;

const rejectLabels = Object.keys(DATA.summary.reject_reason_buckets);
new Chart(document.getElementById("validation-chart"), {
  type: "doughnut",
  data: {
    labels: ["Validated", ...rejectLabels],
    datasets: [{
      data: [DATA.summary.validated_count, ...rejectLabels.map(k => DATA.summary.reject_reason_buckets[k])],
      backgroundColor: ["#4FA8A0", "#B5533C", "#D98A5F", "#8B94A3", "#5A6270"],
      borderColor: "#151B23",
      borderWidth: 2,
    }],
  },
  options: {
    plugins: { legend: { position: "bottom", labels: { color: "#8B94A3", font: { size: 11 }, boxWidth: 10 } } },
  },
});

// ---------- Link prediction metrics ----------
const lm = DATA.summary.link_metrics || {};
const bm = lm.best_metrics || {};
const linkMetricsEl = document.getElementById("link-metrics");
if (lm.best_model) {
  linkMetricsEl.innerHTML = `
    <table>
      <tr><th>Best model</th><td class="mono">${lm.best_model}</td></tr>
      <tr><th>F1</th><td class="mono">${bm.f1 ?? "—"}</td></tr>
      <tr><th>Precision</th><td class="mono">${bm.precision ?? "—"}</td></tr>
      <tr><th>Recall</th><td class="mono">${bm.recall ?? "—"}</td></tr>
      <tr><th>Threshold</th><td class="mono">${bm.threshold ?? "—"}</td></tr>
      <tr><th>Inferred links added</th><td class="mono">${DATA.summary.inferred_count}</td></tr>
    </table>`;
} else {
  linkMetricsEl.innerHTML = `<p class="sub">No link-prediction metrics found — run Stage 8 first.</p>`;
}

// ---------- Domain rules table ----------
const rulesBody = document.querySelector("#rules-table tbody");
rulesBody.innerHTML = DATA.summary.domain_rules.map(r => `
  <tr>
    <td>${r.subject}</td>
    <td class="mono" style="color:#C9A227">${r.relation}</td>
    <td>${r.object}</td>
    <td class="mono">${r.frequency}</td>
    <td><div class="bar-cell"><div class="bar-track"><div class="bar-fill" style="width:${Math.round((r.support||0)*100)}%"></div></div><span class="mono">${(r.support*100).toFixed(0)}%</span></div></td>
    <td><div class="bar-cell"><div class="bar-track"><div class="bar-fill" style="width:${Math.round((r.confidence||0)*100)}%"></div></div><span class="mono">${(r.confidence*100).toFixed(0)}%</span></div></td>
  </tr>
`).join("");
</script>

</body>
</html>
"""


def _read_local_lib(filename):
    """
    Read a bundled JS library from the same directory as this script.
    Embedding these directly (instead of loading from a CDN) means the
    generated dashboard has zero external dependencies — it opens and
    works fully offline, and isn't affected by corporate firewalls or
    CDN outages.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing bundled library: {filename}\n"
            f"Expected it at: {path}\n"
            f"Make sure {filename} is in the same folder as stage11_visualization.py."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_dashboard():
    print("Loading pipeline outputs...")
    ontology, validated, rejected, graph_data, inferred, link_metrics, evaluation = load_all_data()

    graph = normalize_graph(graph_data)
    summary = summarize(ontology, validated, rejected, graph, inferred, link_metrics, evaluation)

    payload = {"graph": graph, "summary": summary}

    print("Bundling visualization libraries (vis-network, Chart.js)...")
    vis_network_js = _read_local_lib("vis-network.min.js")
    chart_js = _read_local_lib("chart.umd.js")

    html = HTML_TEMPLATE.replace("__PIPELINE_DATA__", json.dumps(payload))
    html = html.replace("__VIS_NETWORK_JS__", vis_network_js)
    html = html.replace("__CHART_JS__", chart_js)

    os.makedirs("kg_output", exist_ok=True)
    out_path = "kg_output/dashboard.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nStage 11 — Visualization")
    print(f"  Entities in graph: {graph['total_nodes']}")
    print(f"  Relations in graph: {graph['total_edges']}")
    print(f"  Validated / Rejected triples: {summary['validated_count']} / {summary['rejected_count']}")
    print(f"  Domain rules rendered: {len(summary['domain_rules'])}")
    print(f"\nSaved {out_path}")
    print("Open it in a browser to explore the graph and pipeline metrics.")


if __name__ == "__main__":
    build_dashboard()