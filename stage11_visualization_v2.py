"""
Stage 11 v2: Unvalidated Relations Graph
==========================================
Input:  outputs_ML/relations.json   (Stage 6, pre-validation candidate triples --
                           flat list [{...}] OR {"links": [...]} shape,
                           both are accepted)
Output: kg_output/relations_graph_v2.html   (standalone, separate from
                                              kg_output/dashboard.html)
        kg_output/relations_graph_v2.json   (the normalized {nodes, edges}
                                              structure, in case you want
                                              it as data rather than a page)

This is deliberately NOT a full dashboard. No KPI strip, no charts, no
validation stats, no domain-rules table -- none of that applies to stage 6
data anyway, since it hasn't been validated yet. Just the force-directed
graph: search a node, see its neighborhood, done.

This script never reads graph_completed.json and never merges anything
into it. It is a completely separate, standalone view of the unvalidated
candidate pool only.

Run:
    python stage11_visualization_v2.py
    (then open kg_output/relations_graph_v2.html in a browser)
"""

import json
import os
from collections import Counter


def _load_relations(path="outputs_ML/relations.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. This script visualizes ONLY relations.json "
            f"(stage 6, pre-validation) -- put it next to this script, or "
            f"pass a different path to build_graph(path=...)."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # accept both a flat list of triples and a {"links": [...]} node-link shape
    return data if isinstance(data, list) else data.get("links", [])


def normalize_relations(raw_links):
    """
    Build the FULL (uncapped) {nodes, edges} structure directly from the
    triples -- there's no separate 'nodes' list in relations.json like
    there is in graph_completed.json, so nodes are inferred from every
    distinct source/target seen across the triples. This is what gets
    saved to relations_graph_v2.json (the data file) in full.
    """
    edges = []
    degree = Counter()
    seen_nodes = set()

    for link in raw_links:
        src = (link.get("source") or link.get("subject") or link.get("head")
               or link.get("entity1"))
        tgt = (link.get("target") or link.get("object") or link.get("tail")
               or link.get("entity2"))
        rel = (link.get("relation") or link.get("predicate") or link.get("label")
               or link.get("rel_type"))
        if not src or not tgt or not rel:
            continue  # skip malformed entries rather than fail the whole build
        conf = link.get("confidence", 0.0)
        edges.append({
            "source": str(src),
            "target": str(tgt),
            "relation": str(rel),
            "confidence": conf,
        })
        degree[src] += 1
        degree[tgt] += 1
        seen_nodes.add(src)
        seen_nodes.add(tgt)

    nodes = [{"id": str(n), "degree": degree[n]} for n in seen_nodes]

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }


def cap_for_html(full_graph, max_nodes=400):
    """
    IMPORTANT: for large candidate pools (thousands of triples), embedding
    everything into the HTML makes the page too big for a browser to parse
    (multi-MB inline JSON + physics simulation over it all -> "Not
    Responding"). This caps to the top-N nodes BY DEGREE server-side,
    BEFORE anything gets written to the page -- not just as a client-side
    display filter, which is what caused the 28MB file.
    """
    degree_by_id = {n["id"]: n["degree"] for n in full_graph["nodes"]}
    top_node_ids = {
        n_id for n_id, _ in Counter(degree_by_id).most_common(max_nodes)
    }
    capped_edges = [
        e for e in full_graph["edges"]
        if e["source"] in top_node_ids and e["target"] in top_node_ids
    ]
    capped_nodes = [
        {"id": n_id, "degree": degree_by_id[n_id]} for n_id in top_node_ids
    ]
    return {
        "nodes": capped_nodes,
        "edges": capped_edges,
        "total_nodes": len(capped_nodes),
        "total_edges": len(capped_edges),
        "total_nodes_full": full_graph["total_nodes"],
        "total_edges_full": full_graph["total_edges"],
        "default_limit": len(capped_nodes),
        "capped": full_graph["total_nodes"] > max_nodes,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Unvalidated Relations Graph — Stage 6</title>
<script>__VIS_NETWORK_JS__</script>
<style>
  :root {
    --ink: #0D1117;
    --panel: #151B23;
    --hairline: #2A323F;
    --text: #E6E8EB;
    --text-muted: #8B94A3;
    --rust: #B5533C;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--ink); color: var(--text); font-family: 'Inter', system-ui, sans-serif; }
  header {
    padding: 20px 28px;
    border-bottom: 1px solid var(--hairline);
    display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
  }
  header .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--rust);
    border: 1px solid rgba(181,83,60,0.4);
    padding: 2px 8px;
    border-radius: 2px;
  }
  header h1 { font-size: 18px; margin: 0; font-weight: 600; }
  header .meta { color: var(--text-muted); font-size: 12.5px; font-family: 'IBM Plex Mono', monospace; margin-left: auto; }
  .controls {
    padding: 14px 28px;
    border-bottom: 1px solid var(--hairline);
    display: flex; gap: 12px; align-items: center;
  }
  #search-box {
    flex: 1; max-width: 420px;
    background: var(--panel);
    border: 1px solid var(--hairline);
    color: var(--text);
    padding: 9px 12px;
    border-radius: 4px;
    font-size: 13.5px;
    outline: none;
  }
  #search-box:focus { border-color: var(--rust); }
  #graph-meta { color: var(--text-muted); font-size: 12px; font-family: 'IBM Plex Mono', monospace; }
  #graph { height: calc(100vh - 118px); width: 100%; }
</style>
</head>
<body>
<header>
  <span class="eyebrow">Unvalidated · Stage 6</span>
  <h1>Relations Graph (pre-validation)</h1>
  <span class="meta" id="graph-meta"></span>
</header>
<div class="controls">
  <input id="search-box" type="text" placeholder="Search an entity…">
</div>
<div id="capped-note" style="display:none; padding:8px 28px; font-size:12.5px; color:#B5533C; background:rgba(181,83,60,0.08); border-bottom:1px solid var(--hairline);"></div>
<div id="graph"></div>

<script>
const DATA = __PIPELINE_DATA__;

const allNodes = DATA.graph.nodes;
const allEdges = DATA.graph.edges;

if (DATA.graph.capped) {
  const note = document.getElementById("capped-note");
  note.style.display = "block";
  note.textContent =
    `Showing top ${DATA.graph.total_nodes} of ${DATA.graph.total_nodes_full} entities by connection count ` +
    `(full pool has ${DATA.graph.total_edges_full} relations — see relations_graph_v2.json for the complete data). ` +
    `Search below only searches within this embedded subset.`;
}

const container = document.getElementById("graph");
const nodesDS = new vis.DataSet([]);
const edgesDS = new vis.DataSet([]);
const network = new vis.Network(
  container,
  { nodes: nodesDS, edges: edgesDS },
  {
    nodes: {
      shape: "dot",
      color: { background: "#B5533C", border: "#7A382A" },
      font: { color: "#E6E8EB", size: 12 },
    },
    edges: {
      color: { color: "#4A3530", highlight: "#B5533C" },
      font: { color: "#8B94A3", size: 10, strokeWidth: 0 },
      arrows: { to: { enabled: true, scaleFactor: 0.5 } },
      smooth: { type: "continuous" },
      dashes: true,
    },
    physics: {
      stabilization: { iterations: 150 },
      barnesHut: { gravitationalConstant: -12000, springLength: 110, springConstant: 0.03 },
    },
    interaction: { hover: true, tooltipDelay: 120 },
  }
);
network.once("stabilizationIterationsDone", () => network.setOptions({ physics: false }));

function renderGraph() {
  const query = document.getElementById("search-box").value.trim().toLowerCase();
  const limit = DATA.graph.default_limit;

  let candidateNodes;
  if (query) {
    candidateNodes = allNodes.filter(n => n.id.toLowerCase().includes(query));
    const matchIds = new Set(candidateNodes.map(n => n.id));
    const neighborIds = new Set();
    allEdges.forEach(e => {
      if (matchIds.has(e.source)) neighborIds.add(e.target);
      if (matchIds.has(e.target)) neighborIds.add(e.source);
    });
    candidateNodes = allNodes.filter(n => matchIds.has(n.id) || neighborIds.has(n.id));
  } else {
    candidateNodes = [...allNodes].sort((a, b) => b.degree - a.degree).slice(0, limit);
  }

  const idSet = new Set(candidateNodes.map(n => n.id));
  const visibleEdges = allEdges.filter(e => idSet.has(e.source) && idSet.has(e.target));

  nodesDS.clear(); edgesDS.clear();
  nodesDS.add(candidateNodes.map(n => ({
    id: n.id,
    label: n.id.length > 22 ? n.id.slice(0, 20) + "…" : n.id,
    title: `${n.id}  degree ${n.degree}`,
    value: Math.max(4, Math.min(24, 4 + n.degree)),
  })));
  edgesDS.add(visibleEdges.map((e, i) => ({
    id: i,
    from: e.source,
    to: e.target,
    label: e.relation,
    title: `confidence ${e.confidence} — UNVALIDATED`,
  })));

  document.getElementById("graph-meta").textContent =
    `${candidateNodes.length} of ${DATA.graph.total_nodes} entities, ` +
    `${visibleEdges.length} of ${DATA.graph.total_edges} unvalidated relations` +
    (query ? ` — filtered by "${query}"` : ` — top ${limit} by connections`);
}

document.getElementById("search-box").addEventListener("input", renderGraph);
renderGraph();
</script>
</body>
</html>
"""


def _read_local_lib(filename):
    """Reads a bundled JS library from the same directory as this script,
    so the output HTML has zero external dependencies and works offline."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing bundled library: {filename}\n"
            f"Expected it at: {path}\n"
            f"Make sure {filename} is in the same folder as this script "
            f"(the same copy stage11_visualization.py already uses)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_graph(relations_path="outputs_ML/relations.json", max_nodes=400):
    print("Loading relations.json (stage 6, pre-validation)...")
    raw_links = _load_relations(relations_path)
    full_graph = normalize_relations(raw_links)

    os.makedirs("kg_output", exist_ok=True)

    json_out_path = "kg_output/relations_graph_v2.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(full_graph, f, indent=2)

    html_graph = cap_for_html(full_graph, max_nodes=max_nodes)
    payload = {"graph": html_graph}

    print("Bundling visualization library (vis-network)...")
    vis_network_js = _read_local_lib("vis-network.min.js")

    html = HTML_TEMPLATE.replace("__PIPELINE_DATA__", json.dumps(payload))
    html = html.replace("__VIS_NETWORK_JS__", vis_network_js)

    html_out_path = "kg_output/relations_graph_v2.html"
    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nStage 11 v2 — Unvalidated Relations Graph")
    print(f"  Entities in full pool: {full_graph['total_nodes']}")
    print(f"  Relations in full pool: {full_graph['total_edges']}")
    if html_graph["capped"]:
        print(f"  HTML shows top {html_graph['total_nodes']} entities by degree "
              f"(full pool saved uncapped in the JSON file)")
    print(f"\nSaved {json_out_path}  (full, uncapped graph data)")
    print(f"Saved {html_out_path}  (capped interactive view — sized to stay responsive)")
    print("This is a separate output from kg_output/dashboard.html and never touches graph_completed.json.")


if __name__ == "__main__":
    build_graph()