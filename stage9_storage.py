"""
Stage 9: Storage
==================
Input:  outputs_ML/graph_completed.json  (produced by Stage 8)
Output: kg_output/knowledge_graph.graphml
        kg_output/knowledge_graph.json
        kg_output/knowledge_graph.db

What this does — exports the graph three ways:
  1. GraphML — standard graph interchange format, openable in Gephi/Cytoscape
     for visualization.
  2. Node-link JSON — easy to reload in Python with nx.node_link_graph(),
     or feed to a JS frontend.
  3. SQLite — entities + relations tables, queryable with plain SQL
     (e.g. SELECT * FROM relations WHERE relation='ENTERED_INTO').

Run:
    pip install networkx --break-system-packages
    python stage9_storage.py
"""

import json
import os
import sqlite3

import networkx as nx


def load_graph(path="outputs_ML/graph_completed.json"):
    with open(path, "r") as f:
        data = json.load(f)
    G = nx.node_link_graph(data)
    G.graph["metadata"] = data.get("metadata", {})
    return G


def store_graph(G, output_dir="kg_output"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. GraphML
    graphml_path = os.path.join(output_dir, "knowledge_graph.graphml")
    G_export = nx.MultiDiGraph()
    for n, d in G.nodes(data=True):
        G_export.add_node(n, type=str(d.get("type", "")))
    for u, v, d in G.edges(data=True):
        G_export.add_edge(u, v, relation=str(d.get("relation", "")),
                           confidence=str(d.get("confidence", "")),
                           inferred=str(d.get("inferred", False)))
    nx.write_graphml(G_export, graphml_path)

    # 2. Node-link JSON
    json_path = os.path.join(output_dir, "knowledge_graph.json")
    data = nx.node_link_data(G_export)
    data["metadata"] = G.graph.get("metadata", {})
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # 3. SQLite
    db_path = os.path.join(output_dir, "knowledge_graph.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE entities (name TEXT PRIMARY KEY, type TEXT)")
    cur.execute("""CREATE TABLE relations (
        subject TEXT, relation TEXT, object TEXT,
        confidence REAL, inferred INTEGER
    )""")
    cur.execute("""CREATE TABLE metrics (
        name TEXT PRIMARY KEY,
        value TEXT
    )""")
    for n, d in G.nodes(data=True):
        cur.execute("INSERT OR IGNORE INTO entities VALUES (?, ?)", (n, d.get("type", "")))
    for u, v, d in G.edges(data=True):
        cur.execute("INSERT INTO relations VALUES (?, ?, ?, ?, ?)",
                    (u, d.get("relation", ""), v, d.get("confidence", 0.0), int(d.get("inferred", False))))
    for key, value in flatten_metrics(G.graph.get("metadata", {})).items():
        cur.execute("INSERT OR REPLACE INTO metrics VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()

    return {"graphml": graphml_path, "json": json_path, "sqlite": db_path}


def flatten_metrics(data, prefix=""):
    rows = {}
    if isinstance(data, dict):
        for key, value in data.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.update(flatten_metrics(value, child_prefix))
    else:
        rows[prefix] = data
    return rows


def main():
    G = load_graph()
    paths = store_graph(G)

    print("Stage 9 — Storage")
    print(f"  Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  GraphML: {paths['graphml']}")
    print(f"  JSON:    {paths['json']}")
    print(f"  SQLite:  {paths['sqlite']}")
    print("\n  Try it: sqlite3 kg_output/knowledge_graph.db "
          "\"SELECT * FROM relations LIMIT 5;\"")


if __name__ == "__main__":
    main()
