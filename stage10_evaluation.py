"""
Stage 10: Evaluation
======================
Input:  outputs_ML/graph_completed.json   (produced by Stage 8/9)
        outputs_ML/validated_triples.json (produced by Stage 7)
        outputs_ML/rejected_triples.json  (produced by Stage 7)
        outputs_ML/inferred_links.json    (produced by Stage 8)
Output: kg_output/evaluation_report.json

What this does:
  1. Structural graph statistics — nodes, edges, density, avg degree,
     connected components.
  2. Coverage — % of entities that ended up connected to at least one relation.
  3. Stage 7 pass-rate — validated vs rejected triple ratio.
  4. A random stratified sample of validated triples for manual precision
     spot-checking (since no gold-standard KG exists for CUAD to compare
     against automatically — precision has to be estimated this way).

Run:
    pip install networkx numpy --break-system-packages
    python stage10_evaluation.py
"""

import json
import os
import random

import numpy as np
import networkx as nx


def load_inputs():
    with open("outputs_ML/graph_completed.json", "r") as f:
        graph_data = json.load(f)
        G = nx.node_link_graph(graph_data)
        graph_metadata = graph_data.get("metadata", {})

    with open("outputs_ML/validated_triples.json", "r") as f:
        validated = json.load(f)

    with open("outputs_ML/rejected_triples.json", "r") as f:
        rejected = json.load(f)

    inferred_path = "outputs_ML/inferred_links.json"
    inferred = []
    if os.path.exists(inferred_path):
        with open(inferred_path, "r") as f:
            inferred = json.load(f)

    metrics_path = "outputs_ML/link_prediction_metrics.json"
    link_metrics = graph_metadata.get("link_prediction", {})
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            link_metrics = json.load(f)

    clusters = []
    er_candidates = []
    if os.path.exists("outputs_ML/entity_clusters.json"):
        with open("outputs_ML/entity_clusters.json", "r") as f:
            clusters = json.load(f)
    if os.path.exists("outputs_ML/er_candidates.json"):
        with open("outputs_ML/er_candidates.json", "r") as f:
            er_candidates = json.load(f)

    return G, validated, rejected, inferred, link_metrics, clusters, er_candidates


def f1_score(precision, recall):
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def evaluate_graph(G, validated_triples, rejected_triples, inferred_links,
                   link_metrics, entity_clusters=None, er_candidates=None,
                   sample_size=5):
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    G_undirected = nx.Graph(G)
    density = nx.density(G_undirected)
    components = nx.number_connected_components(G_undirected)
    degrees = [d for _, d in G_undirected.degree()]
    avg_degree = float(np.mean(degrees)) if degrees else 0.0

    total_considered = len(validated_triples) + len(rejected_triples)
    validation_pass_rate = len(validated_triples) / total_considered if total_considered else 0.0
    avg_validated_confidence = float(np.mean([t.get("confidence", 0.0) for t in validated_triples])) if validated_triples else 0.0
    avg_rejected_confidence = float(np.mean([t.get("confidence", 0.0) for t in rejected_triples])) if rejected_triples else 0.0

    # No gold KG is available, so this is a confidence-weighted proxy:
    # validated confidence estimates precision, and accepted share estimates recall.
    stage7_precision_proxy = avg_validated_confidence
    stage7_recall_proxy = validation_pass_rate
    stage7_f1_proxy = f1_score(stage7_precision_proxy, stage7_recall_proxy)

    connected_entities = sum(1 for n in G_undirected.nodes() if G_undirected.degree(n) > 0)
    coverage = connected_entities / n_nodes if n_nodes else 0.0
    entity_clusters = entity_clusters or []
    er_candidates = er_candidates or []
    merged_pairs = [p for p in er_candidates if p.get("merged")]
    cluster_sizes = [len(c.get("mention_ids", [])) for c in entity_clusters]

    report = {
        "num_entities": n_nodes,
        "num_relations": n_edges,
        "num_inferred_links_added": len(inferred_links),
        "graph_density": round(density, 4),
        "connected_components": components,
        "avg_degree": round(avg_degree, 3),
        "entity_coverage": round(coverage, 3),
        "stage7_validation_pass_rate": round(validation_pass_rate, 3),
        "stage7_precision_proxy": round(stage7_precision_proxy, 3),
        "stage7_recall_proxy": round(stage7_recall_proxy, 3),
        "stage7_f1_proxy": round(stage7_f1_proxy, 3),
        "stage7_avg_validated_confidence": round(avg_validated_confidence, 3),
        "stage7_avg_rejected_confidence": round(avg_rejected_confidence, 3),
        "stage8_link_prediction": link_metrics.get("best_metrics", link_metrics),
        "stage3_er_clusters": len(entity_clusters),
        "stage3_er_pairs_scored": len(er_candidates),
        "stage3_er_pairs_merged": len(merged_pairs),
        "stage3_er_avg_cluster_size": round(float(np.mean(cluster_sizes)), 3) if cluster_sizes else 0.0,
        "stage3_er_max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
    }

    print("Stage 10 — Evaluation")
    for k, v in report.items():
        print(f"  {k}: {v}")

    sample = random.sample(validated_triples, min(sample_size, len(validated_triples)))
    print(f"\n  Sample of {len(sample)} triples for manual precision spot-check:")
    for t in sample:
        print(f"    ({t['subject']}, {t['relation']}, {t['object']}) — confidence {t['confidence']}")

    report["manual_review_sample"] = [
        {"subject": t["subject"], "relation": t["relation"], "object": t["object"]}
        for t in sample
    ]

    print("\n  NOTE: precision here is an ESTIMATE from manual spot-check, since no gold-standard "
          "KG exists for CUAD to compare against automatically.")

    return report


def main():
    G, validated, rejected, inferred, link_metrics, clusters, er_candidates = load_inputs()
    report = evaluate_graph(G, validated, rejected, inferred, link_metrics, clusters, er_candidates)

    os.makedirs("kg_output", exist_ok=True)
    with open("kg_output/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nSaved kg_output/evaluation_report.json")


if __name__ == "__main__":
    main()
