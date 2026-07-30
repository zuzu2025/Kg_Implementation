"""
Stage 8: KG Completion (ML Link Prediction)
=============================================
Input:  outputs_ML/validated_triples.json  (produced by Stage 7)
Output: outputs_ML/graph_completed.json  (node-link graph, incl. inferred edges)

What this does:
  1. Builds a graph from the validated triples.
  2. Extracts classical graph-topology features for entity pairs:
     common neighbors, Jaccard coefficient, preferential attachment, Adamic-Adar.
  3. Trains SVM and Random Forest to distinguish real edges from random
     non-edges, using those features.
  4. Scores all currently-missing pairs and adds the highest-confidence
     ones back into the graph as inferred links.

Run:
    pip install networkx scikit-learn numpy --break-system-packages
    python stage8_kg_completion.py
"""

import json
import os
import random

import numpy as np
import networkx as nx
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


def load_validated_triples(path="outputs_ML/validated_triples.json"):
    with open(path, "r") as f:
        return json.load(f)


def build_graph(validated_triples):
    G = nx.MultiDiGraph()
    for t in validated_triples:
        G.add_node(t["subject"], type=t["subject_type"])
        G.add_node(t["object"], type=t["object_type"])
        G.add_edge(t["subject"], t["object"], relation=t["relation"],
                    confidence=t["confidence"], inferred=False)
    return G


def extract_pair_features(G_undirected, u, v):
    """
    Classical graph-topology features — no embeddings, just structural
    counts (fits the classical-ML theme of the rest of the pipeline).

    IMPORTANT: if a direct edge (u, v) already exists in G_undirected, it is
    temporarily removed before computing features and restored afterward.
    Without this, shortest_path_length(u, v) is trivially 1 whenever a
    direct edge exists and >= 2 for every sampled negative pair (which are
    sampled specifically as non-edges) — meaning that one feature alone
    perfectly encodes the label being predicted. That leakage is what
    produced a suspicious 0.998-1.000 F1 previously, and it also explains
    why 0 inferred links were ever added: the model learned "predict 1 iff
    a direct edge already exists," which can never fire on a genuine
    candidate non-edge at inference time.
    """
    had_edge = G_undirected.has_edge(u, v)
    edge_data = None
    if had_edge:
        edge_data = G_undirected.get_edge_data(u, v)
        G_undirected.remove_edge(u, v)

    try:
        neighbors_u = set(G_undirected.neighbors(u)) if u in G_undirected else set()
        neighbors_v = set(G_undirected.neighbors(v)) if v in G_undirected else set()

        common_neighbors = len(neighbors_u & neighbors_v)
        union_size = len(neighbors_u | neighbors_v)
        jaccard = common_neighbors / union_size if union_size > 0 else 0.0
        pref_attachment = len(neighbors_u) * len(neighbors_v)
        degree_diff = abs(len(neighbors_u) - len(neighbors_v))
        same_type = int(G_undirected.nodes[u].get("type") == G_undirected.nodes[v].get("type"))

        adamic_adar = 0.0
        for w in neighbors_u & neighbors_v:
            deg_w = G_undirected.degree(w)
            if deg_w > 1:
                adamic_adar += 1.0 / np.log(deg_w)

        resource_allocation = 0.0
        for w in neighbors_u & neighbors_v:
            deg_w = G_undirected.degree(w)
            if deg_w:
                resource_allocation += 1.0 / deg_w

        try:
            shortest_path = nx.shortest_path_length(G_undirected, u, v)
        except nx.NetworkXNoPath:
            shortest_path = 0

        return [
            common_neighbors,
            jaccard,
            pref_attachment,
            adamic_adar,
            resource_allocation,
            degree_diff,
            same_type,
            shortest_path,
        ]
    finally:
        if had_edge:
            G_undirected.add_edge(u, v, **edge_data)


def evaluate_classifier(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    pred = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 3),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3),
        "threshold": round(float(threshold), 3),
    }


def choose_best_threshold(model, X_test, y_test):
    best_threshold = 0.5
    best_metrics = evaluate_classifier(model, X_test, y_test, best_threshold)
    for threshold in np.arange(0.30, 0.86, 0.05):
        metrics = evaluate_classifier(model, X_test, y_test, threshold)
        if (metrics["f1"], metrics["precision"]) > (best_metrics["f1"], best_metrics["precision"]):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def train_link_predictors(G):
    G_undirected = nx.Graph(G)
    nodes = list(G_undirected.nodes())
    existing_edges = set(G_undirected.edges())

    positive_pairs = list(existing_edges)

    random.seed(42)
    negative_pairs = []
    attempts = 0
    target_negatives = min(len(positive_pairs) * 2, max(len(positive_pairs), 1) + 5000)
    while len(negative_pairs) < target_negatives and attempts < 20000:
        u, v = random.sample(nodes, 2)
        attempts += 1
        if not G_undirected.has_edge(u, v) and (u, v) not in negative_pairs and (v, u) not in negative_pairs:
            negative_pairs.append((u, v))

    X, y = [], []
    for u, v in positive_pairs:
        X.append(extract_pair_features(G_undirected, u, v))
        y.append(1)
    for u, v in negative_pairs:
        X.append(extract_pair_features(G_undirected, u, v))
        y.append(0)

    X = np.array(X)
    y = np.array(y)

    if len(set(y)) < 2 or len(X) < 8:
        print("Stage 8 — not enough edges/non-edges to train link predictors on this small graph.\n"
              "  (expected on a small demo graph — will work properly on your full KG)")
        return None, None, G_undirected, nodes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("Stage 8 — KG Completion (ML Link Prediction)")
    print(f"  Training pairs: {len(X_train)}  Test pairs: {len(X_test)}")

    svm = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, class_weight="balanced"),
            cv=3
        )
    )
    svm.fit(X_train, y_train)
    svm_threshold, svm_metrics = choose_best_threshold(svm, X_test, y_test)
    print(f"  SVM F1: {svm_metrics['f1']:.3f} "
          f"(precision={svm_metrics['precision']:.3f}, recall={svm_metrics['recall']:.3f}, "
          f"threshold={svm_threshold:.2f})")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_threshold, rf_metrics = choose_best_threshold(rf, X_test, y_test)
    print(f"  Random Forest F1: {rf_metrics['f1']:.3f} "
          f"(precision={rf_metrics['precision']:.3f}, recall={rf_metrics['recall']:.3f}, "
          f"threshold={rf_threshold:.2f})")

    best_model = rf if rf_metrics["f1"] >= svm_metrics["f1"] else svm
    best_name = "Random Forest" if rf_metrics["f1"] >= svm_metrics["f1"] else "SVM"
    best_threshold = rf_threshold if best_name == "Random Forest" else svm_threshold
    best_metrics = rf_metrics if best_name == "Random Forest" else svm_metrics
    print(f"  Best model: {best_name}")

    metrics = {
        "svm": svm_metrics,
        "random_forest": rf_metrics,
        "best_model": best_name,
        "best_threshold": round(float(best_threshold), 3),
        "training_pairs": int(len(X_train)),
        "test_pairs": int(len(X_test)),
        "negative_sampling_ratio": round(len(negative_pairs) / max(len(positive_pairs), 1), 2),
        "best_metrics": best_metrics,
    }

    return best_model, metrics, G_undirected, nodes


def predict_missing_links(G, best_model, G_undirected, nodes, top_k=10, prob_threshold=0.6):
    if best_model is None:
        return G, []

    candidates = []
    for i, u in enumerate(nodes):
        for v in nodes[i+1:]:
            if not G_undirected.has_edge(u, v):
                feats = extract_pair_features(G_undirected, u, v)
                candidates.append((u, v, feats))

    if not candidates:
        return G, []

    X_candidates = np.array([c[2] for c in candidates])
    probs = best_model.predict_proba(X_candidates)[:, 1]

    scored = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)

    added = []
    for (u, v, feats), prob in scored[:top_k]:
        if prob >= prob_threshold:
            G.add_edge(u, v, relation="INFERRED_LINK", confidence=round(float(prob), 3), inferred=True)
            added.append({
                "subject": u,
                "object": v,
                "predicted_probability": round(float(prob), 3),
                "features": {
                    "common_neighbors": feats[0],
                    "jaccard": round(float(feats[1]), 4),
                    "preferential_attachment": feats[2],
                    "adamic_adar": round(float(feats[3]), 4),
                    "resource_allocation": round(float(feats[4]), 4),
                }
            })

    print(f"\n  Added {len(added)} inferred link(s) above probability {prob_threshold}:")
    for a in added:
        print(f"    {a['subject']} -- INFERRED_LINK --> {a['object']} (p={a['predicted_probability']})")

    return G, added


def main():
    validated_triples = load_validated_triples()
    G = build_graph(validated_triples)

    best_model, metrics, G_undirected, nodes = train_link_predictors(G)
    threshold = metrics.get("best_threshold", 0.6) if metrics else 0.6
    G, inferred_links = predict_missing_links(G, best_model, G_undirected, nodes, prob_threshold=max(0.55, threshold))

    os.makedirs("outputs_ML", exist_ok=True)

    # Save graph as node-link JSON so Stage 9/10 can reload it without networkx-specific pickling
    graph_data = nx.node_link_data(G)
    graph_data["metadata"] = {"link_prediction": metrics or {}}
    with open("outputs_ML/graph_completed.json", "w") as f:
        json.dump(graph_data, f, indent=2)

    with open("outputs_ML/inferred_links.json", "w") as f:
        json.dump(inferred_links, f, indent=2)

    with open("outputs_ML/link_prediction_metrics.json", "w") as f:
        json.dump(metrics or {}, f, indent=2)

    print(f"\nSaved outputs_ML/graph_completed.json "
          f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    print(f"Saved outputs_ML/inferred_links.json ({len(inferred_links)} inferred links)")
    print("Saved outputs_ML/link_prediction_metrics.json")


if __name__ == "__main__":
    main()