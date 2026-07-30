import json
import os
import re
import random
import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import spacy

nlp = spacy.load("en_core_web_sm")

# ══════════════════════════════════════════════════════
# PART A — CLUSTER ENTITY TYPES
# ══════════════════════════════════════════════════════

def load_entities(path):
    with open(path, 'r') as f:
        results = json.load(f)
    entities = []
    for contract in results:
        for entity in contract["entities"]:
            entities.append(entity)
    return entities

def build_entity_tfidf(entities, max_per_type=500, max_words=8):
    """
    Convert entity mentions to TF-IDF vectors.
    Each entity mention = one document.
    TF-IDF captures what words are distinctive per entity.
    """
    # Sample entities per type for balance
    by_type = defaultdict(list)
    for e in entities:
        text = clean_text(e.get("text", ""))
        label = e.get("label", "UNKNOWN")
        if is_informative_mention(text):
            by_type[label].append(text)

    sampled_texts = []
    sampled_labels = []

    for label, texts in by_type.items():
        # Entity mentions here range from short canonical forms
        # ("Agreement", "OTS") to full 50+ word clauses (JURISDICTION
        # spans are often an entire governing-law sentence). Mixed
        # together, TF-IDF ends up clustering the long spans by shared
        # legal boilerplate ("shall", "the parties", "and") rather than
        # by type-distinctive vocabulary, which collapses every type into
        # one mixed dumping-ground cluster. Preferring short mentions
        # keeps the type-discovery signal clean; only fall back to longer
        # spans if a type genuinely has too few short mentions to sample.
        short_texts = [t for t in texts if len(t.split()) <= max_words]
        pool = short_texts if len(short_texts) >= min(50, max_per_type) else texts
        if len(pool) > max_per_type:
            rng = random.Random(42)
            sample = rng.sample(pool, max_per_type)
        else:
            sample = pool
        sampled_texts.extend(sample)
        sampled_labels.extend([label] * len(sample))

    print(f"Building TF-IDF vectors for {len(sampled_texts)} entities...")

    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        max_features=1000,
        min_df=2,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(sampled_texts)
    X = normalize(X)  # normalize for better clustering

    return X, sampled_labels, sampled_texts, vectorizer


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def is_informative_mention(text):
    """Remove fragments that usually hurt clustering and downstream F1."""
    if not text or len(text) < 2:
        return False
    if len(text) == 1:
        return False
    if re.fullmatch(r"[\W_]+", text):
        return False
    return True


def cluster_quality(cluster_labels, original_labels):
    clusters = defaultdict(list)
    for cluster_id, label in zip(cluster_labels, original_labels):
        clusters[cluster_id].append(label)

    correct = 0
    total = 0
    for labels in clusters.values():
        counts = Counter(labels)
        correct += counts.most_common(1)[0][1]
        total += len(labels)
    return correct / total if total else 0.0

def cluster_entities_kmeans(X, n_clusters=8):
    """
    K-Means clustering on entity vectors.
    
    K-Means algorithm:
    1. Randomly place K centroids
    2. Assign each entity to nearest centroid
    3. Move centroid to mean of its assigned entities
    4. Repeat until stable
    
    n_clusters = how many entity type groups to find
    """
    print(f"\nRunning K-Means with {n_clusters} clusters...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    
    kmeans.fit(X)
    labels = kmeans.labels_
    
    # Silhouette score — measures cluster quality
    # Range: -1 to 1, higher = better separated clusters
    if len(set(labels)) > 1:
        score = silhouette_score(X, labels, sample_size=1000)
        print(f"Silhouette Score: {score:.3f}")
    
    return kmeans, labels

def cluster_entities_hac(X, n_clusters=8):
    """
    Hierarchical Agglomerative Clustering.
    
    HAC algorithm:
    1. Start: every entity is its own cluster
    2. Find two most similar clusters
    3. Merge them into one
    4. Repeat until n_clusters remain
    
    Bottom-up approach — builds a tree (dendrogram)
    More flexible than K-Means — doesn't assume spherical clusters
    """
    print(f"\nRunning HAC with {n_clusters} clusters...")
    
    # Convert sparse to dense for HAC
    X_dense = X.toarray()
    
    hac = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'  # minimize variance within clusters
    )
    
    hac_labels = hac.fit_predict(X_dense)
    
    if len(set(hac_labels)) > 1:
        score = silhouette_score(X_dense, hac_labels, sample_size=1000)
        print(f"Silhouette Score: {score:.3f}")
    
    return hac, hac_labels

def analyze_entity_clusters(cluster_labels, entity_texts, original_labels, algorithm_name):
    """
    Show what each cluster contains — what entity types grouped together?
    """
    print(f"\n{algorithm_name} Entity Clusters:")
    print("=" * 50)
    
    clusters = defaultdict(list)
    for i, cluster_id in enumerate(cluster_labels):
        clusters[cluster_id].append({
            "text": entity_texts[i],
            "original_label": original_labels[i]
        })
    
    ontology_entity_types = {}
    
    for cluster_id, members in sorted(clusters.items()):
        # Count original labels in this cluster
        label_counts = Counter(m["original_label"] for m in members)
        dominant_label = label_counts.most_common(1)[0][0]
        
        # Sample members
        sample_texts = [m["text"] for m in members[:5]]
        
        print(f"\nCluster {cluster_id} → dominant: {dominant_label}")
        print(f"  Size: {len(members)}")
        print(f"  Label mix: {dict(label_counts)}")
        print(f"  Examples: {sample_texts}")
        
        ontology_entity_types[f"CLUSTER_{cluster_id}"] = {
            "dominant_type": dominant_label,
            "size": len(members),
            "label_distribution": dict(label_counts),
            "examples": sample_texts
        }
    
    return ontology_entity_types

# ══════════════════════════════════════════════════════
# PART B — CLUSTER RELATION TYPES
# ══════════════════════════════════════════════════════

def load_relations(path):
    with open(path, 'r') as f:
        return json.load(f)

def build_relation_tfidf(relations, max_relations=5000):
    """
    Convert relation predicates to TF-IDF vectors.
    Each predicate = one document.
    Clustering discovers higher-level relation categories.
    """
    filtered = [
        r for r in relations
        if r.get("relation", "OTHER") != "OTHER"
        and is_informative_mention(r.get("predicate", ""))
        and is_informative_mention(r.get("entity1", ""))
        and is_informative_mention(r.get("entity2", ""))
    ]
    if len(filtered) > max_relations:
        # Random sample instead of filtered[:max_relations] — taking the
        # first N is still order-biased toward whichever contracts were
        # processed first in stage5. A fixed seed keeps this reproducible.
        rng = random.Random(42)
        sampled = rng.sample(filtered, max_relations)
    else:
        sampled = filtered
    predicates = [clean_text(r["predicate"]).lower() for r in sampled]
    relation_types = [r["relation"] for r in sampled]
    
    print(f"\nBuilding TF-IDF vectors for {len(predicates)} relation predicates...")
    
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',  # character n-grams — better for short predicates
        ngram_range=(2, 5),
        max_features=600,
        sublinear_tf=True
    )
    
    X = vectorizer.fit_transform(predicates)
    X = normalize(X)
    
    return X, relation_types, predicates, vectorizer

def cluster_relations_kmeans(X, n_clusters=6):
    print(f"\nRunning K-Means on relations with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    
    if len(set(labels)) > 1:
        score = silhouette_score(X, labels, sample_size=min(1000, X.shape[0]))
        print(f"Silhouette Score: {score:.3f}")
    
    return kmeans, labels

def cluster_relations_hac(X, n_clusters=6):
    print(f"\nRunning HAC on relations with {n_clusters} clusters...")
    
    X_dense = X.toarray()
    
    hac = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    hac_labels = hac.fit_predict(X_dense)
    
    if len(set(hac_labels)) > 1:
        score = silhouette_score(X_dense, hac_labels, sample_size=min(1000, X_dense.shape[0]))
        print(f"Silhouette Score: {score:.3f}")
    
    return hac, hac_labels

def analyze_relation_clusters(cluster_labels, predicates, relation_types, algorithm_name):
    """
    Show what each relation cluster contains.
    Discover higher-level relation categories.
    """
    print(f"\n{algorithm_name} Relation Clusters:")
    print("=" * 50)
    
    clusters = defaultdict(list)
    for i, cluster_id in enumerate(cluster_labels):
        clusters[cluster_id].append({
            "predicate": predicates[i],
            "relation_type": relation_types[i]
        })
    
    ontology_relations = {}
    
    for cluster_id, members in sorted(clusters.items()):
        type_counts = Counter(m["relation_type"] for m in members)
        dominant_type = type_counts.most_common(1)[0][0]
        sample_predicates = list(set(m["predicate"] for m in members[:10]))[:5]
        
        print(f"\nCluster {cluster_id} → dominant: {dominant_type}")
        print(f"  Size: {len(members)}")
        print(f"  Type mix: {dict(type_counts)}")
        print(f"  Sample predicates: {sample_predicates}")
        
        ontology_relations[f"REL_CLUSTER_{cluster_id}"] = {
            "dominant_type": dominant_type,
            "size": len(members),
            "type_distribution": dict(type_counts),
            "sample_predicates": sample_predicates
        }
    
    return ontology_relations
def induce_domain_rules(relations_path, min_count=5, min_support=0.08, min_confidence=0.60):
    """
    Automatically induce domain rules from relation data.
    
    For each relation type → find most common entity type pairs.
    Most frequent pairs become domain rules.
    
    No hardcoding — purely data-driven!
    """
    with open(relations_path, 'r') as f:
        relations = json.load(f)
    
    # Count entity type pairs per relation type
    rule_counts = defaultdict(Counter)
    
    for rel in relations:
        relation_type = rel.get("relation", "OTHER")
        if relation_type == "OTHER":
            continue
        if not is_informative_mention(rel.get("entity1", "")) or not is_informative_mention(rel.get("entity2", "")):
            continue
        
        label1 = rel.get("label1", "UNKNOWN")
        label2 = rel.get("label2", "UNKNOWN")
        
        # Count this subject-relation-object triple
        rule_counts[relation_type][(label1, label2)] += 1
    
    # Build domain rules from most frequent pairs
    domain_rules = []
    
    print("\nInduced Domain Rules:")
    print("=" * 50)
    
    for relation_type, pair_counts in sorted(rule_counts.items()):
        relation_total = sum(pair_counts.values())
        top_pairs = pair_counts.most_common(3)
        
        print(f"\n{relation_type}:")
        for (subj_type, obj_type), count in top_pairs:
            support = count / relation_total if relation_total else 0.0
            subject_total = sum(c for (s, _), c in pair_counts.items() if s == subj_type)
            confidence = count / max(subject_total, 1)
            if count >= min_count and support >= min_support and confidence >= min_confidence:
                rule = {
                    "subject": subj_type,
                    "relation": relation_type,
                    "object": obj_type,
                    "frequency": count,
                    "support": round(support, 4),
                    "confidence": round(confidence, 4),
                    "induced": True  # flag as data-driven
                }
                domain_rules.append(rule)
                print(f"  {subj_type} --[{relation_type}]--> {obj_type} "
                      f"({count} times, support={support:.2f}, confidence={confidence:.2f})")
    
    print(f"\nTotal induced rules: {len(domain_rules)}")
    return domain_rules
# ══════════════════════════════════════════════════════
# PART C — BUILD FINAL ONTOLOGY
# ══════════════════════════════════════════════════════

def build_ontology(entity_clusters_kmeans, entity_clusters_hac,
                   relation_clusters_kmeans, relation_clusters_hac,
                   relations_path):
    
    entity_types = set()
    for cluster in entity_clusters_kmeans.values():
        entity_types.add(cluster["dominant_type"])
    
    relation_categories = {}
    for cluster_id, cluster in relation_clusters_kmeans.items():
        relation_categories[cluster_id] = {
            "category": cluster["dominant_type"],
            "predicates": cluster["sample_predicates"]
        }
    
    # Induce domain rules from data — no hardcoding!
    domain_rules = induce_domain_rules(relations_path)
    
    ontology = {
        "entity_types": list(entity_types),
        "relation_categories": relation_categories,
        "domain_rules": domain_rules,
        "metadata": {
            "source": "Induced from CUAD legal contracts",
            "method": "K-Means + HAC clustering + data-driven rule induction",
            "entity_clusters_kmeans": len(entity_clusters_kmeans),
            "entity_clusters_hac": len(entity_clusters_hac),
            "relation_clusters_kmeans": len(relation_clusters_kmeans),
            "relation_clusters_hac": len(relation_clusters_hac),
            "domain_rules_induced": True,
            "domain_rule_min_support": 0.08,
            "relation_rule_min_confidence": 0.60
        }
    }
    
    return ontology

# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    ENTITIES_PATH = "outputs_ML/crf_all_entities.json"
    RELATIONS_PATH = "outputs_ML/relations.json"
    OUTPUT_PATH = "outputs_ML/ontology.json"

    # ── PART A: Entity Clustering ──
    print("=" * 60)
    print("PART A: ENTITY TYPE CLUSTERING")
    print("=" * 60)
    
    entities = load_entities(ENTITIES_PATH)
    print(f"Total entities loaded: {len(entities)}")
    
    X_entities, original_labels, entity_texts, entity_vectorizer = build_entity_tfidf(entities)
    
    # K-Means
    kmeans_entity, kmeans_entity_labels = cluster_entities_kmeans(X_entities, n_clusters=8)
    entity_clusters_kmeans = analyze_entity_clusters(
        kmeans_entity_labels, entity_texts, original_labels, "K-Means"
    )
    
    # HAC
    hac_entity, hac_entity_labels = cluster_entities_hac(X_entities, n_clusters=8)
    entity_clusters_hac = analyze_entity_clusters(
        hac_entity_labels, entity_texts, original_labels, "HAC"
    )
    
    # ── PART B: Relation Clustering ──
    print("\n" + "=" * 60)
    print("PART B: RELATION TYPE CLUSTERING")
    print("=" * 60)
    
    relations = load_relations(RELATIONS_PATH)
    print(f"Total relations loaded: {len(relations)}")
    
    X_relations, relation_types, predicates, rel_vectorizer = build_relation_tfidf(relations)
    
    # K-Means
    kmeans_rel, kmeans_rel_labels = cluster_relations_kmeans(X_relations, n_clusters=6)
    relation_clusters_kmeans = analyze_relation_clusters(
        kmeans_rel_labels, predicates, relation_types, "K-Means"
    )
    
    # HAC
    hac_rel, hac_rel_labels = cluster_relations_hac(X_relations, n_clusters=6)
    relation_clusters_hac = analyze_relation_clusters(
        hac_rel_labels, predicates, relation_types, "HAC"
    )
    
    # ── PART C: Build Ontology ──
    print("\n" + "=" * 60)
    print("PART C: BUILDING FINAL ONTOLOGY")
    print("=" * 60)
    
    ontology = build_ontology(
        entity_clusters_kmeans, entity_clusters_hac,
        relation_clusters_kmeans, relation_clusters_hac,
        RELATIONS_PATH
    )
    
    os.makedirs("outputs_ML", exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(ontology, f, indent=2)
    
    print(f"\nOntology saved to {OUTPUT_PATH}")
    print(f"Entity types discovered: {len(ontology['entity_types'])}")
    print(f"Relation categories: {len(ontology['relation_categories'])}")
    print(f"Domain rules: {len(ontology['domain_rules'])}")