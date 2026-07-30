import json
import os
import pickle
import re
import Levenshtein
from pathlib import Path
from collections import defaultdict
from rank_bm25 import BM25Okapi
from kg_no_llm_algorithms import cluster_entities, mention_quality

# Reuse the exact feature extraction used to train the CRF model, so
# predictions here are built the same way the model was trained.
from stage2_ner_crf import sentence_to_features

# ── STEP 0: RUN CRF MODEL OVER ALL CONTRACTS ─────────────────────────────────

def generate_crf_entities(data_dir, model_path, output_path):
    """
    Loads the trained CRF model (from stage2) and runs it over every raw
    contract in data_dir, producing outputs_ML/crf_all_entities.json in the
    shape stages 3/5/6 expect:
        [{"filename": ..., "entities": [{"text": ..., "label": ...}, ...]}, ...]

    This step didn't exist as a separate script, so it's folded in here as
    the first thing stage 3 does — no separate "stage 2.5" file needed.
    """
    print(f"Loading CRF model from {model_path}...")
    with open(model_path, 'rb') as f:
        crf = pickle.load(f)

    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.txt'))
    print(f"Running CRF over {len(files)} contracts...")

    all_results = []
    for i, fname in enumerate(files):
        fpath = os.path.join(data_dir, fname)
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as fh:
            text = fh.read()

        # Same sentence splitting used elsewhere at inference time
        # (predict_entities in stage2). Entity offsets aren't needed here —
        # only the extracted text/label pairs — so simple splitting is fine.
        sentences = [s.strip() for s in re.split(r'[.\n]+', text) if s.strip()]

        entities = []
        for sent in sentences:
            words = sent.split()
            if not words:
                continue

            features = sentence_to_features(words)
            labels = crf.predict([features])[0]

            current_entity = None
            for word, label in zip(words, labels):
                if label.startswith('B-'):
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {'text': word, 'label': label[2:]}
                elif label.startswith('I-') and current_entity:
                    current_entity['text'] += ' ' + word
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            if current_entity:
                entities.append(current_entity)

        all_results.append({"filename": fname, "entities": entities})

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(files)} contracts...")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_file.with_suffix(output_file.suffix + ".tmp")
    with tmp_output.open('w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    os.replace(tmp_output, output_file)

    total_entities = sum(len(r["entities"]) for r in all_results)
    print(f"CRF entity extraction complete: {total_entities} entities across {len(all_results)} contracts")
    print(f"Saved to {output_path}")

    return all_results

# ── STEP 1: BUILD KNOWLEDGE BASE ─────────────────────────────────────────────

def build_knowledge_base(crf_entities_path):
    with open(crf_entities_path, 'r') as f:
        all_results = json.load(f)
    
    mention_counts = defaultdict(lambda: defaultdict(int))
    
    for contract in all_results:
        for entity in contract["entities"]:
            mention = entity["text"].strip().lower()
            label = entity["label"]
            if len(mention) > 2 and mention_quality(mention) > 0.0:
                mention_counts[label][mention] += 1
    
    print("Mention counts per type:")
    for label, mentions in mention_counts.items():
        print(f"  {label}: {len(mentions)} unique mentions")
    
    skip_words = {
        'the company', 'the party', 'the parties', 'the licensee',
        'the licensor', 'the vendor', 'the client', 'the customer',
        'the employer', 'the employee', 'the contractor', 'the agent',
        'the distributor', 'the supplier', 'the buyer', 'the seller',
        'it', 'they', 'we', 'you', 'he', 'she', 'them', 'its'
    }
    
    knowledge_base = {}
    
    for label, mentions in mention_counts.items():
        sorted_mentions = sorted(mentions.items(), key=lambda x: x[1], reverse=True)
        
        kb_entries = []
        for mention, count in sorted_mentions:
            if mention.lower() in skip_words:
                continue
            if count < 2:
                continue
            kb_entries.append({
                "canonical": mention,
                "label": label,
                "frequency": count
            })
        
        knowledge_base[label] = kb_entries
    
    return knowledge_base

# ── STEP 2: BUILD BM25 INDEX ──────────────────────────────────────────────────

def build_bm25_index(knowledge_base):
    bm25_indexes = {}
    
    for label, entries in knowledge_base.items():
        bm25_indexes[label] = {}
        
        by_letter = defaultdict(list)
        for entry in entries:
            first_letter = entry["canonical"][0] if entry["canonical"] else "#"
            by_letter[first_letter].append(entry)
        
        for letter, letter_entries in by_letter.items():
            tokenized = [e["canonical"].lower().split() for e in letter_entries]
            if tokenized:
                bm25_indexes[label][letter] = {
                    "index": BM25Okapi(tokenized),
                    "entries": letter_entries
                }
        
        print(f"  {label}: {len(entries)} entries → {len(by_letter)} letter blocks")
    
    return bm25_indexes

# ── SIMILARITY FUNCTIONS ──────────────────────────────────────────────────────

def levenshtein_similarity(s1, s2):
    """
    How similar are two strings character by character?
    1.0 = identical, 0.0 = completely different
    
    Example:
    "june 8 2010" vs "june 8, 2010"
    Only 1 character different → similarity = 0.92
    """
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    distance = Levenshtein.distance(s1.lower(), s2.lower())
    return 1 - (distance / max_len)

_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')

def years_conflict(mention, candidate):
    """
    True if both strings contain a year and those years differ.
    Character-level similarity (Levenshtein) can score "...1999" and
    "...2018" as near-identical since only 2 digits differ, but these
    are different dates and should never be linked as the same entity.
    """
    m_years = set(re.findall(r'\b(?:19|20)\d{2}\b', mention))
    c_years = set(re.findall(r'\b(?:19|20)\d{2}\b', candidate))
    if not m_years or not c_years:
        return False
    return m_years.isdisjoint(c_years)

def bm25_score_normalized(bm25, query_tokens, max_possible=10.0):
    """
    Get normalized BM25 score between 0 and 1.
    BM25 raw scores have no upper bound — we normalize by dividing by max_possible.
    """
    scores = bm25.get_scores(query_tokens)
    best_idx = scores.argmax()
    best_score = scores[best_idx]
    normalized = min(best_score / max_possible, 1.0)  # cap at 1.0
    return best_idx, normalized, scores

# ── STEP 3: LINK WITH ENSEMBLE ────────────────────────────────────────────────

def link_entity_ensemble(mention, label, bm25_indexes, knowledge_base,
                          bm25_weight=0.5, lev_weight=0.5, threshold=0.72):
    """
    Link entity using ENSEMBLE of BM25 + Levenshtein.

    For each KB candidate:
        final_score = (bm25_score × 0.5) + (levenshtein_score × 0.5)

    Pick candidate with highest final_score above threshold.

    Why ensemble?
    - BM25 alone: good for long text, bad for short mentions like dates
    - Levenshtein alone: good for short mentions, bad for completely different wordings
    - Together: covers both cases!

    Example:
    mention = "june 8 2010"

    BM25 score:        0.3  (word overlap ok but not great)
    Levenshtein score: 0.9  (very similar characters!)
    Ensemble score:    0.6  → linked! ✅

    PATCH: threshold was 0.25 here, but ER_THRESHOLD used for the actual
    entity-resolution clustering a few functions down (cluster_entities) is
    0.72 — same file, same corpus, same notion of "is this the same entity",
    two very different bars. At 0.25, whatever mention happens to be the
    most frequent (and therefore top) BM25 candidate in a letter-block —
    often a generic or truncated CRF fragment like "agreement" or a cut-off
    company name — becomes a magnet that unrelated mentions from OTHER
    documents get silently rewritten to. That's the actual source of the
    fan-out hubs showing up downstream in relations.json (e.g. "agreement"
    IS_A 22 different objects) — those are 22 different real entities that
    got merged into one canonical string here, before stage5/stage7 ever
    see them. Raising this to match the 0.72 bar used for real ER stops the
    over-merging at the source instead of only patching its symptoms later
    in the pipeline.
    """
    if label not in bm25_indexes:
        return mention, 0.0, False
    
    first_letter = mention[0].lower() if mention else "#"
    label_index = bm25_indexes[label]
    
    if first_letter not in label_index:
        return mention, 0.0, False
    
    block = label_index[first_letter]
    bm25 = block["index"]
    entries = block["entries"]
    
    query_tokens = mention.lower().split()
    
    # Get BM25 scores for all entries in this block
    raw_scores = bm25.get_scores(query_tokens)
    max_raw = raw_scores.max() if raw_scores.max() > 0 else 1.0
    
    best_score = 0.0
    best_canonical = mention
    
    # Only check top 20 BM25 candidates for Levenshtein
    # (avoids computing Levenshtein for ALL entries)
    top_indices = raw_scores.argsort()[-20:][::-1]
    
    for idx in top_indices:
        candidate = entries[idx]["canonical"]

        # For date-type labels, never link across a year mismatch — character
        # similarity alone ("...1999" vs "...2018") is not a valid signal that
        # two dates are the same entity.
        if label in ("DATE", "EFFECTIVE_DATE") and years_conflict(mention, candidate):
            continue

        # Normalize BM25 score
        bm25_s = raw_scores[idx] / max_raw if max_raw > 0 else 0.0
        
        # Levenshtein similarity
        lev_s = levenshtein_similarity(mention, candidate)
        
        # Ensemble score
        ensemble = (bm25_weight * bm25_s) + (lev_weight * lev_s)
        
        if ensemble > best_score:
            best_score = ensemble
            best_canonical = candidate
    
    if best_score >= threshold:
        return best_canonical, best_score, True
    else:
        return mention, best_score, False

def link_all_entities(crf_entities_path, bm25_indexes, knowledge_base, threshold=0.72):
    with open(crf_entities_path, 'r') as f:
        all_results = json.load(f)
    
    linked_results = []
    total_linked = 0
    total_entities = 0
    
    for i, contract in enumerate(all_results):
        linked_entities = []
        
        for entity in contract["entities"]:
            mention = entity["text"].strip().lower()
            label = entity["label"]

            if mention_quality(mention) == 0.0:
                continue
            
            canonical, score, matched = link_entity_ensemble(
                mention, label, bm25_indexes, knowledge_base, threshold=threshold
            )
            
            linked_entities.append({
                "original_mention": entity["text"],
                "canonical": canonical,
                "label": label,
                "score": round(score, 3),
                "linked": matched,
                "changed": matched and canonical != mention
            })
            
            total_entities += 1
            if matched:
                total_linked += 1
        
        linked_results.append({
            "filename": contract["filename"],
            "entities": linked_entities
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Linked {i+1}/510 contracts...")
    
    return linked_results, total_linked, total_entities

# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate_linking(linked_results):
    by_type = defaultdict(lambda: {"total": 0, "linked": 0, "changed": 0, "examples": []})
    
    for contract in linked_results:
        for entity in contract["entities"]:
            label = entity["label"]
            by_type[label]["total"] += 1
            if entity["linked"]:
                by_type[label]["linked"] += 1
                if entity.get("changed"):
                    by_type[label]["changed"] += 1
                if len(by_type[label]["examples"]) < 3:
                    by_type[label]["examples"].append(
                        f'{entity["original_mention"]} → {entity["canonical"]} (score: {entity["score"]})'
                    )
    
    print("\nEntity Linking V2 Results (BM25 + Levenshtein Ensemble):")
    print("=" * 60)
    for label, stats in by_type.items():
        rate = stats["linked"] / stats["total"] * 100 if stats["total"] > 0 else 0
        changed_rate = stats["changed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"\n{label}:")
        print(f"  Total: {stats['total']}, Linked: {stats['linked']} ({rate:.1f}%)")
        print(f"  Canonical text changed: {stats['changed']} ({changed_rate:.1f}%)")
        if stats["examples"]:
            print(f"  Examples:")
            for ex in stats["examples"]:
                print(f"    {ex}")

# ── COMPARISON ────────────────────────────────────────────────────────────────

def compare_versions(v1_path, v2_results):
    """Compare V1 (BM25 only) vs V2 (BM25 + Levenshtein)"""
    with open(v1_path, 'r') as f:
        v1_results = json.load(f)
    
    v1_linked = sum(1 for c in v1_results for e in c["entities"] if e["linked"])
    v2_linked = sum(1 for c in v2_results for e in c["entities"] if e["linked"])
    v2_changed = sum(1 for c in v2_results for e in c["entities"] if e.get("changed"))
    v1_total = sum(len(c["entities"]) for c in v1_results)
    v2_total = sum(len(c["entities"]) for c in v2_results)

    print(f"\n{'='*60}")
    print(f"COMPARISON: BM25 only vs BM25 + Levenshtein Ensemble")
    print(f"{'='*60}")
    if v1_total != v2_total:
        print(f"NOTE: V1 total ({v1_total}) != V2 total ({v2_total}) — these runs")
        print(f"were likely made against different crf_all_entities.json snapshots,")
        print(f"so the comparison below is directional only, not apples-to-apples.")
    print(f"V1 (BM25 only, changed):     {v1_linked}/{v1_total} ({v1_linked/v1_total*100:.1f}%)")
    print(f"V2 (resolved to KB):         {v2_linked}/{v2_total} ({v2_linked/v2_total*100:.1f}%)")
    print(f"V2 (canonical changed):      {v2_changed}/{v2_total} ({v2_changed/v2_total*100:.1f}%)")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR = "data"
    MODEL_PATH = "outputs_ML/crf_model_v2.pkl"
    CRF_ENTITIES_PATH = "outputs_ML/crf_all_entities.json"
    OUTPUT_PATH = "outputs_ML/linked_entities_v2.json"
    CLUSTERS_PATH = "outputs_ML/entity_clusters.json"
    ER_CANDIDATES_PATH = "outputs_ML/er_candidates.json"
    ER_THRESHOLD = 0.72
    KB_PATH = "outputs_ML/knowledge_base.json"
    V1_PATH = "outputs_ML/linked_entities.json"
    FORCE_REGENERATE_CRF = False

    if FORCE_REGENERATE_CRF or not os.path.exists(CRF_ENTITIES_PATH):
        print("Step 0: Running CRF model over all contracts...")
        generate_crf_entities(DATA_DIR, MODEL_PATH, CRF_ENTITIES_PATH)
    else:
        print(f"Step 0: Reusing existing CRF entities at {CRF_ENTITIES_PATH}")

    print("\nStep 1: Building knowledge base...")
    knowledge_base = build_knowledge_base(CRF_ENTITIES_PATH)
    
    with open(KB_PATH, 'w') as f:
        json.dump(knowledge_base, f, indent=2)
    
    total_kb = sum(len(v) for v in knowledge_base.values())
    print(f"Knowledge base: {total_kb} canonical entries")
    
    print("\nStep 2: Building BM25 indexes...")
    bm25_indexes = build_bm25_index(knowledge_base)
    
    print("\nStep 3: Linking with BM25 + Levenshtein ensemble...")
    linked_results, total_linked, total_entities = link_all_entities(
        CRF_ENTITIES_PATH, bm25_indexes, knowledge_base, threshold=ER_THRESHOLD
    )
    
    os.makedirs("outputs_ML", exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(linked_results, f, indent=2)

    print("\nStep 4: Entity Resolution with blocking + pairwise scoring...")
    clusters, er_candidates, _ = cluster_entities(linked_results, threshold=ER_THRESHOLD)
    with open(CLUSTERS_PATH, 'w') as f:
        json.dump(clusters, f, indent=2)
    with open(ER_CANDIDATES_PATH, 'w') as f:
        json.dump(er_candidates, f, indent=2)
    
    print(f"\nLinking complete!")
    print(f"Total entities:      {total_entities}")
    print(f"Successfully linked: {total_linked} ({total_linked/total_entities*100:.1f}%)")
    print(f"ER clusters:         {len(clusters)}")
    print(f"ER pairs scored:     {len(er_candidates)}")
    
    evaluate_linking(linked_results)

    if os.path.exists(V1_PATH):
        compare_versions(V1_PATH, linked_results)
    else:
        print(f"\n(Skipping V1 comparison — {V1_PATH} not found)")

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Saved ER clusters to {CLUSTERS_PATH}")
    print(f"Saved ER audit pairs to {ER_CANDIDATES_PATH}")
