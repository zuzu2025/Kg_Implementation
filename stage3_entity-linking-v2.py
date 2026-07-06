import json
import os
import Levenshtein
from collections import defaultdict
from rank_bm25 import BM25Okapi

# ── STEP 1: BUILD KNOWLEDGE BASE ─────────────────────────────────────────────

def build_knowledge_base(crf_entities_path):
    with open(crf_entities_path, 'r') as f:
        all_results = json.load(f)
    
    mention_counts = defaultdict(lambda: defaultdict(int))
    
    for contract in all_results:
        for entity in contract["entities"]:
            mention = entity["text"].strip().lower()
            label = entity["label"]
            if len(mention) > 2:
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
                          bm25_weight=0.5, lev_weight=0.5, threshold=0.25):
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
    """
    if label not in bm25_indexes:
        return mention, 0.0
    
    first_letter = mention[0].lower() if mention else "#"
    label_index = bm25_indexes[label]
    
    if first_letter not in label_index:
        return mention, 0.0
    
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
        return best_canonical, best_score
    else:
        return mention, best_score

def link_all_entities(crf_entities_path, bm25_indexes, knowledge_base):
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
            
            canonical, score = link_entity_ensemble(
                mention, label, bm25_indexes, knowledge_base
            )
            
            linked_entities.append({
                "original_mention": entity["text"],
                "canonical": canonical,
                "label": label,
                "score": round(score, 3),
                "linked": canonical != mention
            })
            
            total_entities += 1
            if canonical != mention:
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
    by_type = defaultdict(lambda: {"total": 0, "linked": 0, "examples": []})
    
    for contract in linked_results:
        for entity in contract["entities"]:
            label = entity["label"]
            by_type[label]["total"] += 1
            if entity["linked"]:
                by_type[label]["linked"] += 1
                if len(by_type[label]["examples"]) < 3:
                    by_type[label]["examples"].append(
                        f'{entity["original_mention"]} → {entity["canonical"]} (score: {entity["score"]})'
                    )
    
    print("\nEntity Linking V2 Results (BM25 + Levenshtein Ensemble):")
    print("=" * 60)
    for label, stats in by_type.items():
        rate = stats["linked"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"\n{label}:")
        print(f"  Total: {stats['total']}, Linked: {stats['linked']} ({rate:.1f}%)")
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
    total = sum(len(c["entities"]) for c in v1_results)
    
    print(f"\n{'='*60}")
    print(f"COMPARISON: BM25 only vs BM25 + Levenshtein Ensemble")
    print(f"{'='*60}")
    print(f"V1 (BM25 only):              {v1_linked}/{total} ({v1_linked/total*100:.1f}%)")
    print(f"V2 (BM25 + Levenshtein):     {v2_linked}/{total} ({v2_linked/total*100:.1f}%)")
    print(f"Improvement:                 +{v2_linked-v1_linked} entities linked")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CRF_ENTITIES_PATH = "outputs_ML/crf_all_entities.json"
    OUTPUT_PATH = "outputs_ML/linked_entities_v2.json"
    KB_PATH = "outputs_ML/knowledge_base.json"
    V1_PATH = "outputs_ML/linked_entities.json"
    
    print("Step 1: Building knowledge base...")
    knowledge_base = build_knowledge_base(CRF_ENTITIES_PATH)
    
    with open(KB_PATH, 'w') as f:
        json.dump(knowledge_base, f, indent=2)
    
    total_kb = sum(len(v) for v in knowledge_base.values())
    print(f"Knowledge base: {total_kb} canonical entries")
    
    print("\nStep 2: Building BM25 indexes...")
    bm25_indexes = build_bm25_index(knowledge_base)
    
    print("\nStep 3: Linking with BM25 + Levenshtein ensemble...")
    linked_results, total_linked, total_entities = link_all_entities(
        CRF_ENTITIES_PATH, bm25_indexes, knowledge_base
    )
    
    os.makedirs("outputs_ML", exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(linked_results, f, indent=2)
    
    print(f"\nLinking complete!")
    print(f"Total entities:      {total_entities}")
    print(f"Successfully linked: {total_linked} ({total_linked/total_entities*100:.1f}%)")
    
    evaluate_linking(linked_results)
    compare_versions(V1_PATH, linked_results)
    
    print(f"\nSaved to {OUTPUT_PATH}")