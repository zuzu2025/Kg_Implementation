import json
import os
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
    
    # Generic words to skip
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
    """
    Build BM25 indexes with BLOCKING by first letter.
    Instead of one giant index per type,
    we build one index per type per first letter.
    
    So instead of searching 50,000 PARTY entries:
    → search only ~1,900 PARTY entries starting with same letter
    → 26x faster!
    """
    bm25_indexes = {}
    
    for label, entries in knowledge_base.items():
        bm25_indexes[label] = {}
        
        # Group entries by first letter
        by_letter = defaultdict(list)
        for entry in entries:
            first_letter = entry["canonical"][0] if entry["canonical"] else "#"
            by_letter[first_letter].append(entry)
        
        # Build one BM25 index per letter block
        for letter, letter_entries in by_letter.items():
            tokenized = [e["canonical"].lower().split() for e in letter_entries]
            if tokenized:
                bm25_indexes[label][letter] = {
                    "index": BM25Okapi(tokenized),
                    "entries": letter_entries
                }
        
        print(f"  {label}: {len(entries)} entries → {len(by_letter)} letter blocks")
    
    return bm25_indexes


def link_entity(mention, label, bm25_indexes, threshold=0.2):
    """
    Link mention to canonical using blocked BM25.
    Only search within same first-letter block.
    """
    if label not in bm25_indexes:
        return mention
    
    # Get first letter of mention for blocking
    first_letter = mention[0].lower() if mention else "#"
    
    label_index = bm25_indexes[label]
    
    if first_letter not in label_index:
        return mention  # no KB entries with same first letter
    
    block = label_index[first_letter]
    bm25 = block["index"]
    entries = block["entries"]
    
    query_tokens = mention.lower().split()
    scores = bm25.get_scores(query_tokens)
    
    best_idx = scores.argmax()
    best_score = scores[best_idx]
    
    if best_score >= threshold:
        return entries[best_idx]["canonical"]
    else:
        return mention

# ── STEP 3: LINK MENTIONS TO CANONICAL ENTITIES ───────────────────────────────

def link_all_entities(crf_entities_path, bm25_indexes):
    """
    Run entity linking on all CRF-extracted entities.
    For each mention → find canonical form using BM25.
    """
    with open(crf_entities_path, 'r') as f:
        all_results = json.load(f)
    
    linked_results = []
    total_linked = 0
    total_entities = 0
    
    for contract in all_results:
        linked_entities = []
        
        for entity in contract["entities"]:
            mention = entity["text"].strip().lower()
            label = entity["label"]
            
            # Find canonical form
            canonical = link_entity(mention, label, bm25_indexes)
            
            linked_entities.append({
                "original_mention": entity["text"],
                "canonical": canonical,
                "label": label,
                "linked": canonical != mention  # was it actually linked?
            })
            
            total_entities += 1
            if canonical != mention:
                total_linked += 1
        
        linked_results.append({
            "filename": contract["filename"],
            "entities": linked_entities
        })
    
    return linked_results, total_linked, total_entities

# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate_linking(linked_results):
    """
    Show linking statistics — how many mentions were successfully linked.
    Also show examples of linking decisions.
    """
    by_type = defaultdict(lambda: {"total": 0, "linked": 0, "examples": []})
    
    for contract in linked_results:
        for entity in contract["entities"]:
            label = entity["label"]
            by_type[label]["total"] += 1
            if entity["linked"]:
                by_type[label]["linked"] += 1
                if len(by_type[label]["examples"]) < 3:
                    by_type[label]["examples"].append(
                        f'{entity["original_mention"]} → {entity["canonical"]}'
                    )
    
    print("\nEntity Linking Results:")
    print("=" * 50)
    for label, stats in by_type.items():
        rate = stats["linked"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"\n{label}:")
        print(f"  Total: {stats['total']}, Linked: {stats['linked']} ({rate:.1f}%)")
        if stats["examples"]:
            print(f"  Examples:")
            for ex in stats["examples"]:
                print(f"    {ex}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CRF_ENTITIES_PATH = "outputs_ML/crf_all_entities.json"
    OUTPUT_PATH = "outputs_ML/linked_entities.json"
    KB_PATH = "outputs_ML/knowledge_base.json"
    
    print("Step 1: Building knowledge base...")
    knowledge_base = build_knowledge_base(CRF_ENTITIES_PATH)
    
    # Save knowledge base
    with open(KB_PATH, 'w') as f:
        json.dump(knowledge_base, f, indent=2)
    
    total_kb = sum(len(v) for v in knowledge_base.values())
    print(f"Knowledge base built: {total_kb} canonical entries")
    for label, entries in knowledge_base.items():
        print(f"  {label}: {len(entries)} entries")
    
    print("\nStep 2: Building BM25 indexes...")
    bm25_indexes = build_bm25_index(knowledge_base)
    print(f"BM25 indexes built for {len(bm25_indexes)} entity types")
    
    print("\nStep 3: Linking all entities...")
    linked_results, total_linked, total_entities = link_all_entities(
        CRF_ENTITIES_PATH, bm25_indexes
    )
    
    # Save results
    os.makedirs("outputs_ML", exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(linked_results, f, indent=2)
    
    print(f"\nLinking complete!")
    print(f"Total entities: {total_entities}")
    print(f"Successfully linked: {total_linked} ({total_linked/total_entities*100:.1f}%)")
    
    evaluate_linking(linked_results)
    print(f"\nSaved to {OUTPUT_PATH}")