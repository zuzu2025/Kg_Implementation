"""
Shared no-LLM algorithms for the staged KG pipeline.

These utilities keep the implementation classical/reproducible:
blocking, pairwise string/token/acronym features, thresholded ER scoring,
union-find clustering, and ontology-style type checks.
"""

import hashlib
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher


ENTITY_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "plc", "company", "gmbh", "ag", "sa", "lp", "llp",
}

DOMAIN_RANGE = {
    "PARTY_OF": {("PARTY", "CONTRACT")},
    "ENTERED_INTO": {("PARTY", "CONTRACT"), ("CONTRACT", "PARTY")},
    "EFFECTIVE_ON": {("CONTRACT", "EFFECTIVE_DATE"), ("CONTRACT", "DATE")},
    "GOVERNED_BY": {("CONTRACT", "JURISDICTION"), ("PARTY", "JURISDICTION")},
    "LOCATED_IN": {("PARTY", "JURISDICTION")},
    "ORGANIZED_UNDER": {("PARTY", "JURISDICTION")},
}


def stable_id(prefix, *parts):
    raw = "||".join(str(p) for p in parts).encode("utf-8", errors="ignore")
    return f"{prefix}_{hashlib.sha1(raw).hexdigest()[:12]}"


def normalize_mention(value):
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    return value.strip(" \t\r\n,.;:()[]{}\"'")


def canonical_key(value, label=None):
    value = normalize_mention(value).lower()
    value = re.sub(r"[^a-z0-9&' ]+", " ", value)
    tokens = [t for t in value.split() if t]
    if label == "PARTY":
        tokens = [t for t in tokens if t not in ENTITY_SUFFIXES]
    return " ".join(tokens)


def token_set(value):
    return set(canonical_key(value).split())


def acronym(value):
    tokens = [t for t in re.findall(r"[A-Za-z0-9]+", value)
              if t.lower() not in ENTITY_SUFFIXES]
    return "".join(t[0].lower() for t in tokens if t)


def years_conflict(left, right):
    left_years = set(re.findall(r"\b(?:19|20)\d{2}\b", str(left)))
    right_years = set(re.findall(r"\b(?:19|20)\d{2}\b", str(right)))
    return bool(left_years and right_years and left_years.isdisjoint(right_years))


def block_key(entity):
    label = entity.get("label", "UNKNOWN")
    mention = entity.get("canonical") or entity.get("text") or entity.get("original_mention") or ""
    key = canonical_key(mention, label)
    tokens = key.split()
    if not tokens:
        return label, "#"
    if label in {"DATE", "EFFECTIVE_DATE"}:
        year = re.search(r"\b(?:19|20)\d{2}\b", key)
        return label, year.group(0) if year else tokens[-1]
    # A slightly stronger blocking key keeps common legal words from creating
    # giant O(n^2) blocks such as CONTRACT/agreement or PARTY/the.
    second = tokens[1][:4] if len(tokens) > 1 else ""
    return label, tokens[0][:4], second


def pair_features(left, right):
    label = left.get("label") or right.get("label")
    left_text = left.get("canonical") or left.get("text") or left.get("original_mention") or ""
    right_text = right.get("canonical") or right.get("text") or right.get("original_mention") or ""
    left_key = canonical_key(left_text, label)
    right_key = canonical_key(right_text, label)
    left_tokens = token_set(left_text)
    right_tokens = token_set(right_text)
    union = left_tokens | right_tokens
    return {
        "char_sim": SequenceMatcher(None, left_key, right_key).ratio(),
        "token_jaccard": len(left_tokens & right_tokens) / len(union) if union else 0.0,
        "acronym_match": 1.0 if acronym(left_text) and acronym(left_text) == acronym(right_text) else 0.0,
        "same_file": 1.0 if left.get("filename") == right.get("filename") else 0.0,
        "len_ratio": min(len(left_key), len(right_key)) / max(len(left_key), len(right_key), 1),
    }


def has_hard_conflict(left, right):
    if left.get("label") != right.get("label"):
        return True
    left_text = left.get("canonical") or left.get("text") or left.get("original_mention") or ""
    right_text = right.get("canonical") or right.get("text") or right.get("original_mention") or ""
    if years_conflict(left_text, right_text):
        return True
    if left.get("label") == "PARTY":
        left_key = canonical_key(left_text, "PARTY")
        right_key = canonical_key(right_text, "PARTY")
        if len(left_key) <= 3 or len(right_key) <= 3:
            return left_key != right_key
    return False


def er_probability(features):
    raw = (
        2.4 * features["char_sim"]
        + 2.0 * features["token_jaccard"]
        + 1.5 * features["acronym_match"]
        + 0.4 * features["same_file"]
        + 0.7 * features["len_ratio"]
        - 3.1
    )
    return 1.0 / (1.0 + math.exp(-raw))


class UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def cluster_entities(linked_results, threshold=0.72, max_block_size=120,
                     max_neighbors_per_entity=10, max_candidate_audit=50000,
                     verbose=True):
    flat = []
    for contract in linked_results:
        for idx, entity in enumerate(contract.get("entities", [])):
            text = entity.get("canonical") or entity.get("original_mention") or entity.get("text") or ""
            row = {
                **entity,
                "mention_id": stable_id("m", contract.get("filename", ""), idx, text, entity.get("label", "")),
                "filename": contract.get("filename", ""),
            }
            flat.append(row)

    by_id = {e["mention_id"]: e for e in flat}
    uf = UnionFind(by_id)

    exact_groups = defaultdict(list)
    for entity in flat:
        label = entity.get("label")
        text = entity.get("canonical") or entity.get("text") or entity.get("original_mention") or ""
        exact_groups[(label, canonical_key(text, label))].append(entity)

    representatives = []
    exact_merges = 0
    for (label, key), group in exact_groups.items():
        rep = dict(group[0])
        rep["canonical"] = key
        rep["representative_size"] = len(group)
        representatives.append(rep)
        if len(group) >= 2:
            root = group[0]["mention_id"]
            for member in group[1:]:
                uf.union(root, member["mention_id"])
                exact_merges += 1

    blocks = defaultdict(list)
    for entity in representatives:
        blocks[block_key(entity)].append(entity)

    candidates = []
    skipped_blocks = []
    scored_pairs = 0
    # PATCH: this loop was silent from "Step 4: Entity Resolution..." all the
    # way to the final summary print — for a large entity set (PARTY alone
    # was 103,397 unique mentions in the last run) that's several minutes of
    # zero output, which looks identical to a hang. Print a heartbeat every
    # 200 blocks and every 50k scored pairs so progress is visible instead
    # of having to guess from Task Manager CPU usage.
    total_blocks = len(blocks)
    for block_idx, (block, block_entities) in enumerate(blocks.items()):
        if verbose and block_idx % 200 == 0 and block_idx > 0:
            print(f"    ...processed {block_idx}/{total_blocks} blocks, "
                  f"{scored_pairs} pairs scored so far")

        if len(block_entities) > max_block_size:
            skipped_blocks.append({"block": block, "size": len(block_entities)})
            # Large blocks are handled by exact canonical merging above. Scoring
            # every fuzzy pair here is too expensive and usually low precision.
            continue

        ordered = sorted(
            block_entities,
            key=lambda e: canonical_key(e.get("canonical") or e.get("text") or e.get("original_mention") or "", e.get("label"))
        )
        for i, left in enumerate(ordered):
            window = ordered[i + 1:i + 1 + max_neighbors_per_entity]
            for right in window:
                if has_hard_conflict(left, right):
                    continue
                features = pair_features(left, right)
                score = er_probability(features)
                merged = score >= threshold
                scored_pairs += 1
                if len(candidates) < max_candidate_audit and (merged or score >= 0.45):
                    candidates.append({
                        "left": left["mention_id"],
                        "right": right["mention_id"],
                        "left_text": left.get("original_mention") or left.get("canonical"),
                        "right_text": right.get("original_mention") or right.get("canonical"),
                        "label": left.get("label"),
                        "score": round(score, 4),
                        "features": {k: round(v, 4) for k, v in features.items()},
                        "merged": merged,
                    })
                if merged:
                    uf.union(left["mention_id"], right["mention_id"])

    if verbose:
        print(f"  ER mentions: {len(flat)}")
        print(f"  ER unique canonical forms: {len(representatives)}")
        print(f"  ER exact-key merges: {exact_merges}")
        print(f"  ER fuzzy pairs scored: {scored_pairs}")
        if skipped_blocks:
            biggest = sorted(skipped_blocks, key=lambda x: x["size"], reverse=True)[:5]
            print(f"  ER skipped {len(skipped_blocks)} oversized fuzzy blocks; biggest: {biggest}")

    grouped = defaultdict(list)
    for mention_id, entity in by_id.items():
        grouped[uf.find(mention_id)].append(entity)

    clusters = []
    mention_to_cluster = {}
    for members in grouped.values():
        label = Counter(m.get("label") for m in members).most_common(1)[0][0]
        aliases = sorted({normalize_mention(m.get("original_mention") or m.get("canonical")) for m in members})
        canonical = Counter(canonical_key(m.get("canonical") or aliases[0], label) for m in members).most_common(1)[0][0]
        cluster_id = stable_id("e", label, canonical)
        for member in members:
            mention_to_cluster[member["mention_id"]] = cluster_id
        clusters.append({
            "entity_id": cluster_id,
            "canonical": canonical,
            "label": label,
            "aliases": aliases,
            "mention_ids": [m["mention_id"] for m in members],
            "filenames": sorted({m.get("filename") for m in members}),
            "confidence": round(min(1.0, 0.55 + 0.1 * len(members)), 3),
        })

    return clusters, candidates, mention_to_cluster


def mention_quality(value):
    """
    Score 0.0-1.0: is this actually a plausible entity mention, or CRF noise
    (a clause fragment, boilerplate leakage, a stray initial, a redaction
    placeholder)? Zero means "don't treat this as a real entity at all" —
    used both to gate stage3 entity linking (so junk never gets merged into
    or used as a canonical anchor) and stage7 confidence scoring.
    """
    value = normalize_mention(value)
    if len(value) < 2:
        return 0.0
    if re.fullmatch(r"[\W_]+", value):
        return 0.0
    # real entities (party names, contract titles, dates) are short spans.
    # anything longer is almost always a sentence/clause fragment that got
    # mis-extracted as an entity, not an actual named thing.
    if len(value.split()) > 8:
        return 0.0
    # legal cross-reference / boilerplate leakage: "9 below", "as provided
    # herein", "pursuant to section 15" etc. — never real entity names.
    if re.search(r"\b(below|above|herein|hereof|hereto|hereunder|witnesseth|"
                 r"pursuant|notwithstanding|whereas|shall|hereby)\b", value, re.I):
        return 0.0
    # redaction placeholders, bare numbers, stray initials/section labels
    if re.fullmatch(r"[a-z]\s*\*+", value, re.I):
        return 0.0
    if re.fullmatch(r"\d+", value):
        return 0.0
    if re.fullmatch(r"[a-z]{1,2}", value, re.I):
        return 0.0
    # multiple commas = list/clause structure, not a name
    if value.count(",") >= 2:
        return 0.0
    if len(value) <= 3 and value.isalpha() and value.isupper():
        return 0.45
    if len(value) <= 3:
        return 0.35
    if re.search(r"\b(agreement|contract|inc|corp|llc|ltd|company|effective|date)\b", value, re.I):
        return 1.0
    return 0.80


def relation_type_ok(relation, subject_type, object_type):
    allowed = DOMAIN_RANGE.get(relation)
    return not allowed or (subject_type, object_type) in allowed