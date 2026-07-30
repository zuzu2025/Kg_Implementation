"""
stage12_rag_engine.py

A local, offline retrieval engine over the contract NER + knowledge-graph
pipeline (stages 1-11). No LLM / no API calls: "generation" here means
well-organized extractive retrieval, not generated prose.

Inputs (must exist alongside this script, or pass custom paths):
    combined_training_data.json  -> list of {"text": str, "entities": [...]}
    graph_completed.json         -> networkx node-link JSON
                                     {"nodes": [...], "links": [...], ...}

Two retrieval channels, merged:
    1. Text channel : BM25 over sentence text -> top-K matching sentences.
    2. Graph channel: entity mentions (in the query, or in the top
       sentences) -> 1-hop relations pulled from the completed KG.

Usage:
    python3 rag_engine.py                     # interactive REPL
    python3 rag_engine.py "who is the distributor"   # one-shot query

Or import and call query() programmatically:
    from rag_engine import RagEngine
    engine = RagEngine()
    result = engine.query("governing law of the agreement")
"""

import json
import re
import pickle
import os
from collections import defaultdict

from rank_bm25 import BM25Okapi

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SENTENCES_PATH = os.path.join(DATA_DIR, "combined_training_data.json")
GRAPH_PATH = os.path.join(DATA_DIR, "graph_completed.json")
CACHE_PATH = os.path.join(DATA_DIR, "rag_index_cache.pkl")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Known data-quality issues from upstream stages (documented, not silently
# patched): some "entities" are actually mislabeled fragments (e.g. "The",
# "YOU") and some are entire clauses mistakenly carried through as a single
# entity span (avg ~330 chars) due to the sentence-splitter's boundary bug.
# Both are filtered out of the RETRIEVAL INDEX only -- the source JSON files
# are never modified.
STOPWORD_DENYLIST = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "you",
    "your", "we", "our", "they", "their", "he", "she", "his", "her", "is",
    "are", "was", "were", "be", "been", "being", "and", "or", "but", "of",
    "in", "on", "at", "to", "for", "with", "as", "by", "from", "which",
    "who", "whom", "such", "any", "all", "here", "there",
}
MAX_ENTITY_LEN = 80  # spans longer than this are corrupted clause-spans, not real entities
MIN_ENTITY_LEN = 3

# Generic role/document placeholders (e.g. a contract defining "the Company"
# or "the Agreement" as a local alias). Stage 3's entity linking canonicalized
# these to ONE shared literal string across ALL 510 contracts, instead of
# resolving each to that specific document's actual party/contract name --
# so e.g. the single node "Company" ends up falsely "connected" to 89+
# unrelated real companies, and "Agreement" to 653+ unrelated contracts.
# These are excluded as EXACT (not substring) matches only, so genuine
# names like "Distributor Agreement" or "XYZ Company Inc" are unaffected.
GENERIC_ROLE_TERMS = {
    "company", "distributor", "vendor", "client", "party", "parties",
    "purchaser", "seller", "licensor", "licensee", "contractor", "agreement",
    "supplier", "customer", "buyer", "manufacturer", "the company",
    "the distributor", "the vendor", "the client", "the agreement",
}


def is_valid_entity(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < MIN_ENTITY_LEN or len(stripped) > MAX_ENTITY_LEN:
        return False
    if stripped.lower() in STOPWORD_DENYLIST:
        return False
    if stripped.lower() in GENERIC_ROLE_TERMS:
        return False
    return True


def tokenize(text: str):
    """Lowercase, alphanumeric-only tokenizer. Simple on purpose -
    consistent with the classical (non-neural) style of the rest of
    this pipeline."""
    return _TOKEN_RE.findall(text.lower())


class RagEngine:
    def __init__(self, sentences_path=SENTENCES_PATH, graph_path=GRAPH_PATH,
                 use_cache=True):
        self.sentences_path = sentences_path
        self.graph_path = graph_path

        if use_cache and os.path.exists(CACHE_PATH) and self._cache_is_fresh():
            self._load_from_cache()
        else:
            self._build()
            if use_cache:
                self._save_to_cache()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _cache_is_fresh(self):
        cache_mtime = os.path.getmtime(CACHE_PATH)
        return (cache_mtime > os.path.getmtime(self.sentences_path) and
                cache_mtime > os.path.getmtime(self.graph_path))

    def _build(self):
        print("Building index (first run, or source files changed)...")

        # --- 1. Load sentence corpus ---
        with open(self.sentences_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)  # [{"text":..., "entities":[...]}]

        # --- 2. Build BM25 index over sentence text ---
        self.tokenized_corpus = [tokenize(r["text"]) for r in self.records]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # --- 3. Entity -> sentence-index lookup (for direct entity search) ---
        # keyed by lowercased entity text, so lookups are case-insensitive
        self.entity_to_sentences = defaultdict(set)
        for idx, r in enumerate(self.records):
            for e in r["entities"]:
                if is_valid_entity(e["text"]):
                    self.entity_to_sentences[e["text"].lower()].add(idx)

        # --- 4. Load graph, build fast node -> relations lookup ---
        with open(self.graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        self.node_types = {n["id"]: n.get("type", "UNKNOWN")
                            for n in graph_data["nodes"]
                            if is_valid_entity(n["id"])}

        # relations indexed both ways: as source and as target, since
        # the graph is directed and a query might name either side
        self.relations_by_node = defaultdict(list)
        for link in graph_data["links"]:
            src, tgt = link["source"], link["target"]
            edge = {
                "source": src,
                "relation": link["relation"],
                "target": tgt,
                "confidence": link.get("confidence"),
                "inferred": link.get("inferred", False),
            }
            self.relations_by_node[src].append(edge)
            self.relations_by_node[tgt].append(edge)

        # known entity strings, longest-first, for substring matching
        # against raw query text (so "Electric City Corp" inside a
        # longer question still gets recognized). This pool combines
        # graph nodes AND text-corpus entities -- an entity can be real
        # (appear in your contracts) without ever having made it into
        # the completed graph (see docstring note on KG coverage gaps).
        all_known_entity_names = set(self.node_types.keys()) | set(
            e["text"] for r in self.records for e in r["entities"]
            if is_valid_entity(e["text"])
        )
        self.known_entities_sorted = sorted(all_known_entity_names, key=len, reverse=True)

    def _save_to_cache(self):
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({
                "records": self.records,
                "tokenized_corpus": self.tokenized_corpus,
                "bm25": self.bm25,
                "entity_to_sentences": self.entity_to_sentences,
                "node_types": self.node_types,
                "relations_by_node": self.relations_by_node,
                "known_entities_sorted": self.known_entities_sorted,
            }, f)

    def _load_from_cache(self):
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        self.__dict__.update(cached)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _match_entities_in_text(self, text):
        """Find known KG entity names that appear as substrings of text
        (case-insensitive). Longest-name-first avoids e.g. matching
        'Electric City' inside a longer 'Electric City Corp.' mention
        twice."""
        text_lower = text.lower()
        found = []
        covered = [False] * len(text_lower)
        for name in self.known_entities_sorted:
            name_lower = name.lower()
            if not name_lower or name_lower not in text_lower:
                continue
            start = text_lower.find(name_lower)
            if start != -1 and not any(covered[start:start + len(name_lower)]):
                found.append(name)
                for i in range(start, start + len(name_lower)):
                    if i < len(covered):
                        covered[i] = True
        return found

    def _kg_facts_for_entity(self, entity_name, max_facts=5):
        edges = self.relations_by_node.get(entity_name, [])
        edges_sorted = sorted(
            edges, key=lambda e: (e["confidence"] or 0), reverse=True
        )
        return edges_sorted[:max_facts]

    def query(self, query_text, top_k_sentences=5, max_kg_facts_per_entity=5):
        # --- Text channel: BM25 over sentences ---
        # strip stopwords from the QUERY only (corpus stays as-is) so
        # common words like "is"/"for"/"what" can't rack up spurious
        # matches against sentences that share no real topic with the query
        query_tokens = [t for t in tokenize(query_text) if t not in STOPWORD_DENYLIST]
        if not query_tokens:
            query_tokens = tokenize(query_text)  # fallback if query was all stopwords
        scores = self.bm25.get_scores(query_tokens)
        ranked_idx = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        top_sentence_idxs = [i for i in ranked_idx[:top_k_sentences] if scores[i] > 0]

        matched_sentences = []
        entities_seen = set()
        for idx in top_sentence_idxs:
            rec = self.records[idx]
            clean_entities = [e["text"] for e in rec["entities"] if is_valid_entity(e["text"])]
            matched_sentences.append({
                "text": rec["text"],
                "score": round(float(scores[idx]), 3),
                "entities": clean_entities,
            })
            for e in clean_entities:
                entities_seen.add(e)

        # --- Graph channel: entities named directly in the query itself ---
        query_entities = self._match_entities_in_text(query_text)
        for ent in query_entities:
            entities_seen.add(ent)

        # --- Expand every entity we now know about into 1-hop KG facts ---
        kg_facts = {}
        entities_missing_from_kg = []
        for ent in entities_seen:
            if ent in self.node_types:
                facts = self._kg_facts_for_entity(ent, max_kg_facts_per_entity)
                if facts:
                    kg_facts[ent] = {
                        "type": self.node_types[ent],
                        "facts": facts,
                    }
            elif ent in query_entities:
                # explicitly named in the query, but never made it into
                # the completed graph -- worth telling the user this,
                # rather than silently showing facts about something else
                entities_missing_from_kg.append(ent)

        return {
            "query": query_text,
            "matched_sentences": matched_sentences,
            "query_entities_recognized": query_entities,
            "kg_facts": kg_facts,
            "query_entities_missing_from_kg": entities_missing_from_kg,
        }


def format_result(result):
    lines = []
    lines.append(f"Query: {result['query']}")
    lines.append("")
    if result["query_entities_recognized"]:
        lines.append(
            "Entities recognized in query: "
            + ", ".join(result["query_entities_recognized"])
        )
        lines.append("")

    lines.append(f"-- Top matching sentences ({len(result['matched_sentences'])}) --")
    for i, s in enumerate(result["matched_sentences"], 1):
        lines.append(f"{i}. (score {s['score']}) {s['text']}")
        if s["entities"]:
            lines.append(f"   entities: {', '.join(s['entities'])}")
    lines.append("")

    lines.append(f"-- Knowledge-graph facts ({len(result['kg_facts'])} entities) --")
    for ent, info in result["kg_facts"].items():
        lines.append(f"[{ent}] ({info['type']})")
        for fact in info["facts"]:
            arrow = "inferred" if fact["inferred"] else "extracted"
            lines.append(
                f"   {fact['source']} --{fact['relation']}--> {fact['target']}"
                f"  (confidence {fact['confidence']}, {arrow})"
            )
    if result.get("query_entities_missing_from_kg"):
        lines.append("")
        lines.append(
            "Note: recognized in your contracts but has NO facts in the "
            "completed knowledge graph (likely dropped during validation "
            "or link prediction): "
            + ", ".join(result["query_entities_missing_from_kg"])
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    engine = RagEngine()

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(format_result(engine.query(q)))
    else:
        print("RAG engine ready. Type a question (or 'quit' to exit).")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("quit", "exit"):
                break
            print()
            print(format_result(engine.query(q)))