"""
stage12_rag_engine.py (v3)

A local, offline retrieval engine over the contract NER + knowledge-graph
pipeline (stages 1-11). No LLM / no API calls: "generation" here means
well-organized extractive retrieval, not generated prose.

v3 change from v2: the text channel is now HYBRID, not BM25-only. BM25
only matches literal keyword overlap ("owns" vs "acquired" score zero
shared terms even though they're the same fact). A sentence-embedding
model is added alongside it to catch paraphrases/semantic matches, and
the two rankings are fused (Reciprocal Rank Fusion) before the existing
graph-boost re-ranking runs -- which is completely unchanged. If the
embedding model isn't installed or can't load, this falls back to
exactly the v2 BM25-only behavior, with a printed note.

Requires (only for the new semantic channel -- optional):
    pip install sentence-transformers --break-system-packages
First run downloads the model (~90MB, needs internet once, then cached
locally by sentence-transformers/huggingface as usual).

Inputs (must exist alongside this script, or pass custom paths):
    combined_training_data.json  -> list of {"text": str, "entities": [...]}
    graph_completed.json         -> networkx node-link JSON
                                     {"nodes": [...], "links": [...], ...}
    relations.json                -> OPTIONAL. Stage 6 output (pre-validation),
                                     same node-link shape as graph_completed.json.
                                     Used ONLY to backfill entities that have
                                     ZERO facts in the validated graph -- never
                                     merged into the trusted kg_facts channel.
                                     If the file is missing, the engine behaves
                                     exactly as before (v1).

Retrieval channels, merged:
    1. Text channel (HYBRID) : BM25 (lexical) + sentence embeddings
       (semantic) over sentence text, fused via Reciprocal Rank Fusion
       -> top-K matching sentences.
    2. Graph channel     : entity mentions (in the query, or in the top
       sentences) -> 1-hop relations pulled from the completed (validated) KG.
    3. Unverified backfill: for entities recognized in your contracts that
       have NO facts in the validated graph, pull 1-hop relations from the
       stage-6 candidate pool instead -- returned in a separate field,
       clearly labeled unverified, never ranked alongside validated facts.

Usage:
    python3 rag_engine_v3.py                     # interactive REPL
    python3 rag_engine_v3.py "who is the distributor"   # one-shot query

Or import and call query() programmatically:
    from rag_engine_v3 import RagEngine
    engine = RagEngine()
    result = engine.query("governing law of the agreement")
"""

import json
import re
import pickle
import os
from collections import defaultdict, Counter, deque

from rank_bm25 import BM25Okapi

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SENTENCES_PATH = os.path.join(DATA_DIR, "combined_training_data.json")
GRAPH_PATH = os.path.join(DATA_DIR, "graph_completed.json")
CANDIDATE_TRIPLES_PATH = os.path.join(DATA_DIR, "relations.json")  # optional, stage 6 / pre-validation
CACHE_PATH = os.path.join(DATA_DIR, "rag_index_cache_v3.pkl")  # separate cache file from v2

MAX_UNVERIFIED_FACTS_PER_ENTITY = 5

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for this corpus size
RRF_K = 60  # standard Reciprocal Rank Fusion constant -- dampens the impact of any single rank

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
MAX_SUGGESTION_WORDS = 6  # real named entities are short; longer spans are usually leftover clause fragments

# --- Graph-reasoning tuning ---
# When a query contains one of these phrases, edges whose relation label
# contains one of the mapped substrings are surfaced first (still ranked by
# confidence within that group), instead of relying on confidence alone to
# happen to put the relevant relation type on top.
RELATION_HINTS = {
    "governing law": ["governed_by", "governing_law", "jurisdiction"],
    "jurisdiction": ["jurisdiction", "governed_by"],
    "effective date": ["effective_date", "commences", "start_date"],
    "notice": ["notice_address", "notice_to"],
    "distributor": ["distributor_of", "distributed_by", "distributes"],
    "parent company": ["parent_of", "subsidiary_of", "owns", "owned_by"],
    "subsidiary": ["subsidiary_of", "parent_of"],
    "terminat": ["terminates", "termination_of", "terminated_by"],  # covers terminate/termination
    "govern": ["governed_by", "governing_law"],
}

GRAPH_MAX_HOPS = 2          # cap for both re-ranking reachability and default path search
GRAPH_BOOST_WEIGHT = 0.35   # how much graph-adjacency can lift a sentence's BM25 score
BM25_CANDIDATE_MULTIPLIER = 4  # widen the BM25 shortlist before graph re-ranking narrows it back down
MAX_PATH_HOPS = 4
MAX_ENTITY_PAIRS_FOR_PATHS = 6  # cap combinatorial pairwise path search on multi-entity queries


def normalize_entity_text(text: str) -> str:
    """Strip trailing/leading punctuation noise (commas, quotes) left over
    from span-extraction boundary artifacts, e.g. 'Google Toolbar,' -> 'Google Toolbar'."""
    return text.strip().strip(",.;:'\"")


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
                 candidate_triples_path=CANDIDATE_TRIPLES_PATH, use_cache=True):
        self.sentences_path = sentences_path
        self.graph_path = graph_path
        self.candidate_triples_path = candidate_triples_path

        # Loaded fresh every run (not cached/pickled) since the model itself
        # isn't picklable in a portable way -- this is fast after the first
        # download, sentence-transformers caches the model files locally.
        self.embedder = self._load_embedder()

        if use_cache and os.path.exists(CACHE_PATH) and self._cache_is_fresh():
            self._load_from_cache()
            if self.embedder is not None and self.sentence_embeddings is None:
                # embedder is newly available since the cache was built --
                # rebuild once to backfill the semantic channel
                print("Embedding model now available but cache predates it -- rebuilding index once...")
                self._build()
                self._save_to_cache()
        else:
            self._build()
            if use_cache:
                self._save_to_cache()

    @staticmethod
    def _load_embedder():
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("[semantic channel disabled] `sentence-transformers` not installed -- "
                  "falling back to BM25-only text retrieval (same as v2). "
                  "To enable: pip install sentence-transformers --break-system-packages")
            return None
        try:
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"Loaded embedding model '{EMBEDDING_MODEL_NAME}' for semantic retrieval.")
            return model
        except Exception as e:
            print(f"[semantic channel disabled] Could not load embedding model ({e}). "
                  f"Falling back to BM25-only text retrieval (same as v2).")
            return None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _cache_is_fresh(self):
        cache_mtime = os.path.getmtime(CACHE_PATH)
        fresh = (cache_mtime > os.path.getmtime(self.sentences_path) and
                  cache_mtime > os.path.getmtime(self.graph_path))
        if fresh and os.path.exists(self.candidate_triples_path):
            fresh = cache_mtime > os.path.getmtime(self.candidate_triples_path)
        return fresh

    def _build(self):
        print("Building index (first run, or source files changed)...")

        # --- 1. Load sentence corpus ---
        with open(self.sentences_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)  # [{"text":..., "entities":[...]}]

        # --- 2. Build BM25 index over sentence text ---
        self.tokenized_corpus = [tokenize(r["text"]) for r in self.records]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # --- 2b. Semantic channel (v3): sentence embeddings, if available.
        # This is what lets retrieval catch paraphrases BM25 can't
        # ("owns" vs "acquired") -- see module docstring. ---
        self.sentence_embeddings = None
        if self.embedder is not None:
            print(f"Encoding {len(self.records)} sentences for semantic search "
                  f"(one-time cost, cached afterward)...")
            texts = [r["text"] for r in self.records]
            self.sentence_embeddings = self.embedder.encode(
                texts, normalize_embeddings=True, show_progress_bar=False,
                convert_to_numpy=True,
            )

        # --- 3. Entity -> sentence-index lookup (for direct entity search) ---
        # keyed by lowercased entity text, so lookups are case-insensitive
        self.entity_to_sentences = defaultdict(set)
        self.entity_frequency = Counter()   # for ranking autocomplete suggestions
        self.entity_display = {}            # lowercased -> best-cased display form
        self.entity_label = {}               # lowercased -> most common label
        label_votes = defaultdict(Counter)
        for idx, r in enumerate(self.records):
            for e in r["entities"]:
                if is_valid_entity(e["text"]):
                    norm = normalize_entity_text(e["text"])
                    if not norm:
                        continue
                    key = norm.lower()
                    self.entity_to_sentences[key].add(idx)
                    self.entity_frequency[key] += 1
                    label_votes[key][e["label"]] += 1
                    # prefer the mixed/title-cased form over all-caps for display
                    if key not in self.entity_display or (
                        norm != norm.upper()
                        and self.entity_display[key] == self.entity_display[key].upper()
                    ):
                        self.entity_display[key] = norm
        for key, votes in label_votes.items():
            self.entity_label[key] = votes.most_common(1)[0][0]

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

        # --- 5. OPTIONAL: stage-6 candidate triples (pre-validation). Kept in
        # a completely separate lookup from relations_by_node so it can
        # never accidentally be treated as validated. Only ever consulted
        # as a backfill for entities the validated graph has nothing on. ---
        self.candidate_relations_by_node = defaultdict(list)
        if os.path.exists(self.candidate_triples_path):
            with open(self.candidate_triples_path, "r", encoding="utf-8") as f:
                candidate_data = json.load(f)
            # relations.json may be a flat list of triples, OR the same
            # {"links": [...]} node-link shape as graph_completed.json --
            # accept both so stage-6 exports don't need reformatting.
            candidate_links = (
                candidate_data if isinstance(candidate_data, list)
                else candidate_data.get("links", [])
            )
            for link in candidate_links:
                # tolerate common key-naming variants from different
                # extraction stages (source/subject/head, target/object/tail)
                src = link.get("source") or link.get("subject") or link.get("head")
                tgt = link.get("target") or link.get("object") or link.get("tail")
                rel = link.get("relation") or link.get("predicate") or link.get("label")
                if not src or not tgt or not rel:
                    continue  # skip malformed entries rather than crash the whole build
                edge = {
                    "source": src,
                    "relation": rel,
                    "target": tgt,
                    "confidence": link.get("confidence"),
                    "validated": False,
                }
                self.candidate_relations_by_node[src].append(edge)
                self.candidate_relations_by_node[tgt].append(edge)
            print(f"Loaded {sum(len(v) for v in self.candidate_relations_by_node.values()) // 2} "
                  f"unverified candidate triples for backfill (stage 6, {self.candidate_triples_path})")

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
                "entity_frequency": self.entity_frequency,
                "entity_display": self.entity_display,
                "entity_label": self.entity_label,
                "node_types": self.node_types,
                "relations_by_node": self.relations_by_node,
                "candidate_relations_by_node": self.candidate_relations_by_node,
                "known_entities_sorted": self.known_entities_sorted,
                "sentence_embeddings": self.sentence_embeddings,
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

    def _kg_facts_for_entity(self, entity_name, max_facts=5, preferred_substrings=None):
        edges = self.relations_by_node.get(entity_name, [])
        if preferred_substrings:
            def sort_key(e):
                rel = (e["relation"] or "").lower()
                is_preferred = any(sub in rel for sub in preferred_substrings)
                return (is_preferred, e["confidence"] or 0)
            edges_sorted = sorted(edges, key=sort_key, reverse=True)
        else:
            edges_sorted = sorted(
                edges, key=lambda e: (e["confidence"] or 0), reverse=True
            )
        return edges_sorted[:max_facts]

    def _unverified_facts_for_entity(self, entity_name, max_facts=MAX_UNVERIFIED_FACTS_PER_ENTITY):
        """Stage-6 (pre-validation) 1-hop facts for an entity. Only ever
        called for entities that returned NOTHING from the validated graph --
        see query(). Never merged with _kg_facts_for_entity's output."""
        edges = self.candidate_relations_by_node.get(entity_name, [])
        edges_sorted = sorted(edges, key=lambda e: (e["confidence"] or 0), reverse=True)
        return edges_sorted[:max_facts]

    # ------------------------------------------------------------------
    # Graph traversal -- this is what turns "1-hop fact lookup" into
    # actual graph-based reasoning: multi-hop paths between named
    # entities, and bounded reachability for graph-informed re-ranking.
    # ------------------------------------------------------------------
    def _neighbors(self, node):
        """Yield (neighbor_id, edge) for a node, walking edges in
        whichever direction it wasn't the origin of -- relations_by_node
        already holds each edge under both its source and target."""
        for edge in self.relations_by_node.get(node, []):
            nb = edge["target"] if edge["source"] == node else edge["source"]
            yield nb, edge

    def _bfs_reachable_within_hops(self, start_nodes, max_hops=GRAPH_MAX_HOPS):
        """Bounded BFS from a set of anchor nodes. Returns {node: hop_distance}.
        Used to score how graph-close a candidate sentence's entities are
        to the entities actually named in the query -- the piece that was
        previously missing from sentence ranking entirely."""
        visited = {n: 0 for n in start_nodes}
        frontier = set(start_nodes)
        hop = 0
        while hop < max_hops and frontier:
            next_frontier = set()
            for node in frontier:
                for nb, _ in self._neighbors(node):
                    if nb not in visited:
                        visited[nb] = hop + 1
                        next_frontier.add(nb)
            frontier = next_frontier
            hop += 1
        return visited

    def _shortest_path(self, source, target, max_hops=MAX_PATH_HOPS):
        """BFS shortest path between two named entities, returned as an
        ordered list of edges (the actual chain of relations connecting
        them) -- or None if unreachable within max_hops. This is the
        core of multi-hop reasoning: answering "how is X connected to Y"
        instead of only "what's 1 hop from X" and "what's 1 hop from Y"
        as two unrelated lists."""
        if source == target:
            return []
        came_from = {source: None}  # node -> (prev_node, edge)
        hops = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            if hops[node] >= max_hops:
                continue
            for nb, edge in self._neighbors(node):
                if nb in came_from:
                    continue
                came_from[nb] = (node, edge)
                hops[nb] = hops[node] + 1
                if nb == target:
                    path = []
                    cur = nb
                    while came_from[cur] is not None:
                        prev, e = came_from[cur]
                        path.append(e)
                        cur = prev
                    path.reverse()
                    return path
                queue.append(nb)
        return None

    def find_reasoning_paths(self, entities, max_hops=MAX_PATH_HOPS,
                              max_pairs=MAX_ENTITY_PAIRS_FOR_PATHS):
        """Pairwise shortest-path search across every pair of recognized
        entities that made it into the graph. This is what lets a query
        naming two entities get an answer that's the actual reasoning
        chain between them, rather than two disconnected fact lists."""
        in_graph = [e for e in entities if e in self.node_types]
        pairs = []
        for i in range(len(in_graph)):
            for j in range(i + 1, len(in_graph)):
                pairs.append((in_graph[i], in_graph[j]))
        pairs = pairs[:max_pairs]

        paths = []
        for src, tgt in pairs:
            edge_path = self._shortest_path(src, tgt, max_hops=max_hops)
            if edge_path:  # note: [] would mean src==tgt, already excluded above
                paths.append({
                    "source": src,
                    "target": tgt,
                    "hops": len(edge_path),
                    "path": edge_path,
                })
        # shortest, most confident chains first
        paths.sort(key=lambda p: (p["hops"], -sum(
            (e["confidence"] or 0) for e in p["path"]) / max(len(p["path"]), 1)))
        return paths

    @staticmethod
    def _detect_relation_hints(query_text):
        """Cheap keyword match against RELATION_HINTS -- lets a query like
        'governing law of the agreement' prioritize governed_by/jurisdiction
        edges over whatever happens to have the highest raw confidence."""
        q = query_text.lower()
        hints = set()
        for phrase, relation_substrings in RELATION_HINTS.items():
            if phrase in q:
                hints.update(relation_substrings)
        return hints

    def suggest(self, prefix, limit=8):
        """Autocomplete: known entity names matching `prefix`, ranked by
        how often they appear in the corpus (like a real search engine
        ranks suggestions by popularity). Prefix match first, then
        falls back to substring match if prefix matches are scarce."""
        prefix = prefix.strip().lower()
        if not prefix:
            return []

        starts_with = [
            key for key in self.entity_frequency
            if key.startswith(prefix) and len(key.split()) <= MAX_SUGGESTION_WORDS
        ]
        results = sorted(starts_with, key=lambda k: -self.entity_frequency[k])

        if len(results) < limit:
            contains = [
                key for key in self.entity_frequency
                if prefix in key and key not in starts_with
                and len(key.split()) <= MAX_SUGGESTION_WORDS
            ]
            contains_sorted = sorted(contains, key=lambda k: -self.entity_frequency[k])
            results += contains_sorted

        out = []
        for key in results[:limit]:
            out.append({
                "text": self.entity_display.get(key, key),
                "label": self.entity_label.get(key, "UNKNOWN"),
                "frequency": self.entity_frequency[key],
                "in_graph": key in {k.lower() for k in self.node_types},
            })
        return out

    def example_questions(self, per_type=1):
        """A rotating set of example questions built from real, frequent
        entities of each type -- shown on the dashboard before the user
        has typed anything, so they know what's actually answerable."""
        by_label = defaultdict(list)
        for key, freq in self.entity_frequency.most_common(400):
            if len(key.split()) > MAX_SUGGESTION_WORDS:
                continue
            label = self.entity_label.get(key, "UNKNOWN")
            by_label[label].append(self.entity_display.get(key, key))

        templates = {
            "PARTY": "Who is {}",
            "CONTRACT": "What does the {} cover",
            "JURISDICTION": "Which contracts are governed by {}",
            "DATE": "What happens on {}",
            "EFFECTIVE_DATE": "What is effective on {}",
            "NOTICE": "Where should notice be sent",
        }
        examples = []
        for label, template in templates.items():
            names = by_label.get(label, [])
            if names:
                examples.append(template.format(names[0]))
        return examples

    def query(self, query_text, top_k_sentences=5, max_kg_facts_per_entity=5,
              use_graph_reranking=True):
        # --- Graph channel first: entities named directly in the query ---
        # (moved ahead of sentence ranking because re-ranking needs these)
        query_entities = self._match_entities_in_text(query_text)
        query_entities_in_graph = [e for e in query_entities if e in self.node_types]

        # bounded reachability set from the query's own entities -- this is
        # what lets sentence ranking care about graph adjacency at all
        reachable = {}
        if use_graph_reranking and query_entities_in_graph:
            reachable = self._bfs_reachable_within_hops(query_entities_in_graph)

        # --- Text channel: HYBRID BM25 (lexical) + embeddings (semantic) ---
        # strip stopwords from the QUERY only (corpus stays as-is) so
        # common words like "is"/"for"/"what" can't rack up spurious
        # matches against sentences that share no real topic with the query
        query_tokens = [t for t in tokenize(query_text) if t not in STOPWORD_DENYLIST]
        if not query_tokens:
            query_tokens = tokenize(query_text)  # fallback if query was all stopwords
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranked_idx = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )

        semantic_scores = None
        semantic_ranked_idx = []
        use_semantic = self.embedder is not None and self.sentence_embeddings is not None
        if use_semantic:
            query_vec = self.embedder.encode(
                [query_text], normalize_embeddings=True, convert_to_numpy=True,
            )[0]
            # embeddings are pre-normalized, so dot product == cosine similarity
            semantic_scores = self.sentence_embeddings @ query_vec
            semantic_ranked_idx = sorted(
                range(len(semantic_scores)), key=lambda i: semantic_scores[i], reverse=True
            )

        if use_semantic:
            # Reciprocal Rank Fusion: combines two rankings without needing
            # their raw scores to be on comparable scales (BM25 scores and
            # cosine similarities aren't). A sentence ranked well by EITHER
            # channel gets lifted; ranked well by BOTH gets lifted more.
            rrf_scores = defaultdict(float)
            for rank, idx in enumerate(bm25_ranked_idx):
                rrf_scores[idx] += 1.0 / (RRF_K + rank)
            for rank, idx in enumerate(semantic_ranked_idx):
                rrf_scores[idx] += 1.0 / (RRF_K + rank)
            ranked_idx = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        else:
            ranked_idx = bm25_ranked_idx
        scores = bm25_scores  # kept for the "scores[i] > 0" liveness check below

        # widen the BM25 shortlist before re-ranking, so a sentence that's
        # graph-close but slightly behind on raw lexical score still has a
        # chance to surface -- otherwise re-ranking could only ever
        # re-order the same top-K BM25 already picked, which isn't really
        # graph-informed retrieval, just graph-informed sorting.
        candidate_pool = min(top_k_sentences * BM25_CANDIDATE_MULTIPLIER, len(ranked_idx))
        if use_semantic:
            # don't filter by "bm25 score > 0" here -- a sentence can be a
            # strong SEMANTIC match with zero lexical overlap (e.g. "owns"
            # vs "acquired"), which is exactly the case hybrid retrieval
            # exists to catch. RRF fusion already did the real ranking.
            candidate_idxs = list(ranked_idx[:candidate_pool])
        else:
            candidate_idxs = [i for i in ranked_idx[:candidate_pool] if scores[i] > 0]

        scored_candidates = []
        for idx in candidate_idxs:
            rec = self.records[idx]
            clean_entities = [e["text"] for e in rec["entities"] if is_valid_entity(e["text"])]
            bm25_score = float(bm25_scores[idx])
            semantic_score = float(semantic_scores[idx]) if use_semantic else None
            graph_boost = 0.0
            if reachable:
                for e in clean_entities:
                    hop = reachable.get(e)
                    if hop is not None:
                        # closer hop = bigger boost; hop 0 means the sentence
                        # names a query entity directly
                        graph_boost += 1.0 / (1 + hop)
            # base retrieval score: fused rank position if hybrid is active,
            # otherwise plain BM25 score, same as v2
            base_score = rrf_scores[idx] if use_semantic else bm25_score
            final_score = base_score * (1 + GRAPH_BOOST_WEIGHT * graph_boost)
            scored_candidates.append({
                "idx": idx,
                "text": rec["text"],
                "bm25_score": round(bm25_score, 3),
                "semantic_score": round(semantic_score, 3) if semantic_score is not None else None,
                "graph_boost": round(graph_boost, 3),
                "score": round(final_score, 5),
                "entities": clean_entities,
            })

        scored_candidates.sort(key=lambda c: c["score"], reverse=True)
        matched_sentences = scored_candidates[:top_k_sentences]
        for c in matched_sentences:
            del c["idx"]

        # KG facts are shown ONLY for entities explicitly named in the
        # query -- NOT entities that merely happen to appear in a matched
        # sentence. Otherwise a clause that mentions "Sponsorship Agreement"
        # in passing (while matching on an unrelated word like "Tender")
        # would hijack the whole KG panel with facts about "Sponsorship
        # Agreement" the person never asked about. Text retrieval and the
        # KG panel are now decoupled on purpose.
        entities_seen = set(query_entities)

        # --- Relation-type hints: "governing law", "notice", etc. bias
        # which edges surface first, instead of confidence alone deciding ---
        relation_hints = self._detect_relation_hints(query_text)

        # --- Expand every entity NAMED IN THE QUERY into 1-hop KG facts ---
        kg_facts = {}
        entities_missing_from_kg = []
        unverified_kg_facts = {}
        for ent in entities_seen:
            if ent in self.node_types:
                facts = self._kg_facts_for_entity(
                    ent, max_kg_facts_per_entity,
                    preferred_substrings=relation_hints or None,
                )
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
                # backfill from stage-6 candidates ONLY here -- this entity
                # got zero validated facts, so unverified is strictly better
                # than nothing, as long as it's labeled as such
                unverified = self._unverified_facts_for_entity(ent)
                if unverified:
                    unverified_kg_facts[ent] = {"facts": unverified}

        # --- Multi-hop reasoning: actual paths between query entities,
        # not just independent 1-hop fans from each one ---
        reasoning_paths = []
        if len(query_entities_in_graph) >= 2:
            reasoning_paths = self.find_reasoning_paths(query_entities_in_graph)

        return {
            "query": query_text,
            "matched_sentences": matched_sentences,
            "query_entities_recognized": query_entities,
            "kg_facts": kg_facts,
            "query_entities_missing_from_kg": entities_missing_from_kg,
            "unverified_kg_facts": unverified_kg_facts,
            "reasoning_paths": reasoning_paths,
            "relation_hints_applied": sorted(relation_hints) if relation_hints else [],
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

    if result.get("relation_hints_applied"):
        lines.append(
            "Relation-type hints applied to fact ranking: "
            + ", ".join(result["relation_hints_applied"])
        )
        lines.append("")

    lines.append(f"-- Top matching sentences ({len(result['matched_sentences'])}) --")
    for i, s in enumerate(result["matched_sentences"], 1):
        boost_note = f", graph_boost {s['graph_boost']}" if s.get("graph_boost") else ""
        semantic_note = f", semantic {s['semantic_score']}" if s.get("semantic_score") is not None else ""
        lines.append(f"{i}. (score {s['score']} = bm25 {s['bm25_score']}{semantic_note}{boost_note}) {s['text']}")
        if s["entities"]:
            lines.append(f"   entities: {', '.join(s['entities'])}")
    lines.append("")

    if result.get("reasoning_paths"):
        lines.append(f"-- Multi-hop reasoning paths ({len(result['reasoning_paths'])}) --")
        for p in result["reasoning_paths"]:
            chain = " -> ".join(
                f"{e['source']} --{e['relation']}--> {e['target']}"
                for e in p["path"]
            )
            lines.append(f"[{p['source']} .. {p['target']}] ({p['hops']} hop(s)): {chain}")
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
    if result.get("unverified_kg_facts"):
        lines.append("")
        lines.append(
            "-- UNVERIFIED facts (stage 6, pre-validation -- shown only "
            f"because the entities above have nothing in the validated graph) --"
        )
        for ent, info in result["unverified_kg_facts"].items():
            lines.append(f"[{ent}] (unverified)")
            for fact in info["facts"]:
                lines.append(
                    f"   {fact['source']} --{fact['relation']}--> {fact['target']}"
                    f"  (confidence {fact['confidence']}, UNVALIDATED)"
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