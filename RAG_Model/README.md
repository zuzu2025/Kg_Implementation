# Stage 12 — Local RAG Retrieval Engine

A fully offline retrieval engine over your contract NER + knowledge-graph
pipeline. **No API calls, no LLM.** "Generation" here means organized,
ranked extraction — the retrieved sentences and graph facts *are* the
answer, nothing is paraphrased or invented.

## Setup

```bash
pip install rank_bm25 --break-system-packages   # one-time, only new dependency
```

Everything else (`scikit-learn`, `networkx`) you already have installed.

Keep these three files in the same folder:
- `rag_engine.py`
- `combined_training_data.json`
- `graph_completed.json`

## Running it

**Interactive mode:**
```bash
python3 rag_engine.py
```

**One-shot query:**
```bash
python3 rag_engine.py "who is the distributor"
```

**Programmatic use:**
```python
from rag_engine import RagEngine
engine = RagEngine()
result = engine.query("governing law of the agreement")
print(result["matched_sentences"])
print(result["kg_facts"])
```

First run builds an index and caches it to `rag_index_cache.pkl`
(rebuilds automatically if the source JSON files change).

## How it works

1. **Text channel** — BM25 (classical lexical search, same family of
   technique as stage 3's entity linking) ranks all 15,467 sentences
   against your query, returns the top-K.
2. **Graph channel** — any known KG entity name that appears in your
   query, or in the top-matched sentences, gets expanded to its 1-hop
   relations from `graph_completed.json` (both extracted and stage-8
   inferred edges — inferred ones are labeled `inferred: true` so you
   can tell them apart).
3. Both channels are merged and returned together: matched sentences +
   structured KG facts, ranked by score/confidence, no generation step.

## Dashboard (search-engine-style UI)

A local web dashboard with live autocomplete, on top of the same engine.

**Setup (in addition to the CLI setup above):**
```bash
pip install flask --break-system-packages   # one-time, if not already installed
```

**File layout required:**
```
your_folder/
  rag_engine.py
  rag_server.py
  templates/
    index.html
  combined_training_data.json
  graph_completed.json
```

**Run:**
```bash
python3 rag_server.py
```
Then open **http://127.0.0.1:5050** in your browser.

Type a few letters and it suggests real entity names from your contracts
as you type (ranked by how often each appears — like search-engine
autocomplete), pulled from `RagEngine.suggest()`. Submitting a query shows
matched clauses on the left and knowledge-graph facts on the right, with
confidence "seals" and a dashed seal specifically marking stage 8's
*inferred* (not directly extracted) links.

## Known data-quality issues this engine works around

These were found while building this engine, and are **filtered at
retrieval-index time only** — your source JSON files are untouched:

- **Garbage entity nodes** (`"The"`, `"YOU"`, `"your"`, and ~160 other
  short fragments) exist in `graph_completed.json`, mislabeled as real
  entities (e.g. `"The"` tagged `PARTY`). Filtered via a stopword/length
  denylist so they don't pollute every query.
- **7,205 of 15,467 sentences (46.6%) have an entity span that equals
  the entire sentence** (avg ~330 chars, max 2,586) — a downstream
  effect of the sentence-splitter's boundary-protection bug from
  earlier in the pipeline. Filtered by capping entity length at 80
  characters in the index.
- **Generic role/document placeholders act as false collision hubs.**
  Contracts define local aliases like "the Company" or "the Agreement,"
  and stage 3's entity linking canonicalized these to ONE shared literal
  node across all 510 contracts instead of resolving each to that
  document's actual party/contract name. Result: the single node
  `"Company"` is falsely "connected" to 89+ unrelated real companies,
  and `"Agreement"` to 653+ unrelated contracts. Any query touching one
  of these terms pulled in a random grab-bag of unrelated facts. Filtered
  by excluding an explicit list of generic role terms (company,
  distributor, vendor, party, agreement, etc.) as exact-match-only, so
  real names like "Distributor Agreement" or "XYZ Company Inc" are
  unaffected.

Both are workarounds, not root-cause fixes — the real fix would be
revisiting the splitter's boundary logic and stage 3's entity linking
(to resolve local role aliases to actual party names per-document)
and re-running stages 1.5 onward, which is a separate, bigger job
outside this engine.
