"""
GraphRAG Eval Harness: Stage 6 Ontology Design vs Stage 7 Validated Triples
=============================================================================

Compares two RAG pipelines that differ only in their data source:
  Pipeline A -> Stage 6 ontology design output (all candidate facts / ontology relations)
  Pipeline B -> Stage 7 validated triples only

Everything else (retrieval method, verbalization, prompt, LLM, temperature)
is held identical so any difference in output is attributable to the
validation stage, not the pipeline.

USAGE
-----
1. Put your Stage 6 ontology facts in `ontology_design.json`.
2. Put your Stage 7 validated triples in `validated_triples.json`.
3. You can also keep a combined export in `triples.json` as a fallback.
4. Put your eval questions in `eval_set.json`.
5. Set your Anthropic API key:  export ANTHROPIC_API_KEY="sk-ant-..."
6. Run:  python test.py

OUTPUT
------
- Prints a side-by-side comparison table to stdout
- Writes detailed results to `eval_results.json`
- Writes a summary CSV to `eval_summary.csv`

This script has NO dependency on your actual KG database. It works on
plain JSON triples so you can point it at any export from your pipeline
(Neo4j, RDF store, or a flat file dump of stage 6 / stage 7 output).
"""

import json
import os
import csv
from dataclasses import dataclass
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------

TOP_K = 5                     # how many triples to retrieve per query
LLM_MODEL = "claude-sonnet-4-5"  # change to whatever model you have API access to
ONTOLOGY_FILE = "ontology_design.json"
VALIDATED_FILE = "validated_triples.json"
TRIPLES_FILE = "triples.json"
EVAL_SET_FILE = "eval_set.json"
RESULTS_FILE = "eval_results.json"
SUMMARY_CSV = "eval_summary.csv"


# ---------------------------------------------------------------------------
# 1. SAMPLE DATA (used only if the stage files / eval set don't exist yet)
# ---------------------------------------------------------------------------

SAMPLE_ONTOLOGY_DESIGN = [
    {"subject": "Company A", "predicate": "acquired", "object": "Company B",
     "validated": False, "confidence": 0.42, "source": "stage6_candidate.txt"},
    {"subject": "Company B", "predicate": "headquartered_in", "object": "Berlin",
     "validated": False, "confidence": 0.51, "source": "stage6_candidate.txt"},
    {"subject": "Company B", "predicate": "founded_by", "object": "Person C",
     "validated": False, "confidence": 0.39, "source": "stage6_candidate.txt"},
    {"subject": "Company A", "predicate": "ceo", "object": "Person D",
     "validated": False, "confidence": 0.44, "source": "stage6_candidate.txt"},
    {"subject": "Person D", "predicate": "previously_worked_at", "object": "Company E",
     "validated": False, "confidence": 0.55, "source": "stage6_candidate.txt"},
    {"subject": "Company E", "predicate": "competitor_of", "object": "Company A",
     "validated": False, "confidence": 0.38, "source": "stage6_candidate.txt"},
]

SAMPLE_VALIDATED_TRIPLES = [
    {"subject": "Company A", "predicate": "acquired", "object": "Company B",
     "validated": True, "confidence": 0.95, "source": "stage7_validation.txt"},
    {"subject": "Company B", "predicate": "headquartered_in", "object": "Berlin",
     "validated": True, "confidence": 0.91, "source": "stage7_validation.txt"},
    {"subject": "Company A", "predicate": "ceo", "object": "Person D",
     "validated": True, "confidence": 0.88, "source": "stage7_validation.txt"},
]

SAMPLE_TRIPLES = SAMPLE_ONTOLOGY_DESIGN + SAMPLE_VALIDATED_TRIPLES

SAMPLE_EVAL_SET = [
    {
        "question": "Who acquired Company B?",
        "expected_answer": "Company A",
        "query_type": "single_hop"
    },
    {
        "question": "Where is the company that Company A acquired headquartered?",
        "expected_answer": "Berlin",
        "query_type": "multi_hop"
    },
    {
        "question": "Where did the CEO of Company A previously work?",
        "expected_answer": "Company E (unverified)",
        "query_type": "multi_hop_unvalidated_dependency"
    },
]


def _bootstrap_sample_files():
    if not os.path.exists(ONTOLOGY_FILE):
        with open(ONTOLOGY_FILE, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_ONTOLOGY_DESIGN, f, indent=2)
        print(f"[bootstrap] Wrote sample stage 6 ontology data to {ONTOLOGY_FILE}")
    if not os.path.exists(VALIDATED_FILE):
        with open(VALIDATED_FILE, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_VALIDATED_TRIPLES, f, indent=2)
        print(f"[bootstrap] Wrote sample stage 7 validated triples to {VALIDATED_FILE}")
    if not os.path.exists(TRIPLES_FILE):
        with open(TRIPLES_FILE, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_TRIPLES, f, indent=2)
        print(f"[bootstrap] Wrote combined sample data to {TRIPLES_FILE} (fallback)")
    if not os.path.exists(EVAL_SET_FILE):
        with open(EVAL_SET_FILE, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_EVAL_SET, f, indent=2)
        print(f"[bootstrap] Wrote sample eval set to {EVAL_SET_FILE} (replace with your real questions)")


# ---------------------------------------------------------------------------
# 2. DATA MODEL
# ---------------------------------------------------------------------------

@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    validated: bool
    confidence: float = 0.5
    source: str = ""

    def verbalize(self) -> str:
        """Turn a triple into a natural-language sentence for retrieval + prompting.
        This is the step that was likely missing/weak in your original pipeline —
        BM25 and the LLM both work far better on this than on raw JSON."""
        pred_readable = self.predicate.replace("_", " ")
        tag = "" if self.validated else " [UNVERIFIED]"
        return f"{self.subject} {pred_readable} {self.object}.{tag}"


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_triples(path: str) -> List[Triple]:
    raw = load_json(path)
    if isinstance(raw, dict):
        for key in ("triples", "ontology_design", "validated_triples",
                    "stage6_ontology_design", "stage7_validated_triples"):
            if key in raw:
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of triples in {path}, got {type(raw).__name__}")
    return [Triple(**t) for t in raw]


def load_stage6_triples(path: str, fallback_path: str) -> List[Triple]:
    if os.path.exists(path):
        return load_triples(path)
    if os.path.exists(fallback_path):
        return load_triples(fallback_path)
    return []


def load_stage7_triples(path: str, fallback_path: str) -> List[Triple]:
    if os.path.exists(path):
        return load_triples(path)
    if os.path.exists(fallback_path):
        fallback = load_json(fallback_path)
        if isinstance(fallback, dict):
            for key in ("validated_triples", "stage7_validated_triples"):
                if key in fallback:
                    return load_triples(path) if False else [Triple(**t) for t in fallback[key]]
        if isinstance(fallback, list):
            return [t for t in [Triple(**item) for item in fallback] if t.validated]
    return []


def load_eval_set(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 3. RETRIEVAL (BM25 over verbalized triples — identical method for both pipelines)
# ---------------------------------------------------------------------------

class BM25Retriever:
    def __init__(self, triples: List[Triple]):
        self.triples = triples
        self.corpus = [t.verbalize() for t in triples]
        tokenized = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Triple]:
        if not self.bm25 or not self.triples:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.triples[i] for i in ranked[:top_k] if scores[i] > 0]


# ---------------------------------------------------------------------------
# 4. PROMPT CONSTRUCTION (identical template for both pipelines)
# ---------------------------------------------------------------------------

def build_prompt(question: str, retrieved: List[Triple]) -> str:
    validated_facts = [t.verbalize() for t in retrieved if t.validated]
    unvalidated_facts = [t.verbalize() for t in retrieved if not t.validated]

    validated_block = "\n".join(f"- {f}" for f in validated_facts) or "(none retrieved)"
    unvalidated_block = "\n".join(f"- {f}" for f in unvalidated_facts) or "(none retrieved)"

    return f"""You are answering a question using facts from a knowledge graph pipeline.

STAGE 7 VALIDATED TRIPLES (confirmed facts):
{validated_block}

STAGE 6 ONTOLOGY DESIGN FACTS (candidate facts / unvalidated relations):
{unvalidated_block}

Instructions:
- Answer the question directly and concisely. Do NOT list out the facts.
- Prefer stage 7 validated facts as your primary evidence.
- You may use stage 6 ontology facts only if no validated fact addresses the question,
  and if you do, explicitly say the answer relies on unvalidated information.
- If neither source addresses the question, say you don't have enough information
  rather than guessing.

Question: {question}
Answer:"""


# ---------------------------------------------------------------------------
# 5. LLM CALL
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    """Calls the Anthropic API. Requires ANTHROPIC_API_KEY env var and the
    `anthropic` package (pip install anthropic --break-system-packages)."""
    try:
        import anthropic
    except ImportError:
        return "[ERROR] `anthropic` package not installed. Run: pip install anthropic --break-system-packages"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERROR] ANTHROPIC_API_KEY environment variable not set."

    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()
    except Exception as e:
        return f"[ERROR] LLM call failed: {e}"


# ---------------------------------------------------------------------------
# 6. PIPELINE RUN
# ---------------------------------------------------------------------------

def run_pipeline(name: str, triples: List[Triple], eval_set: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    retriever = BM25Retriever(triples)
    results = []
    for item in eval_set:
        question = item["question"]
        retrieved = retriever.retrieve(question)
        prompt = build_prompt(question, retrieved)
        answer = call_llm(prompt)
        results.append({
            "pipeline": name,
            "question": question,
            "query_type": item.get("query_type", ""),
            "expected_answer": item.get("expected_answer", ""),
            "retrieved_facts": [t.verbalize() for t in retrieved],
            "num_validated_retrieved": sum(1 for t in retrieved if t.validated),
            "num_unvalidated_retrieved": sum(1 for t in retrieved if not t.validated),
            "answer": answer,
        })
    return results


# ---------------------------------------------------------------------------
# 7. SCORING (lightweight heuristic — swap in human review or LLM-as-judge later)
# ---------------------------------------------------------------------------

def heuristic_score(expected: str, answer: str) -> int:
    """Very rough 0-1 containment check. Replace with human scoring or an
    LLM-as-judge call for real evaluation — this is just a sanity-check signal."""
    if not expected:
        return -1  # not scorable
    return int(expected.lower().split("(")[0].strip() in answer.lower())


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

def main():
    _bootstrap_sample_files()

    all_triples = load_stage6_triples(ONTOLOGY_FILE, TRIPLES_FILE)
    validated_triples = load_stage7_triples(VALIDATED_FILE, TRIPLES_FILE)
    if not validated_triples:
        validated_triples = [t for t in all_triples if t.validated]
    if not all_triples:
        all_triples = [t for t in validated_triples] + [t for t in validated_triples if False]
    eval_set = load_eval_set(EVAL_SET_FILE)

    print(f"Loaded {len(all_triples)} stage 6 ontology facts "
          f"({len(validated_triples)} stage 7 validated triples, "
          f"{len(all_triples) - len(validated_triples)} stage 6-only / unvalidated)")
    print(f"Loaded {len(eval_set)} eval questions\n")

    results_a = run_pipeline("A_stage_6_ontology_design", all_triples, eval_set)
    results_b = run_pipeline("B_stage_7_validated_triples", validated_triples, eval_set)

    combined = []
    print(f"{'-'*100}")
    for a, b in zip(results_a, results_b):
        score_a = heuristic_score(a["expected_answer"], a["answer"])
        score_b = heuristic_score(b["expected_answer"], b["answer"])
        a["heuristic_score"] = score_a
        b["heuristic_score"] = score_b
        combined.extend([a, b])

        print(f"Q: {a['question']}  [{a['query_type']}]")
        print(f"  Expected        : {a['expected_answer']}")
        print(f"  Stage 6 ontology design: {a['answer']}  (score={score_a}, "
              f"val={a['num_validated_retrieved']}, unval={a['num_unvalidated_retrieved']})")
        print(f"  Stage 7 validated triples: {b['answer']}  (score={score_b}, "
              f"val={b['num_validated_retrieved']}, unval={b['num_unvalidated_retrieved']})")
        print(f"{'-'*100}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(combined, f, indent=2)

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", "question", "query_type", "expected_answer",
                         "answer", "heuristic_score", "num_validated_retrieved",
                         "num_unvalidated_retrieved"])
        for r in combined:
            writer.writerow([r["pipeline"], r["question"], r["query_type"],
                             r["expected_answer"], r["answer"], r["heuristic_score"],
                             r["num_validated_retrieved"], r["num_unvalidated_retrieved"]])

    total_a = sum(r["heuristic_score"] for r in results_a if r["heuristic_score"] >= 0)
    total_b = sum(r["heuristic_score"] for r in results_b if r["heuristic_score"] >= 0)
    scorable = sum(1 for r in results_a if r["heuristic_score"] >= 0)

    print(f"\nSUMMARY: Stage 6 ontology design scored {total_a}/{scorable} | Stage 7 validated triples scored {total_b}/{scorable}")
    print(f"Full results -> {RESULTS_FILE}")
    print(f"Summary CSV  -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
