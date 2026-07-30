# From Unstructured Text to Knowledge Graphs: A Hybrid Approach

**Integrating Retrieval-Augmented Generation and Comparing LLM-based vs Classical Machine Learning Pipelines**

Summer Internship Project — Machine Learning Lab, IIIT Hyderabad
Supervisor: Dr. Naresh Manwani

---

## Overview

This project builds and compares two independent pipelines for constructing Knowledge Graphs (KGs) from unstructured legal contract text, and layers a Retrieval-Augmented Generation (RAG) system on top of the resulting graph for natural-language question answering.

Two KG construction approaches were implemented end-to-end on the same corpus:

1. **Hybrid Pipeline** — combines an LLM (Groq's Llama 3.3 70B) with classical algorithms across 5 stages: ontology induction, information extraction, entity resolution, KG population, and LLM-based validation.
2. **Classical Pipeline** — an 11-stage, primarily non-LLM pipeline (CRF, SVM, Random Forest, BM25, clustering) that calls an LLM only once, to assist in labeling initial training data. Built to test whether comparable KG quality can be achieved with greater interpretability and lower computational cost than the hybrid approach.

The completed knowledge graph from the classical pipeline feeds into **Docket**, a local RAG system that answers natural-language questions over the contracts using hybrid BM25 + dense retrieval, graph-aware re-ranking, and multi-hop reasoning over validated KG facts.

## Dataset

[**CUAD (Contract Understanding Atticus Dataset)**](https://arxiv.org/abs/2103.06268) — 510 commercial legal contracts sourced from SEC EDGAR filings, spanning 25 contract types, with 13,000+ expert clause annotations across 41 categories. Used here purely as a source of realistic unstructured enterprise text; the KG pipelines do not use CUAD's original clause-extraction labels directly.

## Repository Structure

```
.
├── stage1_preprocessing.py            # Stage 1: sentence/chunk splitting, per-contract stats
├── genrate_training_data_for_crf.py   # Stage 1: LLM-assisted CRF training-data labeling
├── stage2_ner_crf.py                  # Stage 2: Named Entity Recognition (CRF)
├── stage3_entity-linking.py           # Stage 3: Entity Linking (BM25 + Levenshtein)
├── stage3_entity-linking-v2.py        # Stage 3: revised entity linking (year-conflict guard, etc.)
├── stage5_relation-extraction.py      # Stage 5: Relation Extraction (OpenIE + SVM/RF/NB)
├── stage7_triple_validation.py        # Stage 7: Triple Validation (domain/range checks)
├── stage8_kg_completion.py            # Stage 8: KG Completion (link prediction)
├── stage9_storage.py                  # Stage 9: Graph export (GraphML/JSON/SQLite)
├── stage10_evaluation.py              # Stage 10: Evaluation (confidence-weighted proxy metrics)
├── kg_pipeline_no_llm.py              # Classical pipeline orchestrator
├── kg_no_llm_algorithms.py            # Shared classical-pipeline algorithm utilities
├── kg_output/
│   └── knowledge_graph.graphml        # Final exported knowledge graph
├── outputs_ML/                        # Intermediate + final JSON outputs per stage
│   ├── coreference_results.json
│   ├── entity_clusters.json
│   ├── linked_entities_v2.json
│   └── ...
├── RAG_Model/
│   ├── rag_engine.py                  # Retrieval + graph traversal + LLM answer synthesis
│   ├── rag_server.py                  # Local web server for the Docket UI
│   ├── templates/
│   │   └── index.html                 # Docket front-end
│   ├── .env.example                   # Template for required API keys (see below)
│   └── README.md
├── vis-network.min.js                 # Bundled for the offline dashboard/graph explorer
├── chart.umd.js                       # Bundled for the offline dashboard
├── requirements.txt
└── README.md
```

> Note: Stage 4 (coreference resolution), Stage 6 (ontology induction), and Stage 11 (visualization) scripts should sit alongside the files above following the same `stageN_*.py` naming convention — add/rename as applicable to match your local layout.

## Pipeline Stages

### Hybrid Pipeline (5 stages)
| Stage | Description |
|---|---|
| 1. Ontology Induction | LLM (Llama 3.3 70B) induces entity/relation schema from sampled contract excerpts |
| 2. Information Extraction | Regex + dependency-pattern based entity and relation extraction |
| 3. Entity Resolution | BM25 + Levenshtein + TF-IDF ensemble similarity, with character-based blocking |
| 4. KG Population | Assembles resolved entities/relations into a directed graph (NetworkX) |
| 5. LLM-Based Validation | LLM rates sampled nodes/edges for type correctness and semantic validity |

### Classical Pipeline (11 stages)
| Stage | Description |
|---|---|
| 1. Training Data Labeling | One-time LLM-assisted labeling of ~5,100 sentences for CRF training |
| 2. NER (CRF) | Conditional Random Field entity tagger across 6 entity types |
| 3. Entity Linking | BM25 + Levenshtein similarity with a year-conflict guard for dates |
| 4. Coreference Resolution | Hobbs algorithm + SVM classifier (POS/entity-type filtering for pronouns) |
| 5. Relation Extraction | OpenIE-style extraction + SVM/Random Forest/Naive Bayes classifiers |
| 6. Ontology Induction | K-Means / HAC clustering + rule induction over extracted relations |
| 7. Triple Validation | Domain/range checks, automatic repair, confidence scoring |
| 8. KG Completion | Link prediction via graph-topology features (Random Forest) |
| 9. Storage | Export to GraphML, JSON, and SQLite |
| 10. Evaluation | Confidence-weighted precision/recall/F1 proxy metrics |
| 11. Visualization | Self-contained offline HTML dashboard (vis-network + Chart.js) |

## Key Results

| Metric | Hybrid Pipeline | Classical Pipeline |
|---|---|---|
| Final graph size | 7,472 nodes / 2,188 edges | 650 nodes / 621 edges |
| Entity coverage | ~52.6% of nodes typed; only 17 typed nodes have edges | 100% of entities connected to ≥1 relation |
| Quality metric | LLM-judged Overall Quality: **4.14/5** | Confidence-weighted proxy — Precision: 0.515, Recall: 0.870, F1: **0.647** |
| LLM calls required | 2 stages (ontology induction + validation), repeated per batch | 1 one-time call (training data labeling only) |
| Interpretability | Lower (LLM reasoning not decomposable) | Higher (every stage exposes inspectable features) |

Full methodology, failure analysis (feature leakage, sampling truncation bugs, cascading errors across stages), and the comparative discussion are in the accompanying internship report.

## RAG System — Docket

A local, offline-capable RAG interface built on top of the classical pipeline's completed KG:

- **Hybrid retrieval**: BM25 (lexical) + `all-MiniLM-L6-v2` dense embeddings, fused via Reciprocal Rank Fusion
- **Graph-aware re-ranking**: prioritizes text near the query's recognized entities (up to 2 hops)
- **Multi-hop reasoning**: searches for connecting paths between multiple recognized entities (up to 4 hops)
- **Grounded synthesis**: answers generated only from retrieved evidence via Groq (`openai/gpt-oss-120b`), with explicit fallback to raw retrieved text if no API key is configured
- Runs entirely locally aside from the final answer-synthesis LLM call

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Environment Variables

This project requires API keys for the LLM-assisted stages and RAG answer synthesis. **Never commit real keys** — copy the template and fill in your own values locally:

```bash
cp RAG_Model/.env.example RAG_Model/.env
```

```
GROQ_API_KEY=your_key_here
GCP_API_KEY=your_key_here     # only if using GCP-backed components
```

`.env` is listed in `.gitignore` and must stay untracked. Load keys in code via:

```python
import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]
```

## Running the Pipelines

**Classical pipeline (end-to-end):**
```bash
python kg_pipeline_no_llm.py
```

**Individual stages** (useful for debugging a specific stage in isolation):
```bash
python stage1_preprocessing.py
python stage2_ner_crf.py
python stage3_entity-linking-v2.py
python stage5_relation-extraction.py
python stage7_triple_validation.py
python stage8_kg_completion.py
python stage9_storage.py
python stage10_evaluation.py
```

**RAG server (Docket):**
```bash
python RAG_Model/rag_server.py
```
Then open the local address printed in the terminal to access the Docket UI.

## Known Limitations

- No gold-standard KG exists for CUAD, so both pipelines are evaluated via proxy metrics rather than true precision/recall.
- Entity resolution recovers relatively few genuine semantic duplicates (formatting-level merges dominate in both pipelines).
- The classical pipeline's induced ontology and domain rules are heavily `PARTY`-centric.
- Neither pipeline currently handles multimodal contract content (tables, scanned exhibits, signature blocks).
- KG construction is a single batch process; incremental updates as new contracts arrive are not yet supported.

See the full report for the complete limitations and future-work discussion.

## Acknowledgments

Built during a summer internship at the Machine Learning Lab, IIIT Hyderabad, under the supervision of Dr. Naresh Manwani.
