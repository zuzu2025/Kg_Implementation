"""
No-LLM KG pipeline reference implementation for contract/enterprise text.

What it demonstrates:
  - deterministic ingestion with content hashes for incremental updates
  - regex/gazetteer entity extraction
  - blocking + pairwise features + clustering for Entity Resolution
  - rule-based relation extraction
  - ontology-style triple validation
  - optional end-to-end triple evaluation

Run:
  python kg_pipeline_no_llm.py --data-dir data --out-dir kg_no_llm_output --limit 25

Optional gold triples file:
  python kg_pipeline_no_llm.py --gold gold_triples.json

Gold format:
  [
    {"subject": "acme inc", "predicate": "PARTY_OF", "object": "supply agreement"},
    ...
  ]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


ENTITY_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "plc", "company", "gmbh", "ag", "sa", "lp", "llp",
}

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
}

MONTH_RE = (
    r"January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|"
    r"Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?"
)

DATE_RE = re.compile(rf"\b(?:{MONTH_RE})\s+\d{{1,2}},?\s+\d{{4}}\b", re.I)
ORG_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9&'\-]*(?:\s+[A-Z][A-Za-z0-9&'\-]*){0,8}\s+"
    r"(?:Inc\.?|Incorporated|Corp\.?|Corporation|LLC|Ltd\.?|Limited|PLC|"
    r"Company|LP|LLP)\b"
)
CONTRACT_HINT_RE = re.compile(
    r"\b([A-Z][A-Z0-9&,\- ]{4,90}\s+(?:AGREEMENT|CONTRACT|LICENSE|"
    r"AMENDMENT|LEASE|ORDER|STATEMENT OF WORK))\b"
)
JURISDICTION_RE = re.compile(
    r"\b(?:laws of|state of|province of|located in|located at|principal place "
    r"of business (?:at|in))\s+([A-Z][A-Za-z ]{3,40}?)(?=,|\.|\)|;|\band\b|\bwith\b|\s+\d|$)",
    re.I,
)


@dataclass(frozen=True)
class Document:
    doc_id: str
    path: str
    text: str
    sha256: str


@dataclass(frozen=True)
class Mention:
    mention_id: str
    text: str
    label: str
    doc_id: str
    sentence: str
    start: int
    end: int


@dataclass
class Entity:
    entity_id: str
    canonical: str
    label: str
    aliases: list[str]
    mention_ids: list[str]
    confidence: float


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    subject_type: str
    object_type: str
    confidence: float
    evidence: str
    doc_id: str


def normalize_text(value: str) -> str:
    value = value.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n,.;:()[]{}\"'")


def canonical_key(value: str, label: str | None = None) -> str:
    value = normalize_text(value).lower()
    value = re.sub(r"[^a-z0-9&' ]+", " ", value)
    tokens = [t for t in value.split() if t]
    if label == "PARTY":
        tokens = [t for t in tokens if t not in ENTITY_SUFFIXES]
    return " ".join(tokens)


def token_set(value: str) -> set[str]:
    return set(canonical_key(value).split())


def stable_id(prefix: str, parts: Iterable[str]) -> str:
    raw = "||".join(parts).encode("utf-8", errors="ignore")
    return f"{prefix}_{hashlib.sha1(raw).hexdigest()[:12]}"


def sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start))
    right_dot = text.find(".", end)
    right_nl = text.find("\n", end)
    rights = [p for p in (right_dot, right_nl) if p != -1]
    right = min(rights) if rights else min(len(text), end + 300)
    return normalize_text(text[left + 1:right + 1])


def read_documents(data_dir: Path, limit: int | None) -> list[Document]:
    docs = []
    for path in sorted(data_dir.glob("*.txt"))[:limit]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        sha = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        docs.append(Document(path.stem, str(path), text, sha))
    return docs


def load_manifest(out_dir: Path) -> dict[str, str]:
    path = out_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(out_dir: Path, docs: list[Document]) -> None:
    manifest = {d.path: d.sha256 for d in docs}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def extract_mentions(doc: Document) -> list[Mention]:
    """High-precision deterministic extraction. Replace/add CRF output here later."""
    mentions = []
    seen = set()

    title_guess = normalize_text(doc.path.split(os.sep)[-1].replace("_", " "))
    title_match = re.search(r"([A-Za-z ,&'\-]+ Agreement|Contract|License)", title_guess, re.I)
    if title_match:
        fake_start = 0
        mentions.append(Mention(
            stable_id("m", [doc.doc_id, title_match.group(0), "CONTRACT", "0"]),
            title_match.group(0), "CONTRACT", doc.doc_id, title_guess, fake_start, fake_start
        ))

    patterns = [
        ("EFFECTIVE_DATE", DATE_RE),
        ("PARTY", ORG_RE),
        ("CONTRACT", CONTRACT_HINT_RE),
    ]
    for label, pattern in patterns:
        for m in pattern.finditer(doc.text[:15000]):
            value = normalize_text(m.group(1) if label == "CONTRACT" else m.group(0))
            if label == "PARTY" and re.search(r"\b(shall|may|must|will|agreement|section)\b", value, re.I):
                continue
            key = (label, canonical_key(value, label), m.start())
            if len(value) < 3 or key in seen:
                continue
            seen.add(key)
            mentions.append(Mention(
                stable_id("m", [doc.doc_id, value, label, str(m.start())]),
                value, label, doc.doc_id, sentence_around(doc.text, m.start(), m.end()),
                m.start(), m.end()
            ))

    for m in JURISDICTION_RE.finditer(doc.text[:15000]):
        value = normalize_text(m.group(1))
        if canonical_key(value) in US_STATES or len(value.split()) <= 4:
            mentions.append(Mention(
                stable_id("m", [doc.doc_id, value, "JURISDICTION", str(m.start(1))]),
                value, "JURISDICTION", doc.doc_id,
                sentence_around(doc.text, m.start(1), m.end(1)), m.start(1), m.end(1)
            ))
    return mentions


def acronym(value: str) -> str:
    toks = [t for t in re.findall(r"[A-Za-z0-9]+", value) if t.lower() not in ENTITY_SUFFIXES]
    return "".join(t[0].lower() for t in toks if t)


def block_key(m: Mention) -> tuple[str, str]:
    key = canonical_key(m.text, m.label)
    toks = key.split()
    if not toks:
        return m.label, "#"
    if m.label in {"EFFECTIVE_DATE", "DATE"}:
        year = re.search(r"\b(19|20)\d{2}\b", key)
        return m.label, year.group(0) if year else toks[-1]
    return m.label, toks[0][:4]


def pair_features(a: Mention, b: Mention) -> dict[str, float]:
    ka = canonical_key(a.text, a.label)
    kb = canonical_key(b.text, b.label)
    ta, tb = token_set(a.text), token_set(b.text)
    union = ta | tb
    jaccard = len(ta & tb) / len(union) if union else 0.0
    char_sim = SequenceMatcher(None, ka, kb).ratio()
    acr_match = 1.0 if acronym(a.text) and acronym(a.text) == acronym(b.text) else 0.0
    same_doc = 1.0 if a.doc_id == b.doc_id else 0.0
    len_ratio = min(len(ka), len(kb)) / max(len(ka), len(kb), 1)
    return {
        "char_sim": char_sim,
        "token_jaccard": jaccard,
        "acronym_match": acr_match,
        "same_doc": same_doc,
        "len_ratio": len_ratio,
    }


def has_hard_conflict(a: Mention, b: Mention) -> bool:
    if a.label != b.label:
        return True
    years_a = set(re.findall(r"\b(?:19|20)\d{2}\b", a.text))
    years_b = set(re.findall(r"\b(?:19|20)\d{2}\b", b.text))
    if years_a and years_b and years_a.isdisjoint(years_b):
        return True
    if a.label == "PARTY":
        ka, kb = canonical_key(a.text, a.label), canonical_key(b.text, b.label)
        if len(ka) <= 3 or len(kb) <= 3:
            return ka != kb
    return False


def er_score(features: dict[str, float]) -> float:
    """Interpretable weighted pairwise ER model."""
    raw = (
        2.4 * features["char_sim"]
        + 2.0 * features["token_jaccard"]
        + 1.5 * features["acronym_match"]
        + 0.4 * features["same_doc"]
        + 0.7 * features["len_ratio"]
        - 3.1
    )
    return 1.0 / (1.0 + math.exp(-raw))


class UnionFind:
    def __init__(self, ids: Iterable[str]):
        self.parent = {i: i for i in ids}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def resolve_entities(mentions: list[Mention], threshold: float) -> tuple[list[Entity], list[dict]]:
    by_id = {m.mention_id: m for m in mentions}
    blocks = defaultdict(list)
    for m in mentions:
        blocks[block_key(m)].append(m)

    uf = UnionFind(by_id)
    candidates = []
    for block_mentions in blocks.values():
        for i, a in enumerate(block_mentions):
            for b in block_mentions[i + 1:]:
                if has_hard_conflict(a, b):
                    continue
                feats = pair_features(a, b)
                score = er_score(feats)
                candidates.append({
                    "left": a.mention_id, "right": b.mention_id, "score": round(score, 4),
                    "left_text": a.text, "right_text": b.text, "features": feats,
                    "merged": score >= threshold,
                })
                if score >= threshold:
                    uf.union(a.mention_id, b.mention_id)

    clusters = defaultdict(list)
    for mid in by_id:
        clusters[uf.find(mid)].append(by_id[mid])

    entities = []
    for cluster_mentions in clusters.values():
        labels = Counter(m.label for m in cluster_mentions)
        label = labels.most_common(1)[0][0]
        aliases = sorted({normalize_text(m.text) for m in cluster_mentions})
        canonical = Counter(canonical_key(m.text, label) for m in cluster_mentions).most_common(1)[0][0]
        if not canonical:
            canonical = canonical_key(aliases[0], label)
        confidence = min(1.0, 0.55 + 0.1 * len(cluster_mentions))
        entities.append(Entity(
            stable_id("e", [label, canonical]), canonical, label, aliases,
            [m.mention_id for m in cluster_mentions], round(confidence, 3)
        ))
    return entities, candidates


def mention_to_entity(entities: list[Entity]) -> dict[str, Entity]:
    out = {}
    for e in entities:
        for mid in e.mention_ids:
            out[mid] = e
    return out


def extract_relations(docs: list[Document], mentions: list[Mention], entities: list[Entity]) -> list[Triple]:
    by_doc = defaultdict(list)
    for m in mentions:
        by_doc[m.doc_id].append(m)
    m2e = mention_to_entity(entities)
    triples = []
    seen = set()

    def add(subj: Entity, pred: str, obj: Entity, conf: float, evidence: str, doc_id: str) -> None:
        key = (subj.entity_id, pred, obj.entity_id, doc_id)
        if key in seen or subj.entity_id == obj.entity_id:
            return
        seen.add(key)
        triples.append(Triple(
            subj.canonical, pred, obj.canonical, subj.label, obj.label,
            round(conf, 3), evidence[:300], doc_id
        ))

    for doc in docs:
        ms = by_doc[doc.doc_id]
        contracts = [m2e[m.mention_id] for m in ms if m.label == "CONTRACT"]
        parties = [m for m in ms if m.label == "PARTY"]
        dates = [m for m in ms if m.label == "EFFECTIVE_DATE"]
        juris = [m for m in ms if m.label == "JURISDICTION"]
        main_contract = contracts[0] if contracts else None

        if main_contract:
            for p in parties:
                if re.search(r"\b(by and between|between|party|parties)\b", p.sentence, re.I):
                    add(m2e[p.mention_id], "PARTY_OF", main_contract, 0.88, p.sentence, doc.doc_id)
            for d in dates[:2]:
                if re.search(r"\b(effective|made as of|dated as of|this agreement)\b", d.sentence, re.I):
                    add(main_contract, "EFFECTIVE_ON", m2e[d.mention_id], 0.9, d.sentence, doc.doc_id)

        for j in juris:
            if re.search(r"\b(governed by|laws of|state of)\b", j.sentence, re.I):
                anchor = main_contract or (m2e[parties[0].mention_id] if parties else None)
                if anchor:
                    add(anchor, "GOVERNED_BY", m2e[j.mention_id], 0.86, j.sentence, doc.doc_id)
            if re.search(r"\b(principal place of business|located at|located in)\b", j.sentence, re.I):
                nearby = [p for p in parties if p.sentence == j.sentence]
                for p in nearby[:2]:
                    add(m2e[p.mention_id], "LOCATED_IN", m2e[j.mention_id], 0.78, j.sentence, doc.doc_id)
    return triples


DOMAIN_RANGE = {
    "PARTY_OF": {("PARTY", "CONTRACT")},
    "EFFECTIVE_ON": {("CONTRACT", "EFFECTIVE_DATE"), ("CONTRACT", "DATE")},
    "GOVERNED_BY": {("CONTRACT", "JURISDICTION"), ("PARTY", "JURISDICTION")},
    "LOCATED_IN": {("PARTY", "JURISDICTION")},
}


def validate_triples(triples: list[Triple], min_confidence: float) -> tuple[list[Triple], list[dict]]:
    accepted, rejected = [], []
    for t in triples:
        allowed = DOMAIN_RANGE.get(t.predicate, set())
        type_ok = not allowed or (t.subject_type, t.object_type) in allowed
        if type_ok and t.confidence >= min_confidence:
            accepted.append(t)
        else:
            rejected.append({
                **asdict(t),
                "reason": "domain_range" if not type_ok else "low_confidence",
            })
    return accepted, rejected


def triple_key(t: dict | Triple) -> tuple[str, str, str]:
    if isinstance(t, Triple):
        return canonical_key(t.subject), t.predicate, canonical_key(t.object)
    return canonical_key(t["subject"]), t["predicate"], canonical_key(t["object"])


def evaluate_triples(predicted: list[Triple], gold_path: Path | None) -> dict:
    if not gold_path:
        return {"gold_file": None, "note": "No gold triples supplied."}
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    pred_keys = {triple_key(t) for t in predicted}
    gold_keys = {triple_key(t) for t in gold}
    tp = len(pred_keys & gold_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold_file": str(gold_path), "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "acceptance_target": "published triple precision >= 0.95",
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old_manifest = load_manifest(out_dir)
    docs = read_documents(data_dir, args.limit)
    changed = [d for d in docs if old_manifest.get(d.path) != d.sha256]
    active_docs = changed if args.incremental else docs

    mentions = [m for d in active_docs for m in extract_mentions(d)]
    entities, er_candidates = resolve_entities(mentions, args.er_threshold)
    raw_triples = extract_relations(active_docs, mentions, entities)
    triples, rejected = validate_triples(raw_triples, args.min_triple_confidence)
    metrics = evaluate_triples(triples, Path(args.gold) if args.gold else None)

    write_json(out_dir / "documents.json", [asdict(d) for d in active_docs])
    write_json(out_dir / "mentions.json", [asdict(m) for m in mentions])
    write_json(out_dir / "er_candidates.json", er_candidates)
    write_json(out_dir / "entities.json", [asdict(e) for e in entities])
    write_json(out_dir / "raw_triples.json", [asdict(t) for t in raw_triples])
    write_json(out_dir / "validated_triples.json", [asdict(t) for t in triples])
    write_json(out_dir / "rejected_triples.json", rejected)
    write_json(out_dir / "metrics.json", metrics)
    save_manifest(out_dir, docs)

    return {
        "documents_seen": len(docs),
        "documents_processed": len(active_docs),
        "mentions": len(mentions),
        "entities": len(entities),
        "er_pairs_scored": len(er_candidates),
        "raw_triples": len(raw_triples),
        "validated_triples": len(triples),
        "rejected_triples": len(rejected),
        "metrics": metrics,
        "out_dir": str(out_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="kg_no_llm_output")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--incremental", action="store_true", help="Only process changed docs by manifest hash.")
    parser.add_argument("--er-threshold", type=float, default=0.72)
    parser.add_argument("--min-triple-confidence", type=float, default=0.75)
    parser.add_argument("--gold", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    summary = run(parse_args())
    print(json.dumps(summary, indent=2))
