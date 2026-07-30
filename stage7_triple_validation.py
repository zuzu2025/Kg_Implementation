"""
Stage 7: Triple Validation
===========================
Input:  raw extracted triples (from Stage 5) + ontology (from Stage 6)
Output: validated_triples.json, rejected_triples.json

What this does:
  1. Domain/range check — does this relation type connect the entity types
     the ontology says it should? (e.g. ENTERED_INTO must go PARTY -> CONTRACT,
     not PARTY -> PARTY)
  2. Confidence scoring — how many separate sentences/mentions support this
     exact triple, normalized against how often the subject appears overall.
  3. Triples that fail either check are rejected with a clear reason.

Run:
    python stage7_triple_validation.py
"""

import json
import os
import math
import re
from collections import Counter, defaultdict
from kg_no_llm_algorithms import DOMAIN_RANGE, relation_type_ok


def normalize_mention(value):
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    value = value.strip(" \t\r\n,.;:()[]{}\"'")
    return value


def mention_quality(value):
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


def harmonic_mean(values):
    values = [v for v in values if v > 0]
    if not values:
        return 0.0
    return len(values) / sum(1 / v for v in values)


# For a handful of relations, general contract-law domain knowledge gives a
# strong, confident prior on what the OBJECT type should be — independent of
# whatever the induced ontology (outputs_ML/ontology.json) says. This matters
# because those induced rules come from the same OpenIE entity-pairing step
# in Stage 5 that has a known bias: it pairs every two entities within a
# short span in a sentence, and since PARTY is by far the most frequent
# entity type, PARTY ends up as the "range" for GOVERNED_BY/LOCATED_IN/
# ORGANIZED_UNDER in the induced ontology even though that's not what those
# relations actually mean. Relying on the induced ontology alone to validate
# triples would just confirm that same bias rather than catch it. This check
# is intentionally conservative — only applied where the expected type is
# genuinely unambiguous, not a rewrite of the domain/range system.
SEMANTIC_OBJECT_PRIORS = {
    'GOVERNED_BY': {'JURISDICTION'},
    'LOCATED_IN': {'JURISDICTION'},
    'ORGANIZED_UNDER': {'JURISDICTION'},
    'EFFECTIVE_ON': {'DATE', 'EFFECTIVE_DATE'},
}


MONTH_RE = (
    r"January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|"
    r"Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?"
)
DATE_RE = re.compile(
    rf"\b(?:{MONTH_RE})\s+\d{{1,2}}(?:,?\s+\d{{4}})?\b|"
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    re.I,
)
# STRONG markers: words that, on their own, reliably indicate an actual
# contract/document (a title ends in "Agreement", "License", etc.). These
# are the only signals allowed to type something as CONTRACT.
STRONG_CONTRACT_TITLE_RE = re.compile(
    r"\b(?:agreement|contract|license|lease|amendment|order|"
    r"statement of work|sow|mou|memorandum of understanding)\b",
    re.I,
)

# WEAK terms: ordinary legal/business vocabulary that often *appears inside*
# a real contract title ("Distribution Agreement", "Hosting Agreement") but
# is extremely common as a bare noun with no contract attached at all ("the
# affiliate shall...", "cooperation between the parties"). A weak term by
# itself must NEVER be enough to type something as CONTRACT — that was the
# root cause of nonsense triples like ("commnet wireless, llc", ENTERED_INTO,
# "affiliate") passing validation. Kept only for reference/documentation;
# not used to drive type inference.
WEAK_CONTRACT_TERMS = {
    "supply", "services", "maintenance", "distribution", "agency",
    "collaboration", "cooperation", "manufacturing", "hosting",
    "reseller", "affiliate", "endorsement", "sponsorship",
}
JURISDICTION_RE = re.compile(
    r"\b(?:laws of|state of|province of|jurisdiction of|organized under|"
    r"incorporated under|located in|located at)\s+"
    r"([A-Z][A-Za-z ]{2,40}?)(?=,|\.|\)|;|\band\b|\bwith\b|\s+\d|$)",
    re.I,
)
KNOWN_JURISDICTIONS = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "illinois", "indiana",
    "iowa", "kansas", "kentucky", "louisiana", "maryland", "massachusetts",
    "michigan", "minnesota", "missouri", "nevada", "new jersey", "new york",
    "north carolina", "ohio", "oregon", "pennsylvania", "tennessee", "texas",
    "utah", "virginia", "washington", "wisconsin", "england", "bermuda",
    "canada", "china", "germany", "france", "ireland", "israel",
}


# ═══════════════════════════════════════════════════════════════
# DEMO DATA — replace load_ontology_and_triples() with a loader
# that reads your real Stage 5 relations.json + Stage 6 ontology.json
# once they're ready.
# ═══════════════════════════════════════════════════════════════

def load_ontology_and_triples():
    """
    Load real data from Stage 5 (relations) and Stage 6 (ontology).
    """
    # Load ontology from Stage 6
    with open("outputs_ML/ontology.json", "r") as f:
        ontology_raw = json.load(f)
    
    # Build relation constraints from induced domain rules
    relation_constraints = defaultdict(list)
    for rule in ontology_raw.get("domain_rules", []):
        relation = rule["relation"]
        relation_constraints[relation].append({
            "domain": rule["subject"],
            "range": rule["object"],
            "support": rule.get("support", 0.0),
            "confidence": rule.get("confidence", 0.5),
            "frequency": rule.get("frequency", 0)
        })
    
    ontology = {
        "entity_types": ontology_raw.get("entity_types", []),
                "relation_constraints": dict(relation_constraints)
    }
    
    # Load relations from Stage 5
    with open("outputs_ML/relations.json", "r") as f:
        relations = json.load(f)
    
    # Convert to triple format
    raw_triples = []
    for rel in relations:
        if rel.get("relation", "OTHER") == "OTHER":
            continue  # skip unmapped relations
        subj = normalize_mention(rel.get("canonical1") or rel.get("entity1"))
        obj = normalize_mention(rel.get("canonical2") or rel.get("entity2"))
        
        raw_triples.append((
            subj,           # subject
            rel.get("label1", "PARTY"),  # subject type
            rel["relation"],           # relation
            obj,           # object
            rel.get("label2", "PARTY"),  # object type
            normalize_mention(rel.get("predicate", "")),
            rel.get("sentence", ""),
            rel.get("filename", "")  # PATCH: doc identity, for fan-out scoping + spot-check traceability
        ))

    print(f"Loaded {len(raw_triples)} mapped triples from Stage 5")
    print(f"Loaded {len(relation_constraints)} relation constraints from Stage 6")
    
    return ontology, raw_triples


# Relations that are semantically single-valued per subject: a contract has
# ONE effective date, a party is organized under ONE jurisdiction, governed
# by ONE law, located in ONE place. When one subject shows up paired with
# many distinct objects for these relations, that's not many true facts —
# it's Stage 5's short-span entity-pairing grabbing unrelated nearby entities
# (e.g. a boilerplate section header paired with every date/party near it).
# Cap to the best-supported object(s) per subject before these ever become
# validation candidates, so pairing noise doesn't inflate the denominator.
FANOUT_CAP_RELATIONS = {
    "EFFECTIVE_ON": 1,
    "ORGANIZED_UNDER": 1,
    "GOVERNED_BY": 1,
    "LOCATED_IN": 1,
}


def cap_fanout(raw_triples):
    # PATCH — root cause of the 68.1% -> 51.4% regression traced here.
    #
    # This cap was originally keyed on (subject, relation) ONLY, globally
    # across the entire corpus. That was fine back when it was catching
    # same-sentence pairing noise (a boilerplate header fanned out to every
    # nearby date/party). But once stage5's token-gap fix cleaned up the
    # actual noise mechanism, this cap kept running — and a GLOBAL key means
    # a generic canonical string that legitimately recurs across many of the
    # 510 source contracts (e.g. "agreement", or a company name that's a
    # party to several different deals in the corpus) looks identical to one
    # subject with 40 fake relations. The cap kept only the single
    # best-supported object across the WHOLE corpus and silently dropped
    # every other document's true, distinct fact.
    #
    # Fix: scope the cap key to (subject, relation, filename). A contract
    # still only gets to keep one EFFECTIVE_ON object *within that document*
    # (still true — one contract, one effective date), but the same subject
    # string showing up in a different document is no longer treated as the
    # same fan-out group. This is what the doc_id/filename field threaded
    # through from stage5 is for.
    object_counts = defaultdict(lambda: defaultdict(int))
    for subj, subj_t, rel, obj, obj_t, predicate, sentence, filename in raw_triples:
        if rel in FANOUT_CAP_RELATIONS:
            object_counts[(subj, rel, filename)][obj] += 1

    keep_objects = {}
    for key, counts in object_counts.items():
        _, rel, _ = key
        cap = FANOUT_CAP_RELATIONS[rel]
        ranked = sorted(counts.items(), key=lambda x: -x[1])
        keep_objects[key] = {o for o, _ in ranked[:cap]}

    dropped = 0
    filtered = []
    for row in raw_triples:
        subj, subj_t, rel, obj, obj_t, predicate, sentence, filename = row
        if rel in FANOUT_CAP_RELATIONS and obj not in keep_objects[(subj, rel, filename)]:
            dropped += 1
            continue
        filtered.append(row)

    print(f"Fan-out cap: dropped {dropped} pairing-noise candidates "
          f"across {len(object_counts)} (subject, relation, document) groups "
          f"— cap is now scoped per source document, not global")
    return filtered


# ═══════════════════════════════════════════════════════════════
# VALIDATION LOGIC
# ═══════════════════════════════════════════════════════════════

def triple_key_from_row(row):
    subj, subj_t, rel, obj, obj_t, predicate, sentence, filename = row
    return subj, subj_t, rel, obj, obj_t


def triple_key_from_record(record):
    return (
        record["subject"],
        record["subject_type"],
        record["relation"],
        record["object"],
        record["object_type"],
    )


def infer_type(value, current_type):
    text = normalize_mention(value)
    lowered = text.lower()
    if DATE_RE.search(text):
        return "DATE"
    # Only STRONG markers can type something as CONTRACT. A bare weak term
    # ("affiliate", "services") is not evidence of a contract on its own.
    if STRONG_CONTRACT_TITLE_RE.search(text):
        return "CONTRACT"
    if lowered in KNOWN_JURISDICTIONS or any(j in lowered for j in KNOWN_JURISDICTIONS):
        return "JURISDICTION"
    return current_type


# Pronouns and other referring words Stage 5 occasionally captures as if
# they were named entities ("this", "that", "herein"). These never refer
# to anything on their own — there's no entity here to repair.
PRONOUN_ARTIFACTS = {
    "this", "that", "it", "these", "those", "such", "same", "which",
    "who", "whom", "herein", "hereof", "hereto", "hereunder",
}

# Boilerplate clause/section-numbering fragments Stage 5's short-span
# pairing sometimes grabs as entities ("1 definitions 1", "2 2",
# "2 affiliate 1 2"). These are structural artifacts of the document, not
# names of anything, so they must never become the anchor of a fabricated
# fact via repair.
def is_entity_artifact(value):
    text = normalize_mention(value)
    if not text:
        return True
    lowered = text.lower()
    if lowered in PRONOUN_ARTIFACTS:
        return True
    if re.fullmatch(r"[\d\s]+", lowered):
        return True
    tokens = lowered.split()
    numeric_tokens = sum(1 for t in tokens if re.fullmatch(r"\d+", t))
    # Two or more numeric tokens dominating a short span ("2 affiliate 1
    # 2", "1 definitions 1") is section/clause numbering, not a name.
    if numeric_tokens >= 2 and numeric_tokens >= len(tokens) - 2:
        return True
    return False


def window_around(sentence, anchor, window=80):
    """Character window of `sentence` surrounding `anchor`'s own mention,
    used so date/contract/jurisdiction extraction only looks near the
    entity it's supposed to describe instead of anywhere in the sentence.
    Returns None if the anchor can't be located in the sentence at all —
    callers should treat that as "no match" rather than falling back to a
    sentence-wide scan.
    """
    if not sentence or not anchor:
        return None
    pos = sentence.lower().find(normalize_mention(anchor).lower())
    if pos == -1:
        return None
    lo = max(0, pos - window)
    hi = min(len(sentence), pos + len(anchor) + window)
    return sentence[lo:hi]


def extract_date(sentence, anchor=None, window=80):
    text = sentence or ""
    if anchor is not None:
        scoped = window_around(sentence, anchor, window)
        if scoped is None:
            return ""
        text = scoped
    match = DATE_RE.search(text)
    return normalize_mention(match.group(0)) if match else ""


def extract_contract(sentence, anchor=None, window=80):
    text = sentence or ""
    if anchor is not None:
        scoped = window_around(sentence, anchor, window)
        if scoped is None:
            return ""
        text = scoped
    text = normalize_mention(text)
    if not text:
        return ""
    match = re.search(
        r"\b([A-Z][A-Za-z0-9&,'\- ]{0,80}?"
        r"(?:Agreement|Contract|License|Lease|Amendment|Order|Statement of Work))\b",
        text,
        re.I,
    )
    if match:
        return normalize_mention(match.group(1))
    # Fallback now requires a STRONG marker in-window, not any weak term —
    # this used to fire on words like "services"/"affiliate" and return
    # the fabricated placeholder name "agreement" regardless of whether a
    # contract was actually being discussed nearby.
    if STRONG_CONTRACT_TITLE_RE.search(text):
        return "agreement"
    return ""


def extract_jurisdiction(sentence, anchor=None, window=80):
    text = sentence or ""
    if anchor is not None:
        scoped = window_around(sentence, anchor, window)
        if scoped is None:
            return ""
        text = scoped
    match = JURISDICTION_RE.search(text)
    if match:
        return normalize_mention(match.group(1))
    lowered = text.lower()
    for jurisdiction in sorted(KNOWN_JURISDICTIONS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(jurisdiction)}\b", lowered):
            return jurisdiction
    return ""


def is_plausible_party(value):
    text = normalize_mention(value)
    if not text or is_entity_artifact(text):
        return False
    if infer_type(text, "PARTY") != "PARTY":
        return False
    return mention_quality(text) > 0.0


def repair_triple(row):
    subj, subj_t, rel, obj, obj_t, predicate, sentence, filename = row
    subj = normalize_mention(subj)
    obj = normalize_mention(obj)

    # Hard gate: pronouns and clause/section-numbering fragments are not
    # entities. There's nothing to repair here — inventing a replacement
    # for them is exactly the fabrication this function used to do
    # ("1 definitions 1", "this", "2 2" all used to end up as validated
    # subjects). Fail outright instead of attempting repair.
    if is_entity_artifact(subj) or is_entity_artifact(obj):
        return None, "artifact_entity"

    if not subj or not obj or subj.lower() == obj.lower():
        return None, "empty_or_self"

    subj_t = infer_type(subj, subj_t)
    obj_t = infer_type(obj, obj_t)

    # Repair #1 — direction swap. If the pair validates with subject and
    # object reversed, it's the same fact stated backwards, not a new one.
    # Both original entities are kept; nothing is invented.
    if not relation_type_ok(rel, subj_t, obj_t) and relation_type_ok(rel, obj_t, subj_t):
        return (
            obj, obj_t, rel, subj, subj_t, predicate, sentence, filename
        ), "direction_swap"

    # Repair #2 — type relabel. Corrected type inference (e.g. Stage 5
    # mislabeled a date/jurisdiction-shaped string as PARTY) can make the
    # ORIGINAL pair schema-valid as-is. The entities themselves are never
    # touched — only their type label changes, and only using the tightened
    # STRONG-marker-only CONTRACT inference above.
    if relation_type_ok(rel, subj_t, obj_t):
        return (
            subj, subj_t, rel, obj, obj_t, predicate, sentence, filename
        ), "type_relabel"

    # Repair #3 — fill one missing role near a verified anchor. One side
    # already carries the type this relation needs (a real CONTRACT, a
    # verified PARTY, a JURISDICTION) — that side is kept untouched. The
    # other side, if missing, is searched for ONLY in a character window
    # around the anchor's own mention in the sentence, never sentence-wide.
    # If no legitimate anchor exists on either side, the triple is left
    # unrepaired rather than fabricating both roles from scratch — this is
    # the change that stops repairs like turning
    # (array biopharma inc, EFFECTIVE_ON, ono pharmaceutical co) into an
    # unrelated (some other contract title, EFFECTIVE_ON, some other date)
    # grabbed from elsewhere in the sentence.
    if rel == "EFFECTIVE_ON":
        contract = subj if subj_t == "CONTRACT" else (obj if obj_t == "CONTRACT" else "")
        date = subj if subj_t in {"DATE", "EFFECTIVE_DATE"} else (obj if obj_t in {"DATE", "EFFECTIVE_DATE"} else "")
        if contract and not date:
            date = extract_date(sentence, anchor=contract)
        elif date and not contract:
            contract = extract_contract(sentence, anchor=date)
        if contract and date:
            return (
                contract, "CONTRACT", rel, date, "DATE",
                predicate, sentence, filename
            ), "effective_on_contract_date_repair"
        return None, "unrepairable_schema_violation"

    if rel in {"PARTY_OF", "ENTERED_INTO"}:
        party = subj if is_plausible_party(subj) else (obj if is_plausible_party(obj) else "")
        contract = subj if subj_t == "CONTRACT" else (obj if obj_t == "CONTRACT" else "")
        if party and not contract:
            contract = extract_contract(sentence, anchor=party)
        if party and contract:
            return (
                party, "PARTY", rel, contract, "CONTRACT",
                predicate, sentence, filename
            ), f"{rel.lower()}_party_contract_repair"
        return None, "unrepairable_schema_violation"

    if rel in {"GOVERNED_BY", "LOCATED_IN", "ORGANIZED_UNDER"}:
        jurisdiction = subj if subj_t == "JURISDICTION" else (obj if obj_t == "JURISDICTION" else "")

        if rel == "GOVERNED_BY":
            if subj_t in {"CONTRACT", "PARTY"}:
                anchor, anchor_t = subj, subj_t
            elif obj_t in {"CONTRACT", "PARTY"}:
                anchor, anchor_t = obj, obj_t
            else:
                anchor, anchor_t = "", ""
        else:
            if is_plausible_party(subj):
                anchor, anchor_t = subj, "PARTY"
            elif is_plausible_party(obj):
                anchor, anchor_t = obj, "PARTY"
            else:
                anchor, anchor_t = "", ""

        if anchor and not jurisdiction:
            jurisdiction = extract_jurisdiction(sentence, anchor=anchor)
        if anchor and jurisdiction:
            return (
                anchor, anchor_t, rel, jurisdiction, "JURISDICTION",
                predicate, sentence, filename
            ), f"{rel.lower()}_jurisdiction_repair"
        return None, "unrepairable_schema_violation"

    return None, "unrepairable_schema_violation"


def validate_with_repairs(raw_triples, ontology, min_confidence=0.22):
    original_validated, original_rejected = validate_triples(
        raw_triples, ontology, min_confidence=min_confidence
    )
    rejected_keys = {triple_key_from_record(record) for record in original_rejected}

    unique_failed_rows = []
    seen_failed = set()
    for row in raw_triples:
        key = triple_key_from_row(row)
        if key in rejected_keys and key not in seen_failed:
            seen_failed.add(key)
            unique_failed_rows.append(row)

    # FIX (round 2): the previous version deduped repaired triples down to
    # one canonical record per repaired (subject, relation, object), added
    # that canonical record once via a "repaired_validated" pass, and THEN
    # separately looped over every original_rejected row and added a
    # "merged" duplicate for any row whose repair matched a canonical
    # record. That double-counts: the one original row whose repair *was*
    # the canonical record got added twice — once as the canonical entry,
    # once as its own "merged" duplicate. With N originals collapsing onto
    # 1 canonical form, that's N+1 output rows instead of N, which is
    # exactly the over-count AssertionError caught (854+16 vs 702).
    #
    # Fix: don't dedupe repaired triples into a shared canonical record at
    # all. Every original_rejected row gets repaired and re-validated
    # independently, and gets EXACTLY one verdict — validated or rejected —
    # based on its own outcome. This guarantees
    #   len(validated) + len(rejected) == len(original_validated) + len(original_rejected)
    # by construction (every original_rejected row contributes exactly one
    # record to exactly one of the two output lists), with no dedup pass
    # in between that could double- or zero-count a row.
    #
    # Trade-off: multiple original triples that repair to the same
    # canonical (subject, relation, object) will now appear as multiple
    # validated records sharing that same repaired content — each still
    # carries its own original_subject/original_object for traceability.
    # If you want a deduplicated knowledge graph, group validated_triples.json
    # by (subject, relation, object) downstream; if you want full
    # input-triple traceability (matching stage5_unique_input_triples),
    # this is that view.
    repair_rows = []
    repair_notes_by_key = {}
    repair_failures = {}
    repaired_by_original_key = {}
    for row in unique_failed_rows:
        orig_key = triple_key_from_row(row)
        repaired, note = repair_triple(row)
        if repaired is None:
            repair_failures[orig_key] = note
            continue
        repair_rows.append(repaired)
        repair_notes_by_key[triple_key_from_row(repaired)] = note
        repaired_by_original_key[orig_key] = (repaired, note)

    # Still validate all repaired candidates together (not deduped away —
    # duplicates in repair_rows just mean validate_triples computes
    # mention/confidence support across the full repaired-candidate pool,
    # which is what we want for scoring). We just don't use its dedup as
    # the source of the final output rows anymore.
    repaired_validated, repaired_rejected = validate_triples(
        repair_rows, ontology, min_confidence=min_confidence
    )
    repaired_valid_keys = {triple_key_from_record(record) for record in repaired_validated}

    validated = []
    for record in original_validated:
        record["validation_status"] = "validated_original"
        validated.append(record)

    rejected = []
    failed_row_by_key = {triple_key_from_row(row): row for row in unique_failed_rows}
    original_rejected_repaired_successfully = 0
    for record in original_rejected:
        key = triple_key_from_record(record)
        original_row = failed_row_by_key.get(key)
        orig_key = triple_key_from_row(original_row) if original_row else None
        repaired_note = repaired_by_original_key.get(orig_key) if orig_key else None

        if repaired_note is None:
            # repair_triple() returned None outright for this row.
            record["validation_status"] = "rejected_after_repair"
            record["repair_status"] = repair_failures.get(orig_key, "unrepairable_schema_violation")
            rejected.append(record)
            continue

        repaired, note = repaired_note
        repaired_key = triple_key_from_row(repaired)
        if repaired_key in repaired_valid_keys:
            original_rejected_repaired_successfully += 1
            out = dict(record)
            out.update({
                "subject": repaired[0], "subject_type": repaired[1],
                "relation": repaired[2],
                "object": repaired[3], "object_type": repaired[4],
                "validation_status": "validated_after_repair",
                "repair_reason": note,
                "original_subject": record["subject"],
                "original_subject_type": record["subject_type"],
                "original_object": record["object"],
                "original_object_type": record["object_type"],
            })
            validated.append(out)
        else:
            record["validation_status"] = "rejected_after_repair"
            record["repair_status"] = "repair_failed_validation"
            rejected.append(record)

    original_unique = len(original_validated) + len(original_rejected)
    original_resolved = len(original_validated) + original_rejected_repaired_successfully
    rejected_status_counts = Counter(r["repair_status"] for r in rejected)

    # These two rates must be reported side by side, never collapsed into
    # one headline number: original_pass_rate is what the validator does
    # with no repair step at all (the honest, unrepaired figure) and
    # post_repair_pass_rate is what you get after the conservative repairs
    # above are folded in. A shrinking gap between them is the signal that
    # repairs are behaving (fewer, higher-precision fixes); a growing gap
    # is the signal to go back and audit repair_triple() again.
    original_pass_rate = round(len(original_validated) / original_unique * 100, 1) if original_unique else 0.0
    post_repair_pass_rate = round(original_resolved / original_unique * 100, 1) if original_unique else 0.0

    summary = {
        "stage5_unique_input_triples": original_unique,
        "original_validated": len(original_validated),
        "original_rejected": len(original_rejected),
        "original_pass_rate": original_pass_rate,
        "repair_candidates": len(unique_failed_rows),
        "repair_attempted": len(repair_rows),
        "repaired_validated_unique_output": len(repaired_valid_keys),
        "original_rejected_repaired_successfully": original_rejected_repaired_successfully,
        "repaired_still_rejected": len(rejected),
        "stage5_unique_resolved_triples": original_resolved,
        "post_repair_pass_rate": post_repair_pass_rate,
        # Kept for backward compatibility with anything downstream reading
        # these exact keys — they are aliases of post_repair_pass_rate, NOT
        # a separate "true" rate. Prefer original_pass_rate /
        # post_repair_pass_rate above for anything new.
        "stage5_unique_resolution_rate": post_repair_pass_rate,
        "unique_output_pass_rate": post_repair_pass_rate,
        "repair_failure_reasons": dict(rejected_status_counts),
    }

    # Hard invariant: every unique input triple ends up in exactly one of
    # the two output files. If this ever trips, it fails loudly here
    # instead of only showing up as a mismatch against the summary later.
    assert len(validated) + len(rejected) == original_unique, (
        f"Triple accounting mismatch: {len(validated)} validated + "
        f"{len(rejected)} rejected != {original_unique} unique input triples"
    )

    return validated, rejected, summary


def validate_triples(raw_triples, ontology, min_confidence=0.22):
    constraints = ontology["relation_constraints"]

    triple_mention_count = defaultdict(int)
    subject_mention_count = defaultdict(int)
    triple_sentence_count = defaultdict(set)
    predicate_count = defaultdict(int)
    relation_pair_count = defaultdict(int)

    for subj, subj_t, rel, obj, obj_t, predicate, sentence, filename in raw_triples:
        triple_mention_count[(subj, rel, obj)] += 1
        subject_mention_count[subj] += 1
        triple_sentence_count[(subj, rel, obj)].add(sentence[:160])
        predicate_count[(subj, rel, obj, predicate.lower())] += 1
        relation_pair_count[(rel, subj_t, obj_t)] += 1

    seen = set()
    validated = []
    rejected = []

    # PATCH: track first-seen source document per unique triple so it can
    # ride along into the output record — gives you a direct file to check
    # against during manual spot-checks instead of hunting for it.
    first_seen_file = {}
    for subj, subj_t, rel, obj, obj_t, predicate, sentence, filename in raw_triples:
        first_seen_file.setdefault((subj, rel, obj), filename)

    for subj, subj_t, rel, obj, obj_t, predicate, sentence, filename in raw_triples:
        key = (subj, rel, obj)
        if key in seen:
            continue
        seen.add(key)

        rules = constraints.get(rel, [])
        matching_rules = [r for r in rules if r["domain"] == subj_t and r["range"] == obj_t]
        type_prior = max([r.get("confidence", 0.5) for r in matching_rules], default=0.45 if not rules else 0.0)
        domain_ok = relation_type_ok(rel, subj_t, obj_t)

        semantic_expected = SEMANTIC_OBJECT_PRIORS.get(rel)
        semantic_ok = semantic_expected is None or obj_t in semantic_expected

        mentions = triple_mention_count[key]
        sentence_support = len(triple_sentence_count[key])
        mention_support = 1.0 - math.exp(-mentions / 3)
        sentence_score = 1.0 - math.exp(-sentence_support / 2)
        subject_specificity = mentions / max(subject_mention_count[subj], 1)
        predicate_diversity = len([p for s, r, o, p in predicate_count if (s, r, o) == key])
        predicate_score = min(1.0, 0.55 + 0.15 * predicate_diversity)
        quality_score = harmonic_mean([mention_quality(subj), mention_quality(obj)])
        local_pair_prior = min(1.0, relation_pair_count[(rel, subj_t, obj_t)] / max(sum(c for (r, _, _), c in relation_pair_count.items() if r == rel), 1))

        confidence = (
            0.28 * mention_support +
            0.22 * sentence_score +
            0.18 * subject_specificity +
            0.14 * predicate_score +
            0.10 * quality_score +
            0.08 * max(type_prior, local_pair_prior)
        )
        confidence = round(min(confidence, 1.0), 3)

        record = {
            "subject": subj, "subject_type": subj_t,
            "relation": rel,
            "object": obj, "object_type": obj_t,
            "mentions": mentions,
            "sentence_support": sentence_support,
            "type_prior": round(type_prior, 3),
            "confidence": confidence
        }

        if not subj or not obj or subj.lower() == obj.lower():
            record["reject_reason"] = "empty/self triple"
            rejected.append(record)
        elif not domain_ok:
            expected = ", ".join(f"{r['domain']}->{r['range']}" for r in rules[:3])
            record["reject_reason"] = f"domain/range violation: {rel} expects one of [{expected}], got ({subj_t}->{obj_t})"
            rejected.append(record)
        elif not semantic_ok:
            record["reject_reason"] = (
                f"semantic type mismatch: {rel} expects object type in "
                f"{sorted(semantic_expected)}, got {obj_t} "
                f"(passed the induced ontology check, but that ontology was "
                f"itself built from Stage 5's entity-pairing bias)"
            )
            rejected.append(record)
        elif confidence < min_confidence:
            record["reject_reason"] = f"confidence {confidence} below threshold {min_confidence}"
            rejected.append(record)
        else:
            validated.append(record)

    return validated, rejected


def main():
    ontology, raw_triples = load_ontology_and_triples()
    # FIX: cap_fanout() was defined but never called, so the per-document
    # fan-out cap described in its docstring/comment (dropping pairing-noise
    # candidates for single-valued relations like EFFECTIVE_ON) had no
    # effect on stage5_unique_input_triples at all.
    raw_triples = cap_fanout(raw_triples)
    validated, rejected, repair_summary = validate_with_repairs(raw_triples, ontology)

    # Keep the two populations visibly separate everywhere from here on —
    # this is what makes the reported rate honest instead of a single
    # number that quietly blends repaired and unrepaired triples.
    validated_original = [v for v in validated if v["validation_status"] == "validated_original"]
    validated_after_repair = [v for v in validated if v["validation_status"] == "validated_after_repair"]

    print("\nStage 7 — Triple Validation")
    print(f"  Input triples (with duplicates): {len(raw_triples)}")
    print(f"  Unique triples: {len(validated) + len(rejected)}")
    print(f"  Validated (original, no repair): {len(validated_original)}")
    print(f"  Validated (after repair):        {len(validated_after_repair)}")
    print(f"  Rejected:                        {len(rejected)}")
    print(f"  Original pass rate (no repair):  {repair_summary['original_pass_rate']:.1f}%  <- report this as THE validation rate")
    print(f"  Post-repair pass rate:           {repair_summary['post_repair_pass_rate']:.1f}%  (includes {len(validated_after_repair)} repaired triples — lower-confidence, separately tagged)")
    print(f"  Stage 5 unique input triples: {repair_summary['stage5_unique_input_triples']}")

    for r in rejected[:5]:  # show only first 5 rejections
        print(f"    REJECTED ({r['subject'][:30]}, {r['relation']}, "
              f"{r['object'][:30]}) — {r['reject_reason']}")
    
    if len(rejected) > 5:
        print(f"    ... and {len(rejected)-5} more rejections")

    os.makedirs("outputs_ML", exist_ok=True)
    # Combined file kept for backward compatibility with anything already
    # reading validated_triples.json — every record still carries
    # validation_status so the two populations remain distinguishable here.
    with open("outputs_ML/validated_triples.json", "w") as f:
        json.dump(validated, f, indent=2)
    # New: the two populations as their own files, so a consumer can't
    # accidentally treat repaired triples as equal-confidence to originals
    # just by reading "validated_triples.json" and assuming it's one tier.
    with open("outputs_ML/validated_original.json", "w") as f:
        json.dump(validated_original, f, indent=2)
    with open("outputs_ML/validated_after_repair.json", "w") as f:
        json.dump(validated_after_repair, f, indent=2)
    with open("outputs_ML/rejected_triples.json", "w") as f:
        json.dump(rejected, f, indent=2)
    with open("outputs_ML/triple_repair_summary.json", "w") as f:
        json.dump(repair_summary, f, indent=2)

    print(f"\nSaved outputs_ML/validated_triples.json ({len(validated)} triples, combined)")
    print(f"Saved outputs_ML/validated_original.json ({len(validated_original)} triples)")
    print(f"Saved outputs_ML/validated_after_repair.json ({len(validated_after_repair)} triples)")
    print(f"Saved outputs_ML/rejected_triples.json ({len(rejected)} triples)")
    print("Saved outputs_ML/triple_repair_summary.json")

if __name__ == "__main__":
    main()