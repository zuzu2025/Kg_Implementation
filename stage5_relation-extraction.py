import json
import os
import re
import spacy
import numpy as np
from collections import Counter, defaultdict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from kg_no_llm_algorithms import mention_quality

nlp = spacy.load("en_core_web_sm")


def load_stage3_entities(raw_path, linked_path="outputs_ML/linked_entities_v2.json"):
    """
    Prefer Stage 3 linked/canonical mentions when available, but keep the
    original surface mention for text matching.
    """
    path = linked_path if os.path.exists(linked_path) else raw_path
    with open(path, 'r') as f:
        rows = json.load(f)

    if path == linked_path:
        normalized = []
        for contract in rows:
            entities = []
            for e in contract.get("entities", []):
                text = e.get("original_mention", e.get("canonical", ""))
                canonical = e.get("canonical", e.get("original_mention", ""))
                if mention_quality(text) == 0.0 or mention_quality(canonical) == 0.0:
                    continue
                entities.append({
                    "text": text,
                    "canonical": canonical,
                    "label": e.get("label", "OTHER"),
                    "link_score": e.get("score", 0.0),
                    "linked": e.get("linked", False),
                })
            normalized.append({
                "filename": contract["filename"],
                "entities": entities,
            })
        print(f"Loaded canonical Stage 3 entities from {path}")
        return normalized

    print(f"Loaded raw CRF entities from {path}")
    return [
        {
            "filename": contract["filename"],
            "entities": [
                e for e in contract.get("entities", [])
                if mention_quality(e.get("text", "")) > 0.0
            ],
        }
        for contract in rows
    ]

# ══════════════════════════════════════════════════════
# STEP 1 — OPENIE STYLE EXTRACTION
# ══════════════════════════════════════════════════════

def extract_verb_phrase(sent_doc, e1_end_token, e2_start_token):
    """
    Extract the main verb phrase between two entities.
    OpenIE approach — verb phrase = relation.
    """
    between_tokens = [token for token in sent_doc
                      if e1_end_token <= token.i < e2_start_token]

    if not between_tokens:
        return None

    # Find main verb
    main_verb = None
    for token in between_tokens:
        if token.pos_ in ('VERB', 'AUX'):
            main_verb = token
            break

    if not main_verb:
        between_text = " ".join(t.text for t in between_tokens)
        if len(between_tokens) <= 5:
            return between_text.strip()
        return None

    # Build verb phrase
    verb_phrase = [main_verb.text]
    for token in between_tokens:
        if token.dep_ in ('prep', 'part', 'advmod') and token.head == main_verb:
            verb_phrase.append(token.text)
        elif token.pos_ == 'AUX' and token.head == main_verb:
            verb_phrase.insert(0, token.text)

    return " ".join(verb_phrase).strip().lower()

def normalize_predicate(predicate):
    """
    Map surface predicate to canonical relation type.
    Returns OTHER if no match found.
    """
    pred = predicate.lower().strip()

    relation_map = [
    (['entered into', 'enter into', 'executes', 'execute',
      'execution', 'executed'], 'ENTERED_INTO'),
    (['governed by', 'construed by', 'governed and construed',
      'promulgated under', 'promulgated', 'pursuant to'], 'GOVERNED_BY'),
    (['agrees to pay', 'agreed to pay', 'shall pay', 'will pay', 'payable to'], 'PAYABLE_BY'),
    (['grants to', 'granted to', 'grant to', 'license to', 'licenses to'], 'GRANTED_TO'),
    (['effective as of', 'effective date', 'dated as of', 
      'commencing on', 'dated as', 'dated'], 'EFFECTIVE_ON'),
    (['party to', 'parties to', 'is a party', 'are parties',
      'by and between', 'between'], 'PARTY_OF'),
    (['appoints', 'appointed as', 'appoint'], 'APPOINTED_AS'),
    (['subject to', 'conditioned on', 'contingent on'], 'SUBJECT_TO'),
    (['limited to', 'restricted to', 'shall not exceed'], 'LIMITED_TO'),
    (['defined in', 'as defined', 'meaning set forth'], 'DEFINED_IN'),
    (['terminates', 'terminate', 'termination of'], 'TERMINATES'),
    (['owns', 'shall own', 'retains ownership'], 'OWNS'),
    (['located at', 'located in', 'headquartered at', 'headquartered in',
      'situated at', 'situated in'], 'LOCATED_IN'),
    (['organized under', 'organized in', 'incorporated under',
      'incorporated in'], 'ORGANIZED_UNDER'),
    (['referred to as', 'hereinafter referred to as', 'known as'], 'REFERRED_TO_AS'),
    (['is', 'are', 'was', 'were'], 'IS_A'),
    (['has', 'have', 'had', 'having', 'holding', 'holds'], 'HAS'),
    (['contains', 'contained in', 'included in'], 'CONTAINS'),
    (['requested', 'requested with', 'requesting'], 'REQUESTED_BY'),
    (['indicated by', 'indicated'], 'INDICATED_BY'),
    ]

    for keywords, relation_type in relation_map:
        for keyword in keywords:
            if keyword in pred:
                return relation_type

    return 'OTHER'

def extract_entity_pairs_openIE(text, entities):
    pairs = []
    doc = nlp(text[:10000])

    entity_spans = []
    for entity in entities:
        mention = entity["text"].lower()
        for match in re.finditer(re.escape(mention), text.lower()):
            entity_spans.append({
                "text": entity["text"],
                "canonical": entity.get("canonical", entity["text"]),
                "label": entity["label"],
                "start_char": match.start(),
                "end_char": match.end()
            })
            break

    garbage_predicates = {
        'definitions', 'omitted', 'exhibit', 'outsourcing',
        'pages', 'making', 'identified', 'contracted',
        'definitions as', 'contained in by', 'requested with',
        # bare connectors/stopwords with no relational content on their own
        'and', 'or', 'the', 'a', 'an', 'to', 'of', 'in', 'on', 'by', 'for'
    }

    def _is_junk_predicate(vp):
        """
        Reject predicate fragments that are mostly punctuation/digits
        (e.g. '" ) , and', 'address:3885', '( "') — these come from
        verb-phrase extraction picking up quote marks, parenthesis
        remnants, or address fragments near entity spans rather than an
        actual verb phrase, and shouldn't be treated as a relation.
        """
        stripped = vp.replace(' ', '')
        if not stripped:
            return True
        alpha_chars = sum(c.isalpha() for c in stripped)
        return (alpha_chars / len(stripped)) < 0.6

    for sent in doc.sents:
        sent_start = sent.start_char
        sent_end = sent.end_char

        sent_entities = [
            e for e in entity_spans
            if sent_start <= e["start_char"] < sent_end
        ]

        if len(sent_entities) < 2:
            continue

        # Cap how far apart two entities can be and still be paired. Real
        # relational statements ("X is a party to Y", "X dated Y") have
        # their two entities close together with the predicate directly
        # between them. Pairs from opposite ends of a long sentence are
        # almost never a real relation — just two entities that happen to
        # co-occur. This is the surgical fix for the fan-out problem: it
        # suppresses far-apart spurious pairs without discarding whole
        # sentences, which matters because legal prose is naturally
        # entity-dense (median sentence length ~50 words in this corpus) —
        # a blanket per-sentence entity-count skip was tried and reverted
        # here because it discarded legitimate pairs along with noise,
        # collapsing total extracted pairs by >95%.
        MAX_TOKEN_GAP = 25

        for i in range(len(sent_entities)):
            for j in range(i+1, len(sent_entities)):
                e1 = sent_entities[i]
                e2 = sent_entities[j]

                e1_end_token = None
                e2_start_token = None

                for token in sent:
                    if token.idx >= e1["end_char"] - sent_start and e1_end_token is None:
                        e1_end_token = token.i
                    if token.idx >= e2["start_char"] - sent_start and e2_start_token is None:
                        e2_start_token = token.i

                if e1_end_token is None or e2_start_token is None:
                    continue
                if e1_end_token >= e2_start_token:
                    continue
                if e2_start_token - e1_end_token > MAX_TOKEN_GAP:
                    continue

                verb_phrase = extract_verb_phrase(sent, e1_end_token, e2_start_token)

                if not verb_phrase or len(verb_phrase.split()) > 6:
                    continue
                if verb_phrase.lower() in garbage_predicates:
                    continue
                if _is_junk_predicate(verb_phrase):
                    continue
                if verb_phrase.isupper():
                    continue

                known_relation = normalize_predicate(verb_phrase)

                pairs.append({
                    "entity1": e1["text"],
                    "canonical1": e1.get("canonical", e1["text"]),
                    "label1": e1["label"],
                    "entity2": e2["text"],
                    "canonical2": e2.get("canonical", e2["text"]),
                    "label2": e2["label"],
                    "predicate": verb_phrase,
                    "relation": known_relation,
                    "sentence": sent.text
                })

    return pairs

def discover_relations_openIE(all_pairs):
    """
    Show discovered relations — both known and raw predicates.
    """
    known = Counter(p["relation"] for p in all_pairs if p["relation"] != "OTHER")
    other_predicates = Counter(
        p["predicate"] for p in all_pairs if p["relation"] == "OTHER"
    )

    print("\nKnown relation types:")
    for rel, count in sorted(known.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel}: {count}")

    print(f"\nTop 20 raw predicates (unmapped):")
    for pred, count in other_predicates.most_common(20):
        print(f"  '{pred}': {count}")

    print(f"\nTotal pairs with predicate: {len(all_pairs)}")
    print(f"Mapped to known relations:  {sum(known.values())}")
    print(f"Raw predicate (OTHER):      {sum(1 for p in all_pairs if p['relation'] == 'OTHER')}")

    return all_pairs

# ══════════════════════════════════════════════════════
# STEP 2 — FEATURE EXTRACTION
# ══════════════════════════════════════════════════════

def extract_relation_features(pair):
    """
    Features for the relation-type classifier.

    IMPORTANT: earlier versions of this function included has_entry_verb,
    has_gov_word, has_pay_word, has_grant_word, and has_date_word — keyword
    presence checks whose vocabulary directly overlapped normalize_predicate's
    keyword lists (the function that PRODUCES the labels these classifiers
    are trained to predict). That's circular: a classifier fed "does the
    predicate contain 'governed'/'construed'" as a feature doesn't need to
    learn anything about relation extraction to predict GOVERNED_BY — it
    just needs to learn to reproduce the keyword rule that already assigned
    that label. Those five features are removed here. What remains are
    features that are structurally independent of the labeling rule: the
    entity type pair, predicate/sentence length, and generic (non-relation-
    specific) preposition presence. This gives an honest measure of whether
    entity-type context and predicate shape carry signal beyond the literal
    keyword match — a fair test rather than a circular one.
    """
    pred = pair["predicate"].lower()
    pred_words = set(pred.split())

    type_map = {
        'PARTY': 0, 'CONTRACT': 1, 'DATE': 2,
        'JURISDICTION': 3, 'NOTICE': 4, 'EFFECTIVE_DATE': 5,
        'OTHER': 6
    }
    label1 = type_map.get(pair["label1"], 6)
    label2 = type_map.get(pair["label2"], 6)

    pred_length = len(pred.split())
    sent_length = len(pair["sentence"].split())

    prepositions = {'by', 'to', 'of', 'into', 'as', 'on', 'in', 'for'}
    has_preposition = 1.0 if pred_words & prepositions else 0.0

    return [
        label1, label2, pred_length, sent_length, has_preposition
    ]

# ══════════════════════════════════════════════════════
# STEP 3 — TRAIN CLASSIFIERS
# ══════════════════════════════════════════════════════

def train_classifiers(all_pairs):
    relation_counts = Counter(p.get("relation", "OTHER") for p in all_pairs)
    valid_relations = {r for r, c in relation_counts.items()
                       if c >= 10 and r != "OTHER"}

    print(f"Valid relation types (≥10 examples): {valid_relations}")

    filtered_pairs = [p for p in all_pairs
                      if p.get("relation", "OTHER") in valid_relations]

    if len(filtered_pairs) < 50:
        print("Not enough labeled pairs to train!")
        return None, None, None, None

    print(f"\nTraining on {len(filtered_pairs)} labeled pairs...")

    X = np.array([extract_relation_features(p) for p in filtered_pairs])
    y = [p["relation"] for p in filtered_pairs]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    results = {}

    print("\n1. Training SVM...")
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=1.0))
    ])
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print("SVM Results:")
    print(classification_report(y_test, y_pred_svm,
          target_names=le.classes_, zero_division=0))
    results['svm'] = svm

    print("\n2. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 min_samples_leaf=5, class_weight='balanced',
                                 random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("Random Forest Results:")
    print(classification_report(y_test, y_pred_rf,
          target_names=le.classes_, zero_division=0))
    results['rf'] = rf

    print("\n3. Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    print("Naive Bayes Results:")
    print(classification_report(y_test, y_pred_nb,
          target_names=le.classes_, zero_division=0))
    results['nb'] = nb

    return results, le, X_train, X_test

# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    DATA_DIR = "data"
    ENTITIES_PATH = "outputs_ML/crf_all_entities.json"
    OUTPUT_PATH = "outputs_ML/relations.json"

    print("Loading entities...")
    all_entity_results = load_stage3_entities(ENTITIES_PATH)

    print("Extracting entity pairs with OpenIE...")
    all_pairs = []

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]

    for i, result in enumerate(all_entity_results):
        fname = result["filename"]
        fpath = os.path.join(DATA_DIR, fname)

        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        entities = result["entities"]
        if len(entities) < 2:
            continue

        pairs = extract_entity_pairs_openIE(text, entities)

        # PATCH: stamp source filename on every pair. Stage 7's fan-out cap
        # (which keeps only the best-supported object per subject+relation
        # for single-valued relations like EFFECTIVE_ON) needs to know which
        # document each pair came from. Without it, the cap can only key on
        # (subject, relation) globally — and a generic canonical string like
        # "agreement" or a company name that legitimately shows up across
        # many different contracts in this 510-doc corpus gets treated as
        # one entity with dozens of relations, when really it's dozens of
        # different documents each contributing one true fact. This one
        # field is what lets stage7 tell those two cases apart.
        for p in pairs:
            p["filename"] = fname

        all_pairs.extend(pairs)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/510... ({len(all_pairs)} pairs found)")

    print(f"\nTotal entity pairs found: {len(all_pairs)}")

    print("\nDiscovering relations (OpenIE)...")
    all_pairs = discover_relations_openIE(all_pairs)

    print("\nTraining classifiers...")
    classifier_results, label_encoder, X_train, X_test = train_classifiers(all_pairs)

    os.makedirs("outputs_ML", exist_ok=True)
    tmp_output_path = OUTPUT_PATH + ".tmp"
    with open(tmp_output_path, 'w') as f:
        # Previously this saved only all_pairs[:5000] — 1.7% of the total,
        # and not a random 1.7%: since all_pairs is built by iterating
        # files in order, it was biased toward whichever contracts were
        # processed first. Stage 6's induce_domain_rules() reads this
        # entire file to count entity-type-pair frequencies per relation
        # type, so that truncation was silently starving ontology
        # induction of most of the extracted data. Saving everything here
        # fixes that at the source.
        json.dump(all_pairs, f, indent=2)
    os.replace(tmp_output_path, OUTPUT_PATH)

    print(f"\nDone!")
    print(f"Total pairs extracted: {len(all_pairs)}")
    print(f"Saved to {OUTPUT_PATH}")
