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

nlp = spacy.load("en_core_web_sm")

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
    (['entered into', 'enter into', 'executes', 'execute'], 'ENTERED_INTO'),
    (['governed by', 'construed by', 'governed and construed'], 'GOVERNED_BY'),
    (['agrees to pay', 'agreed to pay', 'shall pay', 'will pay', 'payable to'], 'PAYABLE_BY'),
    (['grants to', 'granted to', 'grant to', 'license to', 'licenses to'], 'GRANTED_TO'),
    (['effective as of', 'effective date', 'dated as of', 
      'commencing on', 'dated as', 'dated'], 'EFFECTIVE_ON'),
    (['party to', 'parties to', 'is a party', 'are parties'], 'PARTY_OF'),
    (['appoints', 'appointed as', 'appoint'], 'APPOINTED_AS'),
    (['subject to', 'conditioned on', 'contingent on'], 'SUBJECT_TO'),
    (['limited to', 'restricted to', 'shall not exceed'], 'LIMITED_TO'),
    (['defined in', 'as defined', 'meaning set forth'], 'DEFINED_IN'),
    (['terminates', 'terminate', 'termination of'], 'TERMINATES'),
    (['owns', 'shall own', 'retains ownership'], 'OWNS'),
    (['is', 'are', 'was', 'were'], 'IS_A'),
    (['has', 'have', 'had'], 'HAS'),
    (['contains', 'contained in', 'included in'], 'CONTAINS'),
    (['requested', 'requested with'], 'REQUESTED_BY'),
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
                "label": entity["label"],
                "start_char": match.start(),
                "end_char": match.end()
            })
            break

    garbage_predicates = {
        'definitions', 'omitted', 'exhibit', 'outsourcing',
        'pages', 'making', 'identified', 'contracted',
        'definitions as', 'contained in by', 'requested with'
    }

    for sent in doc.sents:
        sent_start = sent.start_char
        sent_end = sent.end_char

        sent_entities = [
            e for e in entity_spans
            if sent_start <= e["start_char"] < sent_end
        ]

        if len(sent_entities) < 2:
            continue

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

                verb_phrase = extract_verb_phrase(sent, e1_end_token, e2_start_token)

                if not verb_phrase or len(verb_phrase.split()) > 6:
                    continue
                if verb_phrase.lower() in garbage_predicates:
                    continue
                if verb_phrase.isupper():
                    continue

                known_relation = normalize_predicate(verb_phrase)

                pairs.append({
                    "entity1": e1["text"],
                    "label1": e1["label"],
                    "entity2": e2["text"],
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

    entry_verbs = {'entered', 'enter', 'execute', 'executes', 'sign', 'signed'}
    has_entry_verb = 1.0 if pred_words & entry_verbs else 0.0

    gov_words = {'governed', 'construed', 'jurisdiction', 'laws'}
    has_gov_word = 1.0 if pred_words & gov_words else 0.0

    pay_words = {'pay', 'pays', 'paid', 'payment', 'payable'}
    has_pay_word = 1.0 if pred_words & pay_words else 0.0

    grant_words = {'grant', 'grants', 'granted', 'license', 'licenses'}
    has_grant_word = 1.0 if pred_words & grant_words else 0.0

    date_words = {'effective', 'dated', 'commencing', 'starting', 'beginning'}
    has_date_word = 1.0 if pred_words & date_words else 0.0

    sent_length = len(pair["sentence"].split())

    prepositions = {'by', 'to', 'of', 'into', 'as', 'on', 'in', 'for'}
    has_preposition = 1.0 if pred_words & prepositions else 0.0

    return [
        label1, label2, pred_length,
        has_entry_verb, has_gov_word, has_pay_word,
        has_grant_word, has_date_word, sent_length, has_preposition
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
    with open(ENTITIES_PATH, 'r') as f:
        all_entity_results = json.load(f)

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
        all_pairs.extend(pairs)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/510... ({len(all_pairs)} pairs found)")

    print(f"\nTotal entity pairs found: {len(all_pairs)}")

    print("\nDiscovering relations (OpenIE)...")
    all_pairs = discover_relations_openIE(all_pairs)

    print("\nTraining classifiers...")
    classifier_results, label_encoder, X_train, X_test = train_classifiers(all_pairs)

    os.makedirs("outputs_ML", exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_pairs[:5000], f, indent=2)

    print(f"\nDone!")
    print(f"Total pairs extracted: {len(all_pairs)}")
    print(f"Saved to {OUTPUT_PATH}")