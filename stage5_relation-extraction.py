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
# STEP 1 — EXTRACT ENTITY PAIRS WITH CONTEXT
# ══════════════════════════════════════════════════════

def extract_entity_pairs(text, entities):
    """
    Find sentences that contain TWO entities.
    Extract the words between them as the predicate.
    
    Example:
    Text: "Google LLC entered into this Agreement on January 2021"
    Entities: ["Google LLC" → PARTY, "Agreement" → CONTRACT, "January 2021" → DATE]
    
    Pairs found:
    (Google LLC, Agreement) → predicate: "entered into this"
    (Google LLC, January 2021) → predicate: "entered into this Agreement on"
    (Agreement, January 2021) → predicate: "on"
    """
    pairs = []
    doc = nlp(text[:10000])
    
    # Find entity positions in text
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
            break  # first occurrence only
    
    # Find pairs within same sentence
    for sent in doc.sents:
        sent_start = sent.start_char
        sent_end = sent.end_char
        sent_text = sent.text
        
        # Find entities in this sentence
        sent_entities = [
            e for e in entity_spans
            if sent_start <= e["start_char"] < sent_end
        ]
        
        if len(sent_entities) < 2:
            continue
        
        # Generate all pairs
        for i in range(len(sent_entities)):
            for j in range(i+1, len(sent_entities)):
                e1 = sent_entities[i]
                e2 = sent_entities[j]
                
                # Extract words between the two entities
                e1_end = e1["end_char"] - sent_start
                e2_start = e2["start_char"] - sent_start
                
                if e1_end < e2_start:
                    between_text = sent_text[e1_end:e2_start].strip()
                else:
                    between_text = sent_text[e2["end_char"]-sent_start:e1["start_char"]-sent_start].strip()
                
                # Clean predicate
                between_text = re.sub(r'\s+', ' ', between_text).strip()
                
                if between_text and len(between_text.split()) <= 8:
                    pairs.append({
                        "entity1": e1["text"],
                        "label1": e1["label"],
                        "entity2": e2["text"],
                        "label2": e2["label"],
                        "predicate": between_text,
                        "sentence": sent_text
                    })
    
    return pairs

# ══════════════════════════════════════════════════════
# STEP 2 — DISCOVER RELATION TYPES FROM DATA
# ══════════════════════════════════════════════════════

def normalize_predicate(predicate):
    """
    Normalize predicate text to a relation type.
    
    We map surface predicates to canonical relation types
    based on what actually appears in the contracts.
    
    Example:
    "entered into"    → ENTERED_INTO
    "governed by"     → GOVERNED_BY  
    "agrees to pay"   → PAYABLE_BY
    "grants to"       → GRANTED_TO
    "effective as of" → EFFECTIVE_ON
    """
    pred = predicate.lower().strip()
    
    # Define mapping from surface text → relation type
    relation_map = [
        (['entered into', 'enter into', 'executes', 'execute'], 'ENTERED_INTO'),
        (['governed by', 'construed by', 'governed and construed'], 'GOVERNED_BY'),
        (['agrees to pay', 'agreed to pay', 'shall pay', 'will pay', 'payable to'], 'PAYABLE_BY'),
        (['grants to', 'granted to', 'grant to', 'license to', 'licenses to'], 'GRANTED_TO'),
        (['effective as of', 'effective date', 'dated as of', 'commencing on'], 'EFFECTIVE_ON'),
        (['party to', 'parties to', 'is a party', 'are parties'], 'PARTY_OF'),
        (['appoints', 'appointed as', 'appoint'], 'APPOINTED_AS'),
        (['subject to', 'conditioned on', 'contingent on'], 'SUBJECT_TO'),
        (['limited to', 'restricted to', 'shall not exceed'], 'LIMITED_TO'),
        (['defined in', 'as defined', 'meaning set forth'], 'DEFINED_IN'),
        (['terminates', 'terminate', 'termination of'], 'TERMINATES'),
        (['owns', 'shall own', 'retains ownership'], 'OWNS'),
    ]
    
    for keywords, relation_type in relation_map:
        for keyword in keywords:
            if keyword in pred:
                return relation_type
    
    return 'OTHER'

def discover_relations(all_pairs):
    """
    Count how often each relation type appears.
    Filter out OTHER and rare relations.
    """
    relation_counts = defaultdict(int)
    
    for pair in all_pairs:
        relation = normalize_predicate(pair["predicate"])
        pair["relation"] = relation
        relation_counts[relation] += 1
    
    print("\nDiscovered relation types:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel}: {count} instances")
    
    return all_pairs, relation_counts

# ══════════════════════════════════════════════════════
# STEP 3 — FEATURE EXTRACTION FOR CLASSIFIERS
# ══════════════════════════════════════════════════════

def extract_relation_features(pair):
    """
    Extract numerical features from an entity pair.
    These features are used by SVM, Random Forest, and Naive Bayes.
    
    Features:
    1. Entity type combination (PARTY+CONTRACT, PARTY+DATE etc)
    2. Predicate length (number of words)
    3. Predicate contains specific verbs
    4. Sentence length
    5. Entity1 position relative to Entity2
    6. Predicate contains legal keywords
    """
    pred = pair["predicate"].lower()
    pred_words = set(pred.split())
    
    # Feature 1: entity type pair encoded as numbers
    type_map = {
        'PARTY': 0, 'CONTRACT': 1, 'DATE': 2,
        'JURISDICTION': 3, 'NOTICE': 4, 'EFFECTIVE_DATE': 5,
        'OTHER': 6
    }
    label1 = type_map.get(pair["label1"], 6)
    label2 = type_map.get(pair["label2"], 6)
    
    # Feature 2: predicate length
    pred_length = len(pred.split())
    
    # Feature 3: contains entry/execution verbs
    entry_verbs = {'entered', 'enter', 'execute', 'executes', 'sign', 'signed'}
    has_entry_verb = 1.0 if pred_words & entry_verbs else 0.0
    
    # Feature 4: contains governance words
    gov_words = {'governed', 'construed', 'jurisdiction', 'laws'}
    has_gov_word = 1.0 if pred_words & gov_words else 0.0
    
    # Feature 5: contains payment words
    pay_words = {'pay', 'pays', 'paid', 'payment', 'payable'}
    has_pay_word = 1.0 if pred_words & pay_words else 0.0
    
    # Feature 6: contains grant words
    grant_words = {'grant', 'grants', 'granted', 'license', 'licenses'}
    has_grant_word = 1.0 if pred_words & grant_words else 0.0
    
    # Feature 7: contains date words
    date_words = {'effective', 'dated', 'commencing', 'starting', 'beginning'}
    has_date_word = 1.0 if pred_words & date_words else 0.0
    
    # Feature 8: sentence length
    sent_length = len(pair["sentence"].split())
    
    # Feature 9: predicate contains preposition
    prepositions = {'by', 'to', 'of', 'into', 'as', 'on', 'in', 'for'}
    has_preposition = 1.0 if pred_words & prepositions else 0.0
    
    return [
        label1, label2,
        pred_length,
        has_entry_verb,
        has_gov_word,
        has_pay_word,
        has_grant_word,
        has_date_word,
        sent_length,
        has_preposition
    ]

# ══════════════════════════════════════════════════════
# STEP 4 — TRAIN ALL THREE CLASSIFIERS
# ══════════════════════════════════════════════════════

def train_classifiers(all_pairs):
    """
    Train SVM, Random Forest, and Naive Bayes on relation pairs.
    Compare their performance.
    """
    # Count relations first
    from collections import Counter
    relation_counts = Counter(p.get("relation", "OTHER") for p in all_pairs)

# Keep only relations with at least 10 examples AND not OTHER
    valid_relations = {r for r, c in relation_counts.items() 
                  if c >= 10 and r != "OTHER"}

    print(f"Valid relation types (≥10 examples): {valid_relations}")

    filtered_pairs = [p for p in all_pairs 
                  if p.get("relation", "OTHER") in valid_relations]
    
    if len(filtered_pairs) < 50:
        print("Not enough labeled pairs to train!")
        return None, None, None, None
    
    print(f"\nTraining on {len(filtered_pairs)} labeled pairs...")
    
    # Extract features and labels
    X = [extract_relation_features(p) for p in filtered_pairs]
    y = [p["relation"] for p in filtered_pairs]
    
    X = np.array(X)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    results = {}
    
    # ── SVM ──
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
    
    # ── RANDOM FOREST ──
    print("\n2. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("Random Forest Results:")
    print(classification_report(y_test, y_pred_rf,
          target_names=le.classes_, zero_division=0))
    results['rf'] = rf
    
    # ── NAIVE BAYES ──
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
    
    # Load entities
    print("Loading entities...")
    with open(ENTITIES_PATH, 'r') as f:
        all_entity_results = json.load(f)
    
    # Extract pairs from all contracts
    print("Extracting entity pairs...")
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
        
        pairs = extract_entity_pairs(text, entities)
        all_pairs.extend(pairs)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/510... ({len(all_pairs)} pairs found)")
    
    print(f"\nTotal entity pairs found: {len(all_pairs)}")
    
    # Discover relations
    print("\nDiscovering relation types...")
    all_pairs, relation_counts = discover_relations(all_pairs)
    
    # Train classifiers
    print("\nTraining classifiers...")
    classifier_results, label_encoder, X_train, X_test = train_classifiers(all_pairs)
    
    # Save results
    os.makedirs("outputs_ML", exist_ok=True)
    
    # Save pairs with relations
    saveable_pairs = all_pairs[:5000]  # save top 5000
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(saveable_pairs, f, indent=2)
    
    print(f"\nDone!")
    print(f"Total pairs extracted: {len(all_pairs)}")
    print(f"Saved to {OUTPUT_PATH}")