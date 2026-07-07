import json
import os
import re
import pickle
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

# Legal gazetteers — known legal terms
LEGAL_GAZETTEERS = {
    'company_suffixes': {
        'llc', 'inc', 'corp', 'ltd', 'plc', 'gmbh', 'ag', 'sa',
        'corporation', 'incorporated', 'limited', 'company', 'co',
        'group', 'holdings', 'partners', 'associates', 'ventures'
    },
    'jurisdictions': {
        'delaware', 'california', 'york', 'texas', 'florida',
        'illinois', 'nevada', 'england', 'wales', 'ontario',
        'germany', 'france', 'china', 'india', 'israel', 'pennsylvania',
        'kansas', 'massachusetts', 'virginia', 'georgia', 'ohio'
    },
    'months': {
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'
    },
    'legal_terms': {
        'agreement', 'contract', 'license', 'arrangement', 'deed',
        'indenture', 'amendment', 'addendum', 'exhibit', 'schedule',
        'annex', 'attachment', 'appendix'
    },
    'notice_terms': {
        'notice', 'notification', 'written notice', 'prior notice',
        'advance notice', 'formal notice'
    },
    'date_indicators': {
        'dated', 'effective', 'executed', 'entered', 'signed',
        'commencing', 'expiring', 'terminating'
    }
}

def word_features(sentence, i):
    """
    Enhanced feature extraction with:
    1. Original features (word shape, prefix, suffix)
    2. Gazetteer features (known legal terms)
    3. Dependency features (grammatical role)
    4. Context window features (±2 words)
    """
    word = sentence[i]
    word_lower = word.lower()

    features = {
        # ── ORIGINAL FEATURES ──
        'word.lower': word_lower,
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
        'word.prefix2': word[:2].lower(),
        'word.prefix3': word[:3].lower(),
        'word.suffix2': word[-2:].lower(),
        'word.suffix3': word[-3:].lower(),
        'word.has_hyphen': '-' in word,
        'word.has_digit': any(c.isdigit() for c in word),

        # ── GAZETTEER FEATURES ──
        'word.is_company_suffix': word_lower in LEGAL_GAZETTEERS['company_suffixes'],
        'word.is_jurisdiction': word_lower in LEGAL_GAZETTEERS['jurisdictions'],
        'word.is_month': word_lower in LEGAL_GAZETTEERS['months'],
        'word.is_legal_term': word_lower in LEGAL_GAZETTEERS['legal_terms'],
        'word.is_notice_term': word_lower in LEGAL_GAZETTEERS['notice_terms'],
        'word.is_date_indicator': word_lower in LEGAL_GAZETTEERS['date_indicators'],

        # ── SHAPE FEATURES ──
        'word.is_all_caps': word.isupper() and len(word) > 1,
        'word.has_currency': word.startswith('$') or word_lower in {'usd', 'dollars'},
        'word.is_section': word_lower in {'section', 'article', 'clause', 'exhibit'},
        'word.length': min(len(word), 20),  # capped at 20
    }

    # ── PREVIOUS WORD FEATURES (context -1) ──
    if i > 0:
        prev = sentence[i-1]
        features.update({
            'prev_word.lower': prev.lower(),
            'prev_word.istitle': prev.istitle(),
            'prev_word.isupper': prev.isupper(),
            'prev_word.is_company_suffix': prev.lower() in LEGAL_GAZETTEERS['company_suffixes'],
            'prev_word.is_month': prev.lower() in LEGAL_GAZETTEERS['months'],
            'prev_word.is_date_indicator': prev.lower() in LEGAL_GAZETTEERS['date_indicators'],
        })
    else:
        features['BOS'] = True

    # ── NEXT WORD FEATURES (context +1) ──
    if i < len(sentence) - 1:
        next_w = sentence[i+1]
        features.update({
            'next_word.lower': next_w.lower(),
            'next_word.istitle': next_w.istitle(),
            'next_word.isupper': next_w.isupper(),
            'next_word.is_company_suffix': next_w.lower() in LEGAL_GAZETTEERS['company_suffixes'],
            'next_word.is_month': next_w.lower() in LEGAL_GAZETTEERS['months'],
            'next_word.is_legal_term': next_w.lower() in LEGAL_GAZETTEERS['legal_terms'],
        })
    else:
        features['EOS'] = True

    # ── CONTEXT WINDOW -2 ──
    if i > 1:
        prev2 = sentence[i-2]
        features.update({
            'prev2_word.lower': prev2.lower(),
            'prev2_word.istitle': prev2.istitle(),
            'prev2_word.is_company_suffix': prev2.lower() in LEGAL_GAZETTEERS['company_suffixes'],
        })

    # ── CONTEXT WINDOW +2 ──
    if i < len(sentence) - 2:
        next2 = sentence[i+2]
        features.update({
            'next2_word.lower': next2.lower(),
            'next2_word.istitle': next2.istitle(),
            'next2_word.is_legal_term': next2.lower() in LEGAL_GAZETTEERS['legal_terms'],
        })

    return features

def sentence_to_features(sentence):
    return [word_features(sentence, i) for i in range(len(sentence))]

def sentence_to_labels(sentence, entities, text):
    words_with_pos = []
    for match in re.finditer(r'\S+', text):
        words_with_pos.append((match.group(), match.start(), match.end()))
    
    labels = ['O'] * len(words_with_pos)
    
    for entity in entities:
        ent_start = entity['start']
        ent_end = entity['end']
        ent_label = entity['label']
        first = True
        for idx, (word, w_start, w_end) in enumerate(words_with_pos):
            if w_start >= ent_start and w_end <= ent_end:
                if first:
                    labels[idx] = f'B-{ent_label}'
                    first = False
                else:
                    labels[idx] = f'I-{ent_label}'
    
    words = [w for w, _, _ in words_with_pos]
    return words, labels

def prepare_training_data(ner_data_path):
    with open(ner_data_path, 'r', encoding='utf-8') as f:
        ner_data = json.load(f)
    
    X = []
    y = []
    
    print(f"Preparing {len(ner_data)} training examples...")
    
    for example in ner_data:
        text = example['text']
        entities = example['entities']
        
        # Split text into sentences BUT track their position in full text
        sentences_with_pos = []
        for match in re.finditer(r'[^\.\n]+[\.\n]?', text):
            sent = match.group().strip()
            if sent:
                sentences_with_pos.append((sent, match.start(), match.end()))
        
        for sent, sent_start, sent_end in sentences_with_pos:
            words_with_pos = []
            for match in re.finditer(r'\S+', sent):
                # Position relative to FULL TEXT, not sentence
                abs_start = sent_start + match.start()
                abs_end = sent_start + match.end()
                words_with_pos.append((match.group(), abs_start, abs_end))
            
            if not words_with_pos:
                continue
            
            # Assign BIO labels using FULL TEXT positions
            labels = ['O'] * len(words_with_pos)
            
            for entity in entities:
                ent_start = entity['start']
                ent_end = entity['end']
                ent_label = entity['label']
                first = True
                
                for idx, (word, w_start, w_end) in enumerate(words_with_pos):
                    if w_start >= ent_start and w_end <= ent_end:
                        if first:
                            labels[idx] = f'B-{ent_label}'
                            first = False
                        else:
                            labels[idx] = f'I-{ent_label}'
            
            words = [w for w, _, _ in words_with_pos]
            
            # Only keep sentences with at least one entity
            if any(l != 'O' for l in labels):
                features = sentence_to_features(words)
                X.append(features)
                y.append(labels)
    
    return X, y

def train_crf(X_train, y_train):
    print("Training CRF model...")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.05,           # reduced regularization
        c2=0.05,           # reduced regularization
        max_iterations=300, # more iterations
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    return crf

def evaluate_crf(crf, X_test, y_test):
    y_pred = crf.predict(X_test)
    labels = list(crf.classes_)
    labels = [l for l in labels if l != 'O']
    print("\nCRF Evaluation Results:")
    print("=" * 50)
    report = metrics.flat_classification_report(
        y_test, y_pred, labels=labels, digits=3
    )
    print(report)
    return y_pred

def predict_entities(crf, text):
    sentences = [s.strip() for s in re.split(r'[.\n]+', text) if s.strip()]
    all_entities = []
    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        features = sentence_to_features(words)
        labels = crf.predict([features])[0]
        current_entity = None
        for word, label in zip(words, labels):
            if label.startswith('B-'):
                if current_entity:
                    all_entities.append(current_entity)
                current_entity = {'text': word, 'label': label[2:]}
            elif label.startswith('I-') and current_entity:
                current_entity['text'] += ' ' + word
            else:
                if current_entity:
                    all_entities.append(current_entity)
                    current_entity = None
        if current_entity:
            all_entities.append(current_entity)
    return all_entities

if __name__ == "__main__":
    NER_DATA_PATH = "outputs_ML/ner_training_data.json"
    MODEL_PATH = "outputs_ML/crf_model.pkl"
    
    X, y = prepare_training_data(NER_DATA_PATH)
    print(f"Total sentences prepared: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} sentences")
    print(f"Test:  {len(X_test)} sentences")
    
    crf = train_crf(X_train, y_train)
    evaluate_crf(crf, X_test, y_test)
    
    os.makedirs("outputs_ML", exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(crf, f)
    print(f"\nModel saved to {MODEL_PATH}")
    
    sample = "Google LLC entered into this Agreement with Apple Inc on January 2021 governed by Delaware law."
    print(f"\nSample prediction:")
    print(f"Text: {sample}")
    entities = predict_entities(crf, sample)
    print(f"Entities found: {entities}")