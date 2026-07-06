import json
import os
import spacy
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nlp = spacy.load("en_core_web_sm")

# ══════════════════════════════════════════════════════
# ALGORITHM 1 — HOBBS ALGORITHM
# ══════════════════════════════════════════════════════

def get_noun_phrases(doc):
    """
    Extract all noun phrases from a spaCy doc.
    These are candidate antecedents for pronouns.
    
    Example:
    "Google LLC signed the Agreement"
    → noun phrases: ["Google LLC", "the Agreement"]
    """
    noun_phrases = []
    for chunk in doc.noun_chunks:
        noun_phrases.append({
            "text": chunk.text,
            "start": chunk.start,
            "end": chunk.end,
            "root": chunk.root.text,
            "root_dep": chunk.root.dep_
        })
    return noun_phrases

def is_pronoun(token):
    """Check if a token is a pronoun that needs resolution"""
    pronouns = {'it', 'its', 'they', 'them', 'their', 'he', 'she', 
                'his', 'her', 'this', 'that', 'these', 'those'}
    return token.text.lower() in pronouns

def hobbs_resolve(pronoun_token, doc, noun_phrases):
    """
    Simplified Hobbs Algorithm for pronoun resolution.
    
    Original Hobbs (1978) traverses a full parse tree.
    Our simplified version:
    
    Step 1: Look at the same sentence first
            Find noun phrases to the LEFT of the pronoun
            Pick the closest one that agrees in number
    
    Step 2: If not found, look at PREVIOUS sentences
            Find the most recent noun phrase
            Pick the one with matching number
    
    Why left-to-right and backwards?
    Because antecedents almost always PRECEDE pronouns:
    "Google LLC signed... IT agreed..." → "it" refers to "Google LLC" (came before)
    """
    pronoun_text = pronoun_token.text.lower()
    pronoun_pos = pronoun_token.i  # position in doc
    
    # Determine if pronoun is singular or plural
    is_plural = pronoun_text in {'they', 'them', 'their', 'these', 'those'}
    
    # Step 1: Search same sentence (left of pronoun)
    same_sent_candidates = []
    for np in noun_phrases:
        if np["end"] <= pronoun_pos:  # noun phrase is before pronoun
            # Check number agreement
            np_text = np["text"].lower()
            np_is_plural = np_text.endswith('s') or 'parties' in np_text
            
            if is_plural == np_is_plural:
                same_sent_candidates.append(np)
    
    if same_sent_candidates:
        # Pick the CLOSEST noun phrase (rightmost before pronoun)
        return same_sent_candidates[-1]["text"]
    
    # Step 2: Search previous sentences
    previous_candidates = []
    for np in noun_phrases:
        if np["end"] < pronoun_pos:
            previous_candidates.append(np)
    
    if previous_candidates:
        # Pick most recent noun phrase
        return previous_candidates[-1]["text"]
    
    return None  # couldn't resolve

def run_hobbs(text):
    """
    Run Hobbs algorithm on a full text.
    Returns list of (pronoun, resolved_antecedent) pairs.
    """
    doc = nlp(text[:10000])  # limit to 10k chars for speed
    noun_phrases = get_noun_phrases(doc)
    
    resolutions = []
    
    for token in doc:
        if is_pronoun(token):
            antecedent = hobbs_resolve(token, doc, noun_phrases)
            if antecedent:
                resolutions.append({
                    "pronoun": token.text,
                    "position": token.i,
                    "resolved_to": antecedent,
                    "algorithm": "hobbs"
                })
    
    return resolutions

# ══════════════════════════════════════════════════════
# ALGORITHM 2 — MENTION PAIR SVM
# ══════════════════════════════════════════════════════

def extract_mention_features(mention1, mention2, doc):
    """
    Extract features for a mention pair (mention1, mention2).
    SVM uses these features to decide: do they refer to the same entity?
    
    Features:
    1. Distance between mentions (in tokens)
    2. Do they share any words?
    3. Is one a pronoun?
    4. Are they in the same sentence?
    5. String match (exact or partial)
    6. Both start with capital letter?
    7. One is substring of other?
    """
    text1 = mention1["text"].lower()
    text2 = mention2["text"].lower()
    
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    # Feature 1: distance in tokens
    distance = abs(mention1["start"] - mention2["start"])
    distance_norm = min(distance / 100, 1.0)  # normalize to 0-1
    
    # Feature 2: word overlap (Jaccard)
    if words1 | words2:
        jaccard = len(words1 & words2) / len(words1 | words2)
    else:
        jaccard = 0.0
    
    # Feature 3: is mention1 a pronoun?
    pronouns = {'it', 'its', 'they', 'them', 'their', 'he', 'she', 'this', 'that'}
    is_pronoun_1 = 1.0 if text1 in pronouns else 0.0
    is_pronoun_2 = 1.0 if text2 in pronouns else 0.0
    
    # Feature 4: same sentence
    # (simplified: if distance < 20 tokens, likely same sentence)
    same_sentence = 1.0 if distance < 20 else 0.0
    
    # Feature 5: exact string match
    exact_match = 1.0 if text1 == text2 else 0.0
    
    # Feature 6: both capitalized (proper nouns → likely entities)
    both_capitalized = 1.0 if (mention1["text"][0].isupper() and 
                                mention2["text"][0].isupper()) else 0.0
    
    # Feature 7: substring match
    substring_match = 1.0 if (text1 in text2 or text2 in text1) else 0.0
    
    # Feature 8: length difference
    len_diff = abs(len(text1) - len(text2)) / max(len(text1), len(text2), 1)
    
    return [
        distance_norm,
        jaccard,
        is_pronoun_1,
        is_pronoun_2,
        same_sentence,
        exact_match,
        both_capitalized,
        substring_match,
        len_diff
    ]

def generate_training_pairs(texts, max_texts=50):
    """
    Generate synthetic training pairs for SVM.
    
    Since we have no labeled coreference data,
    we use HEURISTIC labeling:
    
    POSITIVE pairs (coreferent = same entity):
    - Exact string matches
    - Pronoun + nearest preceding noun phrase
    - Substring matches (e.g. "Google" and "Google LLC")
    
    NEGATIVE pairs (not coreferent):
    - Mentions far apart with no word overlap
    - Different entity types (date vs company name)
    """
    positive_pairs = []
    negative_pairs = []
    
    for text in texts[:max_texts]:
        doc = nlp(text[:5000])
        noun_phrases = get_noun_phrases(doc)
        
        if len(noun_phrases) < 2:
            continue
        
        for i in range(len(noun_phrases)):
            for j in range(i+1, min(i+10, len(noun_phrases))):
                m1 = noun_phrases[i]
                m2 = noun_phrases[j]
                
                text1 = m1["text"].lower()
                text2 = m2["text"].lower()
                words1 = set(text1.split())
                words2 = set(text2.split())
                
                features = extract_mention_features(m1, m2, doc)
                
                # Heuristic positive: exact match or substring
                if text1 == text2 or text1 in text2 or text2 in text1:
                    positive_pairs.append(features)
                
                # Heuristic positive: high word overlap + close distance
                elif (len(words1 & words2) > 0 and 
                      abs(m1["start"] - m2["start"]) < 15):
                    positive_pairs.append(features)
                
                # Heuristic negative: no word overlap + far apart
                elif (len(words1 & words2) == 0 and 
                      abs(m1["start"] - m2["start"]) > 30):
                    negative_pairs.append(features)
    
    return positive_pairs, negative_pairs

def train_svm_coref(texts):
    """
    Train SVM mention-pair classifier.
    Input: list of contract texts
    Output: trained SVM pipeline
    """
    print("Generating training pairs...")
    positive_pairs, negative_pairs = generate_training_pairs(texts)
    
    print(f"Positive pairs: {len(positive_pairs)}")
    print(f"Negative pairs: {len(negative_pairs)}")
    
    if not positive_pairs or not negative_pairs:
        print("Not enough training data!")
        return None
    
    # Balance classes
    min_size = min(len(positive_pairs), len(negative_pairs))
    X = positive_pairs[:min_size] + negative_pairs[:min_size]
    y = [1] * min_size + [0] * min_size
    
    X = np.array(X)
    y = np.array(y)
    
    # Train SVM
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True))
    ])
    
    print("Training SVM...")
    svm_pipeline.fit(X_train, y_train)
    
    y_pred = svm_pipeline.predict(X_test)
    print("\nSVM Evaluation:")
    print(classification_report(y_test, y_pred, 
          target_names=['Not Coreferent', 'Coreferent']))
    
    return svm_pipeline

def run_svm_coref(text, svm_model):
    """
    Run SVM coreference resolution on a text.
    For each pronoun → find best coreferent noun phrase.
    """
    if svm_model is None:
        return []
    
    doc = nlp(text[:10000])
    noun_phrases = get_noun_phrases(doc)
    resolutions = []
    
    pronouns = {'it', 'its', 'they', 'them', 'their', 
                'he', 'she', 'this', 'that', 'these', 'those'}
    
    for token in doc:
        if token.text.lower() not in pronouns:
            continue
        
        pronoun_mention = {
            "text": token.text,
            "start": token.i,
            "end": token.i + 1
        }
        
        best_score = 0.0
        best_antecedent = None
        
        # Compare pronoun against all preceding noun phrases
        candidates = [np for np in noun_phrases if np["end"] <= token.i]
        candidates = candidates[-10:]  # only last 10 (closest)
        
        for candidate in candidates:
            features = extract_mention_features(pronoun_mention, candidate, doc)
            features_array = np.array(features).reshape(1, -1)
            
            # Get probability of being coreferent
            prob = svm_model.predict_proba(features_array)[0][1]
            
            if prob > best_score:
                best_score = prob
                best_antecedent = candidate["text"]
        
        if best_antecedent and best_score > 0.5:
            resolutions.append({
                "pronoun": token.text,
                "position": token.i,
                "resolved_to": best_antecedent,
                "confidence": round(best_score, 3),
                "algorithm": "svm"
            })
    
    return resolutions

# ══════════════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════════════

def compare_algorithms(text, svm_model):
    """
    Run both algorithms on same text and compare results.
    """
    print("\n" + "="*60)
    print("HOBBS ALGORITHM:")
    print("="*60)
    hobbs_results = run_hobbs(text)
    for r in hobbs_results[:10]:
        print(f"  '{r['pronoun']}' → '{r['resolved_to']}'")
    print(f"Total resolutions: {len(hobbs_results)}")
    
    print("\n" + "="*60)
    print("SVM MENTION-PAIR:")
    print("="*60)
    svm_results = run_svm_coref(text, svm_model)
    for r in svm_results[:10]:
        print(f"  '{r['pronoun']}' → '{r['resolved_to']}' (confidence: {r['confidence']})")
    print(f"Total resolutions: {len(svm_results)}")
    
    return hobbs_results, svm_results

# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    DATA_DIR = "data"
    OUTPUT_PATH = "outputs_ML/coreference_results.json"
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    
    # Load sample texts for training
    print("Loading contracts...")
    texts = []
    for fname in files[:100]:  # use 100 contracts for training
        with open(os.path.join(DATA_DIR, fname), 'r', 
                  encoding='utf-8', errors='ignore') as f:
            texts.append(f.read())
    
    # Train SVM
    print("\nTraining SVM coreference model...")
    svm_model = train_svm_coref(texts)
    
    # Test on sample
    sample_text = """
    Google LLC entered into this Agreement with Apple Inc. 
    It agreed to pay the sum of $5,000,000 to Apple Inc. 
    The Company shall deliver the software within 30 days.
    They will provide technical support for 2 years.
    Apple Inc accepted the terms. It signed on January 2021.
    """
    
    print("\nTesting on sample text:")
    print(f"Text: {sample_text}")
    hobbs_results, svm_results = compare_algorithms(sample_text, svm_model)
    
    # Run on all contracts
    print("\nRunning on all 510 contracts...")
    all_results = []
    
    for i, fname in enumerate(files):
        with open(os.path.join(DATA_DIR, fname), 'r',
                  encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        hobbs = run_hobbs(text)
        svm = run_svm_coref(text, svm_model)
        
        all_results.append({
            "filename": fname,
            "hobbs_resolutions": len(hobbs),
            "svm_resolutions": len(svm),
            "hobbs": hobbs[:20],  # save top 20
            "svm": svm[:20]
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/510...")
    
    os.makedirs("outputs_ML", exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    total_hobbs = sum(r["hobbs_resolutions"] for r in all_results)
    total_svm = sum(r["svm_resolutions"] for r in all_results)
    
    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON:")
    print(f"{'='*60}")
    print(f"Hobbs Algorithm:  {total_hobbs} total resolutions")
    print(f"SVM Mention-Pair: {total_svm} total resolutions")
    print(f"\nSaved to {OUTPUT_PATH}")