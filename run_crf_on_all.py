import json
import os
import pickle
import re

def predict_entities(crf, text):
    sentences = [s.strip() for s in re.split(r'[.\n]+', text) if s.strip()]
    all_entities = []
    
    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        
        from stage2_ner_crf import sentence_to_features
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
    print("Loading CRF model...")
    with open("outputs_ML/crf_model.pkl", 'rb') as f:
        crf = pickle.load(f)
    
    DATA_DIR = "data"
    OUTPUT_PATH = "outputs_ML/crf_all_entities.json"
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    print(f"Running CRF on {len(files)} contracts...")
    
    all_results = []
    
    for i, fname in enumerate(files):
        fpath = os.path.join(DATA_DIR, fname)
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        entities = predict_entities(crf, text)
        
        all_results.append({
            "filename": fname,
            "entities": entities
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(files)}...")
    
    os.makedirs("outputs_ML", exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    total_entities = sum(len(r["entities"]) for r in all_results)
    print(f"\nDone!")
    print(f"Total entities found: {total_entities}")
    print(f"Saved to {OUTPUT_PATH}")