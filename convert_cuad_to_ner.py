import json
import os

def convert_cuad_to_ner(cuad_path, output_path):
    """
    Convert CUAD JSON annotations to NER training format.
    
    CUAD format:
    {
        "data": [
            {
                "title": "contract name",
                "paragraphs": [
                    {
                        "context": "raw contract text",
                        "qas": [
                            {
                                "question": "What is the party name?",
                                "answers": [
                                    {
                                        "text": "Google LLC",
                                        "answer_start": 45
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    NER format we need:
    [
        {
            "text": "Google LLC signed the Agreement",
            "entities": [
                {"start": 0, "end": 10, "label": "PARTY"}
            ]
        }
    ]
    """
    
    print("Loading CUAD JSON...")
    with open(cuad_path, 'r', encoding='utf-8') as f:
        cuad = json.load(f)
    
    # CUAD question → entity type mapping
    # CUAD asks questions like "Who are the parties?" 
    # We map those to our entity types
    question_to_entity = {
        "parties": "PARTY",
        "party": "PARTY",
        "agreement date": "DATE",
        "effective date": "EFFECTIVE_DATE",
        "expiration date": "DATE",
        "governing law": "JURISDICTION",
        "payment": "PAYMENT",
        "notice": "NOTICE",
        "agreement": "AGREEMENT",
        "contract": "CONTRACT",
        "license": "LICENSE",
        "term": "TERM",
    }
    
    ner_data = []
    skipped = 0
    
    print("Converting annotations...")
    
    for contract in cuad["data"]:
        for para in contract["paragraphs"]:
            context = para["context"]  # raw contract text
            entities = []
            
            for qa in para["qas"]:
                question = qa["question"].lower()
                
                # Find matching entity type from question
                entity_label = None
                for keyword, label in question_to_entity.items():
                    if keyword in question:
                        entity_label = label
                        break
                
                if not entity_label:
                    continue
                
                # Get all answers for this question
                for answer in qa.get("answers", []):
                    text = answer["text"].strip()
                    start = answer["answer_start"]
                    end = start + len(text)
                    
                    if text:  # skip empty answers
                        entities.append({
                            "start": start,
                            "end": end,
                            "label": entity_label,
                            "text": text
                        })
            
            if entities:  # only keep paragraphs that have at least one entity
                ner_data.append({
                    "text": context,
                    "entities": entities
                })
            else:
                skipped += 1
    
    # Save NER training data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ner_data, f, indent=2)
    
    print(f"\nDone!")
    print(f"Training examples created: {len(ner_data)}")
    print(f"Paragraphs skipped (no entities): {skipped}")
    print(f"Saved to {output_path}")
    
    # Show sample
    print(f"\nSample training example:")
    if ner_data:
        sample = ner_data[0]
        print(f"Text: {sample['text'][:100]}...")
        print(f"Entities: {sample['entities'][:3]}")

if __name__ == "__main__":
    CUAD_PATH = r"C:\Users\keert\OneDrive\Desktop\IIITH_Research\CUAD_v1\CUAD_v1.json"
    OUTPUT_PATH = "outputs_ML/ner_training_data.json"
    
    convert_cuad_to_ner(CUAD_PATH, OUTPUT_PATH)