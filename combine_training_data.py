import json

print("Loading CUAD training data (sentence-level)...")
with open("outputs_ML/cuad_sentences.json", "r") as f:
    cuad_data = json.load(f)

print("Loading LLM training data...")
with open("outputs_ML/llm_training_data.json", "r") as f:
    llm_data = json.load(f)

def convert_llm_to_cuad_format(llm_examples):
    converted = []
    for idx, example in enumerate(llm_examples):
        sentence = example["sentence"]
        labels = example["labels"]

        if not labels or not isinstance(labels[0], dict):
            continue

        entities = []
        current_entity = None

        words_in_sent = sentence.split()
        char_positions = []
        pos = 0
        for w in words_in_sent:
            start = sentence.find(w, pos)
            if start == -1:
                start = pos
            char_positions.append((start, start + len(w)))
            pos = start + len(w)

        for token_idx, token in enumerate(labels):
            if not isinstance(token, dict):
                continue

            if "word" in token:
                word = token["word"]
                label = token.get("label", "O")
            else:
                word = list(token.keys())[0]
                label = token[word] if isinstance(token[word], str) else "O"

            if token_idx < len(char_positions):
                word_start, word_end = char_positions[token_idx]
            else:
                continue

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "start": word_start,
                    "end": word_end,
                    "label": label[2:],
                    "text": word
                }
            elif label.startswith("I-") and current_entity:
                current_entity["end"] = word_end
                current_entity["text"] += " " + word
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        if entities:
            converted.append({
                "text": sentence,
                "entities": entities
            })

    return converted

print("Converting LLM format...")
llm_converted = convert_llm_to_cuad_format(llm_data)

combined = cuad_data + llm_converted

print(f"\nCombined training data (all sentence-level):")
print(f"  CUAD sentences:  {len(cuad_data)}")
print(f"  LLM sentences:   {len(llm_converted)}")
print(f"  Total combined:  {len(combined)}")

with open("outputs_ML/combined_training_data.json", "w") as f:
    json.dump(combined, f, indent=2)

print(f"Saved to outputs_ML/combined_training_data.json")