import json
import random

with open("outputs_ML/crf_all_entities.json", "r") as f:
    results = json.load(f)

all_entities = [e for contract in results for e in contract["entities"]]
samples = random.sample(all_entities, 30)

print(f"Total entities: {len(all_entities)}")
print(f"\nRandom sample of 30 entities:")
print("=" * 50)
for e in samples:
    words = len(e['text'].split())
    print(f"{e['label']:15} ({words:2} words) → '{e['text']}'")