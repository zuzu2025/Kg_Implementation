import json

with open("outputs_ML/ontology.json", "r") as f:
    ontology = json.load(f)

print("Induced Domain Rules:")
print("=" * 50)
for rule in ontology["domain_rules"]:
    print(f"{rule['subject']} --[{rule['relation']}]--> {rule['object']} (frequency: {rule['frequency']})")