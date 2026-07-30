import json
from collections import Counter

with open("outputs_ML/rejected_triples.json", "r") as f:
    rejected = json.load(f)

with open("outputs_ML/validated_triples.json", "r") as f:
    validated = json.load(f)

# Break down rejections by reason category
reason_types = Counter()
for r in rejected:
    reason = r["reject_reason"]
    if reason.startswith("domain/range"):
        reason_types["domain/range violation (ontology check)"] += 1
    elif reason.startswith("semantic type mismatch"):
        reason_types["semantic type mismatch (new check)"] += 1
    elif reason.startswith("confidence"):
        reason_types["low confidence"] += 1
    else:
        reason_types["other"] += 1

print("Rejection reason breakdown:")
for reason, count in reason_types.most_common():
    print(f"  {reason}: {count}")

# Specifically check: did any GOVERNED_BY/LOCATED_IN/ORGANIZED_UNDER/EFFECTIVE_ON
# with PARTY as object make it through to validated (should be zero now)?
target_relations = {"GOVERNED_BY", "LOCATED_IN", "ORGANIZED_UNDER", "EFFECTIVE_ON"}
bad_validated = [
    t for t in validated
    if t["relation"] in target_relations
    and t["object_type"] not in ({"JURISDICTION"} if t["relation"] != "EFFECTIVE_ON" else {"DATE", "EFFECTIVE_DATE"})
]
print(f"\nTarget-relation triples with wrong object type that slipped into validated_triples.json: {len(bad_validated)}")
if bad_validated:
    for t in bad_validated[:5]:
        print(f"  ({t['subject']}, {t['relation']}, {t['object']}) type={t['object_type']}")

# Show a few examples of what the semantic check actually rejected
semantic_examples = [r for r in rejected if r["reject_reason"].startswith("semantic type mismatch")]
print(f"\nSample semantic-mismatch rejections (showing up to 5 of {len(semantic_examples)}):")
for r in semantic_examples[:5]:
    print(f"  ({r['subject'][:40]}, {r['relation']}, {r['object'][:40]}) type={r['object_type']}")