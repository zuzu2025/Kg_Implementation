import json
import os
import re
import sys
import time
from google import genai

# ── Configure Gemini ──────────────────────────────────────────────
client = genai.Client(api_key="AQ.Ab8RN6LYF80FvAEZ6_tX-czzy1su71YnapSjO9rNxbTLhzdurQ")

TAG_INSTRUCTIONS = """Label each word in each sentence with one of these IOB tags:
- B-PARTY: Beginning of a party/company name
- I-PARTY: Inside a party/company name
- B-CONTRACT: Beginning of a contract/agreement name
- I-CONTRACT: Inside a contract/agreement name
- B-DATE: Beginning of a date
- I-DATE: Inside a date
- B-JURISDICTION: Beginning of a jurisdiction/location
- I-JURISDICTION: Inside a jurisdiction/location
- B-EFFECTIVE_DATE: Beginning of an effective date
- I-EFFECTIVE_DATE: Inside an effective date
- B-NOTICE: Beginning of a notice clause
- I-NOTICE: Inside a notice clause
- O: Not an entity"""


class QuotaExhaustedError(Exception):
    pass


def get_llm_labels_batch(sentences, max_retries=3):
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

    prompt = f"""You are a Named Entity Recognition expert for legal contracts.

{TAG_INSTRUCTIONS}

Below are {len(sentences)} sentences, numbered. Label each one.

{numbered}

Return ONLY a JSON array with {len(sentences)} elements (one per sentence, in order).
Each element is itself a JSON array of {{"word": "...", "label": "..."}} objects
for that sentence's words. No explanation, no markdown, just the JSON array of arrays."""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt
            )
            raw = response.text.strip()

            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            return json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"    JSON parse error (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(1)
        except Exception as e:
            err_str = str(e).lower()
            if "quota" in err_str or "429" in err_str or "resource_exhausted" in err_str:
                print(f"    Quota/rate error: {e}")
                if attempt < max_retries - 1:
                    print(f"    Waiting 30s before retry ({attempt+1}/{max_retries})...")
                    time.sleep(30)
                else:
                    raise QuotaExhaustedError(str(e))
            else:
                print(f"    Error: {e}")
                time.sleep(2)

    return None


def generate_training_data(data_dir, output_path, num_contracts=500,
                            sentences_per_contract=20, batch_size=20, resume=True):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")])[:num_contracts]

    training_data = []
    done_files = set()

    if resume and os.path.exists(output_path):
        with open(output_path, "r") as f:
            training_data = json.load(f)
        done_files = {item["source_file"] for item in training_data if "source_file" in item}
        print(f"Resuming — {len(training_data)} sentences already labeled "
              f"across {len(done_files)} documents done.")

    total_sentences = len(training_data)
    failed = 0

    print(f"Target: {len(files)} contracts x {sentences_per_contract} sentences "
          f"= ~{len(files) * sentences_per_contract} sentences\n")

    try:
        for i, fname in enumerate(files):
            if fname in done_files:
                continue

            fpath = os.path.join(data_dir, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            sentences = [s.strip() for s in re.split(r"[.\n]+", text) if s.strip()]
            good_sentences = [
                s for s in sentences
                if 5 <= len(s.split()) <= 30 and any(c.isupper() for c in s)
            ][:sentences_per_contract]

            if not good_sentences:
                continue

            print(f"Doc {i+1}/{len(files)}: {fname[:40]} ({len(good_sentences)} sentences)")

            for start in range(0, len(good_sentences), batch_size):
                batch = good_sentences[start:start + batch_size]
                labeled_batch = get_llm_labels_batch(batch)

                if labeled_batch is None:
                    failed += len(batch)
                    continue

                for sent, labeled in zip(batch, labeled_batch):
                    words = sent.split()
                    if labeled and len(labeled) == len(words):
                        training_data.append({
                            "sentence": sent,
                            "labels": labeled,
                            "source_file": fname
                        })
                        total_sentences += 1
                    else:
                        failed += 1

                time.sleep(0.5)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(training_data, f, indent=2)

            if (i + 1) % 25 == 0:
                print(f"  --- Progress: {total_sentences} sentences, "
                      f"{i+1}/{len(files)} docs done ---")

    except QuotaExhaustedError as e:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(training_data, f, indent=2)
        print(f"\n⚠️  Quota exhausted: {e}")
        print(f"Progress saved: {total_sentences} sentences labeled.")
        print("Re-run tomorrow — it will resume automatically.")
        sys.exit(0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"\nDone! Total: {total_sentences} sentences, Failed: {failed}")
    return training_data


if __name__ == "__main__":
    DATA_DIR = "data"
    OUTPUT_PATH = "outputs_ML/llm_training_data.json"

    generate_training_data(
        data_dir=DATA_DIR,
        output_path=OUTPUT_PATH,
        num_contracts=500,
        sentences_per_contract=20,
        batch_size=20,
        resume=True
    )