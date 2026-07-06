import os
import re
import json
import spacy
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
    Clean raw contract text.
    Legal contracts have lots of noise:
    - Page numbers
    - Headers/footers
    - Extra whitespace
    - Special characters
    """
    # Remove page numbers like "Page 1 of 10" or just "1."
    text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove excessive whitespace and blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove special characters but keep legal punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\(\)\-\$\%\/\']', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def tokenize_sentences(text):
    """
    Split text into sentences using NLTK.
    Why NLTK here? Because spaCy struggles with
    legal text abbreviations like 'Inc.' 'Ltd.' 'Sec.'
    thinking they're end of sentence.
    """
    sentences = sent_tokenize(text)
    return sentences

def chunk_text(sentences, chunk_size=10, overlap=2):
    """
    Group sentences into overlapping chunks.
    
    Why overlapping? 
    If a relation spans two chunks:
    Chunk 1: "Google LLC (hereinafter the Company)..."
    Chunk 2: "...the Company agrees to pay Apple Inc."
    
    Without overlap → we lose that "Company = Google LLC" context
    With overlap → Chunk 2 starts 2 sentences before, so it carries context
    
    chunk_size = number of sentences per chunk
    overlap = number of sentences carried over from previous chunk
    """
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap  # step forward but keep overlap sentences
    return chunks

def process_all_contracts(data_dir, output_dir):
    """
    Process all .txt contracts:
    1. Clean text
    2. Tokenize into sentences
    3. Chunk with overlap
    4. Save as JSON
    """
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    print(f"Processing {len(files)} contracts...")
    
    all_stats = {
        "total_contracts": len(files),
        "total_sentences": 0,
        "total_chunks": 0,
        "contracts": []
    }
    
    for i, fname in enumerate(files):
        fpath = os.path.join(data_dir, fname)
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
        
        # Step 1 — Clean
        cleaned = clean_text(raw_text)
        
        # Step 2 — Tokenize into sentences
        sentences = tokenize_sentences(cleaned)
        
        # Step 3 — Chunk with overlap
        chunks = chunk_text(sentences, chunk_size=10, overlap=2)
        
        # Step 4 — Save processed contract
        output = {
            "filename": fname,
            "num_sentences": len(sentences),
            "num_chunks": len(chunks),
            "chunks": chunks
        }
        
        out_path = os.path.join(output_dir, fname.replace('.txt', '_processed.json'))
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        # Update stats
        all_stats["total_sentences"] += len(sentences)
        all_stats["total_chunks"] += len(chunks)
        all_stats["contracts"].append({
            "filename": fname,
            "sentences": len(sentences),
            "chunks": len(chunks)
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(files)} contracts...")
    
    # Save overall stats
    with open(os.path.join(output_dir, '_stats.json'), 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    return all_stats

if __name__ == "__main__":
    DATA_DIR = "data"
    OUTPUT_DIR = "outputs_ml/stage1_preprocessing"
    
    print("Starting preprocessing...")
    stats = process_all_contracts(DATA_DIR, OUTPUT_DIR)
    
    print(f"\nDone!")
    print(f"Total contracts processed: {stats['total_contracts']}")
    print(f"Total sentences extracted: {stats['total_sentences']}")
    print(f"Total chunks created:      {stats['total_chunks']}")
    print(f"Saved to {OUTPUT_DIR}/")