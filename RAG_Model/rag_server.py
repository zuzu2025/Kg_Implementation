"""
rag_server.py

Local web dashboard for the contract RAG engine (v4). Flask serves the
search UI. Retrieval (BM25 + embeddings + graph) runs entirely locally as
before; the ONE new thing is that /api/search now also asks Groq to
synthesize a direct answer from that retrieved evidence, via
RagEngine.answer() instead of RagEngine.query().

If GROQ_API_KEY isn't set, this still works exactly like the v2/v3
dashboard -- engine.answer() falls back to raw retrieval with no
synthesized answer, and the UI simply won't show an answer panel.

Run:
    pip install flask --break-system-packages   # if not already installed
    export GROQ_API_KEY="your-key-here"         # optional, enables synthesis
    python3 rag_server.py

Then open http://127.0.0.1:5050 in your browser.
"""

from flask import Flask, jsonify, request, render_template

from rag_engine_v4 import RagEngine

app = Flask(__name__)
print("Loading RAG engine (first run builds the index, may take a moment)...")
engine = RagEngine()
print("Ready.")


@app.route("/")
def index():
    return render_template("index.html", examples=engine.example_questions())


@app.route("/api/suggest")
def api_suggest():
    q = request.args.get("q", "")
    return jsonify(engine.suggest(q, limit=8))


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "")
    if not q.strip():
        return jsonify({"error": "empty query"}), 400
    result = engine.answer(q, top_k_sentences=6, max_kg_facts_per_entity=5)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)