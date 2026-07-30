"""
rag_server.py

Local web dashboard for the contract RAG engine. No API calls, no LLM --
Flask just serves a search UI and exposes the same RagEngine you already
have running locally.

This server is separate from the stage 6 vs stage 7 evaluation harness.

Run:
    pip install flask --break-system-packages   # if not already installed
    python3 rag_server.py

Then open http://127.0.0.1:5050 in your browser.
"""

from flask import Flask, jsonify, request, render_template

from rag_engine_v3 import RagEngine

app = Flask(__name__)
print("Loading RAG engine (first run builds the index, may take a moment)...")
engine = RagEngine()
print("Ready.")


@app.route("/")
def index():
    return render_template("index_v2.html", examples=engine.example_questions())


@app.route("/api/suggest")
def api_suggest():
    q = request.args.get("q", "")
    return jsonify(engine.suggest(q, limit=8))


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "")
    if not q.strip():
        return jsonify({"error": "empty query"}), 400
    result = engine.query(q, top_k_sentences=6, max_kg_facts_per_entity=5)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)