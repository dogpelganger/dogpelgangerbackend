import os
import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS

APP_NAME = "Dogpelgänger Backend (FAISS-only)"

R2_BASE = os.environ.get(
    "R2_BASE",
    "https://pub-4b9f9bd46442471da196ba4ed4966ab0.r2.dev"
)

EMB_PATH = os.environ.get("DOG_EMBEDDINGS_NPZ", "embeddings_dogs.npz")
EXPECTED_DIM = 1280

app = Flask(__name__)
CORS(app)

_index = None
_embeddings = None
_filenames = None


def load_index():
    global _index, _embeddings, _filenames

    d = np.load(EMB_PATH, allow_pickle=True)
    emb = d["embeddings"].astype("float32")
    names = d["filenames"]

    if emb.ndim != 2 or emb.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"Embedding dim {emb.shape} != expected (*, {EXPECTED_DIM})"
        )

    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(EXPECTED_DIM)
    index.add(emb)

    _index = index
    _embeddings = emb
    _filenames = names


load_index()


@app.get("/")
def root():
    return jsonify({"ok": True, "service": APP_NAME})


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": APP_NAME,
        "dogs": int(_embeddings.shape[0]),
        "dim": int(_embeddings.shape[1])
    })


@app.post("/match")
def match():
    payload = request.get_json(force=True, silent=True) or {}
    emb = payload.get("embedding")
    top_k = int(payload.get("top_k", 6))

    if emb is None:
        return jsonify({"error": "Missing embedding"}), 400

    vec = np.array(emb, dtype="float32").reshape(1, -1)
    if vec.shape[1] != EXPECTED_DIM:
        return jsonify({
            "error": f"Embedding dim {vec.shape[1]} != {EXPECTED_DIM}"
        }), 400

    faiss.normalize_L2(vec)
    scores, idx = _index.search(vec, top_k)

    results = []
    for i, s in zip(idx[0], scores[0]):
        fname = str(_filenames[i])
        results.append({
            "dog_image": fname,
            "dog_url": f"{R2_BASE}/{fname}",
            "score": float(s)
        })

    return jsonify({"ok": True, "top_matches": results})
