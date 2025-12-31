cat > app.py <<'PY'
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

import faiss  # faiss-cpu

APP_NAME = "Dogpelgänger Backend (FAISS-only)"
R2_BASE = os.environ.get("R2_BASE", "https://pub-4b9f9bd46442471da196ba4ed4966ab0.r2.dev")
EMB_PATH = os.environ.get("DOG_EMBEDDINGS_NPZ", "embeddings_dogs.npz")
EXPECTED_DIM = int(os.environ.get("EXPECTED_DIM", "1280"))

app = Flask(__name__)
CORS(app)

_index = None
_dog_embeddings = None
_dog_filenames = None

def load_index():
    global _index, _dog_embeddings, _dog_filenames

    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(f"Missing {EMB_PATH}. Upload a 1280-d embeddings_dogs.npz.")

    d = np.load(EMB_PATH, allow_pickle=True)
    if "embeddings" not in d.files or "filenames" not in d.files:
        raise ValueError(f"{EMB_PATH} must contain 'embeddings' and 'filenames'. Found: {d.files}")

    emb = d["embeddings"].astype("float32")
    names = d["filenames"]

    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2D. Got shape {emb.shape}")

    n, dim = emb.shape
    if dim != EXPECTED_DIM:
        raise ValueError(f"Embedding dim {dim} != expected {EXPECTED_DIM}")

    # cosine similarity via inner product on normalized vectors
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    _index = index
    _dog_embeddings = emb
    _dog_filenames = names

load_index()

@app.get("/")
def root():
    return jsonify({"ok": True, "service": APP_NAME})

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": APP_NAME,
        "dogs": int(_dog_embeddings.shape[0]),
        "dim": int(_dog_embeddings.shape[1]),
        "r2_base": R2_BASE
    })

@app.post("/match")
def match():
    payload = request.get_json(force=True, silent=True) or {}
    emb = payload.get("embedding")
    top_k = int(payload.get("top_k", 6))

    if emb is None:
        return jsonify({"error": "Missing 'embedding' in JSON body"}), 400

    vec = np.array(emb, dtype="float32").reshape(1, -1)
    if vec.shape[1] != EXPECTED_DIM:
        return jsonify({"error": f"Embedding dim {vec.shape[1]} != expected {EXPECTED_DIM}"}), 400

    faiss.normalize_L2(vec)
    scores, idx = _index.search(vec, top_k)

    top_matches = []
    for i, s in zip(idx[0].tolist(), scores[0].tolist()):
        if i < 0:
            continue
        fname = str(_dog_filenames[i])
        top_matches.append({
            "dog_image": fname,
            "dog_url": f"{R2_BASE}/{fname}",
            "score": float(s)
        })

    return jsonify({"ok": True, "top_matches": top_matches})
PY
