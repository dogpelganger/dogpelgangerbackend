import os
from typing import List, Optional, Tuple

import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS

APP_TITLE = "Dogpelgänger Backend (FAISS-only)"

R2_BASE_URL = os.environ.get(
    "R2_BASE_URL",
    "https://pub-4b9f9bd46442471da196ba4ed4966ab0.r2.dev"
).rstrip("/")

EMBEDDINGS_PATH = os.environ.get("EMBEDDINGS_PATH", "embeddings_dogs.npz")
DEFAULT_TOPK = int(os.environ.get("TOPK", "6"))

app = Flask(__name__)
CORS(app)

index: Optional[faiss.Index] = None
dog_filenames: List[str] = []
dog_matrix: Optional[np.ndarray] = None


def _load_embeddings() -> Tuple[np.ndarray, List[str]]:
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"Missing {EMBEDDINGS_PATH}. Put it in the repo root or set EMBEDDINGS_PATH."
        )

    d = np.load(EMBEDDINGS_PATH, allow_pickle=True)

    emb = d["embeddings"].astype(np.float32)     # (N, 512)
    names = [str(x) for x in d["filenames"].tolist()]  # (N,)

    if emb.ndim != 2:
        raise ValueError(f"'embeddings' must be 2D (N,D). Got shape {emb.shape}")
    if len(names) != emb.shape[0]:
        raise ValueError(f"filenames length {len(names)} != embeddings rows {emb.shape[0]}")

    return emb, names


def _build_faiss_index(emb: np.ndarray) -> faiss.Index:
    # cosine similarity = inner product after L2 normalization
    faiss.normalize_L2(emb)
    d = emb.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(emb)
    return idx


def _ensure_loaded():
    global index, dog_filenames, dog_matrix
    if index is not None:
        return

    emb, names = _load_embeddings()
    dog_matrix = emb
    dog_filenames = names
    index = _build_faiss_index(dog_matrix)

    print(f"[startup] Loaded {len(dog_filenames)} dogs. dim={dog_matrix.shape[1]}", flush=True)


@app.get("/")
def home():
    return jsonify({"ok": True, "service": APP_TITLE})


@app.get("/health")
def health():
    try:
        _ensure_loaded()
        return jsonify({"ok": True, "dogs": len(dog_filenames)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/match")
def match():
    """
    JSON body:
      { "embedding": [float,...], "top_k": 6 }

    Returns:
      { "top_matches": [ {dog_image, score, dog_url}, ... ] }
    """
    _ensure_loaded()

    payload = request.get_json(silent=True) or {}
    emb_list = payload.get("embedding")
    if not isinstance(emb_list, list) or len(emb_list) == 0:
        return jsonify({"error": "Missing 'embedding' (list of floats)."}), 400

    try:
        q = np.array(emb_list, dtype=np.float32).reshape(1, -1)
    except Exception:
        return jsonify({"error": "Invalid embedding format."}), 400

    if dog_matrix is None:
        return jsonify({"error": "Embeddings not loaded."}), 500

    if q.shape[1] != dog_matrix.shape[1]:
        return jsonify({"error": f"Embedding dim {q.shape[1]} != expected {dog_matrix.shape[1]}"}), 400

    faiss.normalize_L2(q)

    top_k = int(payload.get("top_k", DEFAULT_TOPK))
    top_k = max(1, min(top_k, 50))

    scores, idxs = index.search(q, top_k)

    out = []
    for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
        if i < 0:
            continue
        fname = dog_filenames[i]
        out.append({
            "dog_image": fname,
            "score": float(score),
            "dog_url": f"{R2_BASE_URL}/assets/dogs/{fname}",
        })

    return jsonify({"top_matches": out})
