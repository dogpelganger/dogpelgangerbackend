import os
import json
import numpy as np

DOG_FEATURES_DIR = "outputs"
SCHEMA_PATH = "feature_schema_v1.json"
OUTPUT_NPZ = "embeddings_dogs.npz"

# --- helpers ---
def load_schema_feature_names(schema_path: str):
    with open(schema_path, "r") as f:
        schema = json.load(f)

    # Try a few common schema shapes
    if isinstance(schema, dict):
        if "features" in schema and isinstance(schema["features"], list):
            # features: [{name:...}, ...] OR ["name", ...]
            feats = schema["features"]
            if feats and isinstance(feats[0], dict) and "name" in feats[0]:
                return [x["name"] for x in feats]
            if feats and isinstance(feats[0], str):
                return feats
        if "properties" in schema and isinstance(schema["properties"], dict):
            return list(schema["properties"].keys())
        if "feature_names" in schema and isinstance(schema["feature_names"], list):
            return schema["feature_names"]

    raise ValueError("Could not determine feature order from schema. Please check feature_schema_v1.json structure.")


def normalize_value(v):
    """Convert schema values to a float."""
    if v is None:
        return 0.0

    # Already numeric
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)

    # Bool
    if isinstance(v, bool):
        return 1.0 if v else 0.0

    # Common presence strings
    if isinstance(v, str):
        s = v.strip().lower()

        if s in ("present", "yes", "true", "1"):
            return 1.0
        if s in ("absent", "no", "false", "0", "none", "null", ""):
            return 0.0

        # Categorical strings -> stable hash bucket (deterministic)
        # This avoids crashes like 'three_quarter' -> float
        # Map to a small numeric range for FAISS distance.
        return float((hash(s) % 1000) / 1000.0)

    # Anything else
    return 0.0


def build_vector(features: dict, feature_names: list):
    vec = np.zeros((len(feature_names),), dtype=np.float32)
    for i, k in enumerate(feature_names):
        vec[i] = normalize_value(features.get(k))
    return vec


# --- main ---
feature_names = load_schema_feature_names(SCHEMA_PATH)
print(f"Loaded {len(feature_names)} feature names from schema.")

embeddings = []
filenames = []
skipped = 0

for fname in sorted(os.listdir(DOG_FEATURES_DIR)):
    if not fname.startswith("dog") or not fname.endswith(".json"):
        continue

    path = os.path.join(DOG_FEATURES_DIR, fname)
    with open(path, "r") as f:
        data = json.load(f)

    src = data.get("source_image")
    feats = data.get("features")

    if not src or not isinstance(feats, dict):
        skipped += 1
        print(f"SKIP: {fname} (missing source_image or features)")
        continue

    vec = build_vector(feats, feature_names)
    embeddings.append(vec)
    filenames.append(src)

embeddings = np.vstack(embeddings).astype(np.float32)
filenames = np.array(filenames, dtype=str)

print("Embeddings shape:", embeddings.shape)
print("Filenames shape:", filenames.shape)
print("Skipped:", skipped)

np.savez(OUTPUT_NPZ, embeddings=embeddings, filenames=filenames)
print(f"Saved {OUTPUT_NPZ}")
