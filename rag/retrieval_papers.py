import json
import chromadb

FEATURES_PATH = "features.json"
CHROMA_DIR = r"C:\Users\aemekkawi\Documents\GitHub\matching-system\chroma_pubmed_trd_mdd"
COLLECTION_NAME = "pubmed_mdd"

TOP_K = 10
DISTANCE_THRESHOLD = 0.8  # cosine distance

# =========================
# Load features
# =========================
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    features = json.load(f)

print(f"[SANITY] Loaded {len(features)} features")

# =========================
# Init Chroma (CORRECT)
# =========================
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

count = collection.count()
print(f"[SANITY] COLLECTION COUNT: {count}")

if count == 0:
    raise RuntimeError("Collection is empty — ingestion failed")

# =========================
# Probe query
# =========================
peek = collection.query(
    query_texts=["major depressive disorder antidepressant response"],
    n_results=5
)

print("[SANITY] Peek distances:",
      [round(d, 3) for d in peek["distances"][0]])
print("[SANITY] Peek PMIDs:",
      [m["pmid"] for m in peek["metadatas"][0]])

# =========================
# Query builder
# =========================
def build_query(feature, info):
    if info.get("type") == "gene":
        return f"{feature} gene antidepressant response depression"
    else:
        return info.get("definition", feature)

# =========================
# Retrieval
# =========================
results_all = {}

for feature, info in features.items():
    query = build_query(feature, info)

    res = collection.query(
        query_texts=[query],
        n_results=TOP_K
    )

    kept = []

    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0]
    ):
        if dist <= DISTANCE_THRESHOLD:
            kept.append({
                "feature": feature,
                "pmid": meta["pmid"],
                "cosine_distance": dist,
                "text": doc
            })

    results_all[feature] = kept
    print(f"[{feature}] kept {len(kept)} / {TOP_K}")

# =========================
# Save
# =========================
with open("retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump(results_all, f, indent=2)

print("Saved retrieval_results.json")
