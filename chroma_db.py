import json
import os
import chromadb
from chromadb.utils import embedding_functions

INPUT_FILE = r"pubmed_records_unique.json"

# Human names
CHROMA_DIR = r"./chroma_data/pubmed_trd_mdd"
COLLECTION_NAME = "pubmed_trd_mdd"

# Keep the embedding function inside the collection config (avoids conflicts later)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

BATCH = 5000


def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    ids, docs, metas = [], [], []

    for rec in records:
        pmid = str(rec.get("pmid") or "").strip()
        title = rec.get("title") or ""
        abstract = rec.get("abstract") or ""

        if not pmid or not abstract:
            continue

        text = f"{title}\n\n{abstract}".strip()

        ids.append(pmid)
        docs.append(text)
        metas.append({
            "pmid": pmid,
            "source": "pubmed",
            "topic": "TRD_MDD",
        })

    print(f"Prepared {len(ids)} documents. Writing to Chroma in batches...")

    # Add in batches (faster + safer)
    total = 0
    for ids_b, docs_b, metas_b in zip(chunk(ids, BATCH), chunk(docs, BATCH), chunk(metas, BATCH)):
        collection.add(ids=ids_b, documents=docs_b, metadatas=metas_b)
        total += len(ids_b)
        print("Added:", total)

    print("\nDone.")
    print("DB folder :", CHROMA_DIR)
    print("Collection:", COLLECTION_NAME)
    print("Count     :", collection.count())


if __name__ == "__main__":
    main()
