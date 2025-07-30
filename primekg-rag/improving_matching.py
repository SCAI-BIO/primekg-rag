import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from tqdm import tqdm
import logging
import sys

# --- Configuration ---
BASE_DIR = Path(__file__).parent
NODE_CSV_PATH = BASE_DIR / "nodes.csv"
PERSIST_DIR = BASE_DIR / "node_db"
COLLECTION_NAME = "node_embeddings"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000

# --- Logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# --- Main ---
if __name__ == "__main__":
    logger.info("--- Starting Node Embedding Pipeline ---")

    if not NODE_CSV_PATH.is_file():
        logger.critical(f"CSV file not found: '{NODE_CSV_PATH}'")
        sys.exit(1)

    # Load and deduplicate
    df = pd.read_csv(NODE_CSV_PATH)
    required_cols = ["id", "name", "type"]
    if not all(col in df.columns for col in required_cols):
        logger.critical(f"CSV must contain columns: {required_cols}")
        sys.exit(1)

    logger.info(f"Loaded {len(df)} nodes from '{NODE_CSV_PATH.name}'.")

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["id"])
    after_dedup = len(df)
    logger.info(
        f"Removed {before_dedup - after_dedup} duplicate nodes. {after_dedup} remain."
    )

    # Connect to ChromaDB
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))

    # Get or create collection
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=embedding_fn
        )
        logger.info(f"🔄 Existing collection '{COLLECTION_NAME}' loaded.")
    except:
        collection = client.create_collection(
            name=COLLECTION_NAME, embedding_function=embedding_fn
        )
        logger.info(f"📦 New collection '{COLLECTION_NAME}' created.")

    # Prepare IDs
    df["chroma_id"] = df["id"].apply(lambda x: f"node_{x}")

    # Check which IDs are already in DB
    existing_ids = set(collection.get(include=[])["ids"])  # Get just the IDs
    df_to_embed = df[~df["chroma_id"].isin(existing_ids)]

    logger.info(f"⚙️ {len(existing_ids)} nodes already embedded.")
    logger.info(f"🧠 {len(df_to_embed)} new nodes to embed.")

    # Embed new entries only
    for i in tqdm(range(0, len(df_to_embed), BATCH_SIZE), desc="Embedding New Nodes"):
        batch = df_to_embed.iloc[i : i + BATCH_SIZE]
        documents = batch["name"].astype(str).tolist()
        ids = batch["chroma_id"].tolist()
        metadatas = batch.drop(columns=["name", "chroma_id"]).to_dict(orient="records")

        try:
            collection.add(documents=documents, ids=ids, metadatas=metadatas)
        except Exception as e:
            logger.error(f"Failed to add batch at index {i}: {e}")

    logger.info(f"✅ Done. Total in collection: {collection.count()} nodes.")
