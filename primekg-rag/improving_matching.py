import os
import chromadb
import pandas as pd
from tqdm import tqdm

# we just worked with the questions instead of q,a which made the matching hallucinating a bit
# --- Configuration ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_DB_PATH = "primekg_unified_db_asis"
CHROMA_COLLECTION_NAME = "unified_knowledge_asis"

QA_FILE_PATH = os.path.join(BASE_DIR, "mini_sample_cleaned.csv")
OUTPUT_FILE_PATH = os.path.join(BASE_DIR, "qa_to_node_matches_improved.csv")

# --- Initialize ChromaDB Client ---
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    print("Successfully connected to the ChromaDB collection.")
except Exception as e:
    print(f" Failed to connect to ChromaDB: {e}")
    collection = None


def find_best_node_for_qa(query_text: str):
    """Finds the single best matching node from nodes.csv for a query."""
    if collection is None:
        return None, None

    try:
        results = collection.query(query_texts=[query_text], n_results=1, where={"source": "nodes"})

        if not results["ids"] or not results["ids"][0]:
            return None, None

        best_match = results["metadatas"][0][0]
        distance = results["distances"][0][0]
        similarity = 1 - distance
        best_node_name = best_match["name"]

        return best_node_name, f"{similarity:.4f}"

    except Exception as e:
        print(f"An error occurred during query for '{query_text}': {e}")
        return None, None


if __name__ == "__main__":
    if collection:
        qa_df = pd.read_csv(QA_FILE_PATH)
        qa_df.dropna(inplace=True)

        match_results = []

        print(f"Finding best matches for {len(qa_df)} questions...")
        for index, row in tqdm(qa_df.iterrows(), total=qa_df.shape[0]):
            question = row["Question"]
            answer = row["Answer"]

            # --- THIS IS THE KEY CHANGE ---
            # Clean the question to get the core keyword for a better match.
            cleaned_query = question.replace("?", "").strip()

            # Use the cleaned query to find the best matching node
            node, score = find_best_node_for_qa(cleaned_query)

            if node and score:
                result_dict = {
                    "q": question,  # Store the original question
                    "a": answer,
                    "best_match_node": node,
                    "similarity_score": score,
                }
                match_results.append(result_dict)

        final_df = pd.DataFrame(match_results)
        final_df.to_csv(OUTPUT_FILE_PATH, index=False)

        print(f"\nDone! Improved results saved to '{OUTPUT_FILE_PATH}'")
        print("\n--- Sample of the new results ---")
        print(final_df.head())
