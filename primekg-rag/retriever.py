import chromadb
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
CHROMA_DB_PATH = "primekg_unified_db_asis"
CHROMA_COLLECTION_NAME = "unified_knowledge_asis"
QA_FILE_PATH = r"C:\Users\aemekkawi\Documents\GitHub\primekg-rag\primekg-rag\mini_sample_cleaned.csv"
OUTPUT_FILE_PATH = "qa_to_node_matches.csv"

# --- Initialize ChromaDB Client ---
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    print("✅ Successfully connected to the ChromaDB collection.")
except Exception as e:
    print(f"❌ Failed to connect to ChromaDB: {e}")
    collection = None


def find_best_node_for_qa(question_text: str):
    """Finds the single best matching node from nodes.csv for a question."""
    if collection is None:
        return None, None

    try:
        results = collection.query(
            query_texts=[question_text],
            n_results=1,  # We only want the single best match
            where={"source": "nodes"},
        )

        if not results["ids"] or not results["ids"][0]:
            return None, None

        # Extract info for the best match (the first result)
        best_match = results["metadatas"][0][0]
        distance = results["distances"][0][0]

        similarity = 1 - distance
        best_node_name = best_match["name"]

        return best_node_name, f"{similarity:.4f}"

    except Exception as e:
        print(f"An error occurred during query for '{question_text}': {e}")
        return None, None


if __name__ == "__main__":
    if collection:
        # Load the Q&A file to iterate through
        qa_df = pd.read_csv(QA_FILE_PATH)
        qa_df.dropna(inplace=True)

        match_results = []

        print(f"Finding best matches for {len(qa_df)} questions...")
        # Use tqdm for a progress bar
        for index, row in tqdm(qa_df.iterrows(), total=qa_df.shape[0]):
            question = row["Question"]
            answer = row["Answer"]

            # Find the best matching node and its score
            node, score = find_best_node_for_qa(question)

            if node and score:
                # Create the dictionary in the desired format
                result_dict = {
                    "q": question,
                    "a": answer,
                    "best_match_node": node,
                    "similarity_score": score,
                }
                match_results.append(result_dict)

        # Convert the list of dictionaries to a pandas DataFrame
        final_df = pd.DataFrame(match_results)

        # Save the final DataFrame to a CSV file
        final_df.to_csv(OUTPUT_FILE_PATH, index=False)

        print(f"\n✅ Done! Results saved to '{OUTPUT_FILE_PATH}'")
        print("\n--- Sample of the results ---")
        print(final_df.head())
