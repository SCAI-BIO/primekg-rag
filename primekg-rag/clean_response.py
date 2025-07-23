import chromadb
from tqdm import tqdm

# --- Configuration ---
ANALYSIS_DB_PATH = "analyses_db"
ANALYSIS_COLLECTION_NAME = "subgraph_analyses"


def clean_analyses_in_db():
    """
    Connects to the analyses database, cleans each document, and updates it.
    """
    print(f"--- 🧼 Cleaning stored analyses in: {ANALYSIS_DB_PATH} ---")

    try:
        client = chromadb.PersistentClient(path=ANALYSIS_DB_PATH)
        collection = client.get_collection(name=ANALYSIS_COLLECTION_NAME)
    except Exception as e:
        print(f"❌ ERROR: Could not connect to the database: {e}")
        return

    total_items = collection.count()
    if total_items == 0:
        print("⚠️ The analyses database is empty. Nothing to clean.")
        return

    all_analyses = collection.get(limit=total_items, include=["documents"])

    ids_to_update = all_analyses["ids"]
    cleaned_documents = []

    # --- NEW: Counter for printing ---
    cleaned_count = 0
    print_limit = 3  # Set how many examples to print

    print(f"Cleaning {len(ids_to_update)} documents...")
    for i, doc in enumerate(tqdm(all_analyses["documents"], desc="Cleaning Documents")):
        original_doc = doc
        clean_text = doc  # Default to original text

        if "</think>" in doc:
            try:
                # Split the text and keep only the part after the marker
                potential_clean_text = doc.split("</think>")[1].strip()
                if potential_clean_text:  # Ensure the result is not empty
                    clean_text = potential_clean_text
                    if i < print_limit:  # Only print for the first few items
                        cleaned_count += 1

            except IndexError:
                # This will be hit if the split fails unexpectedly
                clean_text = doc

        # --- NEW: Print statements for the first few documents ---
        if i < print_limit:
            print(f"\n\n--- 🕵️ Checking Document {i+1} ---")
            print("--- Original Text ---")
            print(original_doc)
            print("\n--- Cleaned Text ---")
            print(clean_text)
            print("--------------------------")

        cleaned_documents.append(clean_text)

    if cleaned_documents:
        print("\nUpdating database with cleaned documents...")
        collection.update(ids=ids_to_update, documents=cleaned_documents)
        print("✅ Database cleaning complete.")
        print(
            f"({cleaned_count} out of the first {print_limit} examples shown were cleaned.)"
        )
    else:
        print("No documents were cleaned.")


if __name__ == "__main__":
    clean_analyses_in_db()
