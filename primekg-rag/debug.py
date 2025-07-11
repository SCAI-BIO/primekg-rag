import chromadb
import os

# --- Configuration ---
ANALYSIS_DB_PATH = "analyses_db"
ANALYSIS_COLLECTION_NAME = "subgraph_analyses"

def view_stored_analyses():
    """
    Connects to the analyses database and prints all stored AI responses.
    """
    print(f"--- 🔎 Retrieving Stored Analyses from: {ANALYSIS_DB_PATH} ---")

    # 1. Check if the database directory exists
    if not os.path.exists(ANALYSIS_DB_PATH):
        print(f"❌ ERROR: The database directory '{ANALYSIS_DB_PATH}' was not found.")
        return

    # 2. Connect to the database and collection
    try:
        client = chromadb.PersistentClient(path=ANALYSIS_DB_PATH)
        collection = client.get_collection(name=ANALYSIS_COLLECTION_NAME)
    except Exception as e:
        print(f"❌ ERROR: Could not connect to the database or collection: {e}")
        return

    # 3. Retrieve all items from the collection
    # We get the total count first to ensure we retrieve everything.
    total_items = collection.count()
    if total_items == 0:
        print("⚠️ The analyses database is empty.")
        return
    
    print(f"Found {total_items} stored analyses. Displaying now...")
    all_analyses = collection.get(limit=total_items, include=["documents", "metadatas"])

    # 4. Print each analysis in a readable format
    for i in range(len(all_analyses['ids'])):
        topic = all_analyses['metadatas'][i].get('topic', 'Unknown Topic')
        analysis_text = all_analyses['documents'][i]

        print("\n" + "="*50)
        print(f"ANALYSIS FOR TOPIC: {topic.upper()}")
        print("="*50)
        print(analysis_text)
        print("="*50 + "\n")


if __name__ == "__main__":
    view_stored_analyses()