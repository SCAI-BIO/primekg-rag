# import pandas as pd
# import os
# from tqdm import tqdm
# import chromadb
# from chromadb.utils import embedding_functions
# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# # --- Subgraph & RAG DB Config ---
# # Define paths for the subgraph CSVs and the RAG-specific database.
# SUBGRAPHS_DIR = os.path.join(BASE_DIR, "subgraphs")
# RAG_DB_PATH = os.path.join(BASE_DIR, "rag_db") 
# RAG_COLLECTION_NAME = "subgraph_relations"

# def create_rag_database():
#     """
#     Reads all subgraph files, verbalizes the relationships into sentences,
#     and embeds them into a new ChromaDB collection for RAG.
#     """
#     if not os.path.exists(SUBGRAPHS_DIR):
#         print(f"ERROR: The directory '{SUBGRAPHS_DIR}' was not found.")
#         return

#     subgraph_files = [f for f in os.listdir(SUBGRAPHS_DIR) if f.endswith('.csv')]
#     if not subgraph_files:
#         print(f"No subgraph files were found in '{SUBGRAPHS_DIR}'.")
#         return

#     all_documents = []
#     for file_name in tqdm(subgraph_files, desc="Reading Subgraphs"):
#         file_path = os.path.join(SUBGRAPHS_DIR, file_name)
#         df = pd.read_csv(file_path)
#         for _, row in df.iterrows():
#             # Verbalize the relationship into a full sentence
#             sentence = f"{row['x_name']} {row['display_relation']} {row['y_name']}"
#             all_documents.append(sentence)
            
#     # Remove any duplicate sentences
#     all_documents = sorted(list(set(all_documents)))
#     print(f"Found {len(all_documents)} unique relationship sentences to embed.")

#     # Initialize a new ChromaDB collection
#     client = chromadb.PersistentClient(path=RAG_DB_PATH)
#     embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
#     collection = client.get_or_create_collection(
#         name=RAG_COLLECTION_NAME,
#         embedding_function=embedding_func,
#         metadata={"hnsw:space": "cosine"}
#     )
    
#     doc_ids = [f"doc_{i}" for i in range(len(all_documents))]
    
#     # Add documents to the collection in batches
#     batch_size = 4096
#     for i in tqdm(range(0, len(doc_ids), batch_size), desc="Embedding Documents"):
#         collection.add(
#             ids=doc_ids[i:i+batch_size], 
#             documents=all_documents[i:i+batch_size]
#         )

#     print(f"\nRAG Vector Database is ready.")
#     print(f"Total documents for retrieval: {collection.count()}")

# if __name__ == "__main__":
#     create_rag_database()