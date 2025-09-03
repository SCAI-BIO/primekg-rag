from chromadb import Client

# connect to local Chroma DB (default settings)
client = Client()

# list all collections (like tables)
collections = client.list_collections()
print("Collections:", collections)

# get a specific collection
collection = client.get_collection("medical_analysis")

# fetch some data or query
results = collection.query(query_texts=["example query"], n_results=5)
print(results)
