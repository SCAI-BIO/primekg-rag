import chromadb
from chromadb.config import Settings
from chromadb import Client

class AnalysesQueryInterface:
    def __init__(self, db_path: str = "./analyses_db"):
        self.db_path = db_path

        try:
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=db_path,
                anonymized_telemetry=False
            )
            self.client = Client(settings)

            try:
                self.collection = self.client.get_collection("medical_analyses")
            except Exception:
                print("Collection 'medical_analyses' does not exist. Creating it now.")
                self.collection = self.client.create_collection("medical_analyses")

            # Add test documents if empty
            if self.collection.count() == 0:
                print("No documents found. Adding test documents...")
                self.collection.add(
                    documents=[
                        "This is a test medical analysis document.",
                        "Another medical report for testing."
                    ],
                    metadatas=[
                        {"filename": "test1", "has_evidence": True, "has_summary": False, "content_length": 40},
                        {"filename": "test2", "has_evidence": False, "has_summary": True, "content_length": 30}
                    ],
                    ids=["doc1", "doc2"]
                )
                print(f"Added 2 test documents. Collection now has {self.collection.count()} documents.")

            print(f"Connected to analyses database: {db_path}")
            print(f"Collection contains {self.collection.count()} documents")
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            self.collection = None

    def search(self, query: str, n_results: int = 5, show_full: bool = False) -> None:
        if not self.collection:
            print("Database not connected!")
            return

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )

            if not results['documents'][0]:
                print("No results found!")
                return

            print(f"\n=== Search Results for: '{query}' ===")
            print(f"Found {len(results['documents'][0])} results\n")

            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                filename = metadata.get('filename', 'Unknown')
                similarity = 1 - distance

                print(f"{i+1}. File: {filename}")
                print(f"   Similarity: {similarity:.3f}")
                print(f"   Has Evidence: {metadata.get('has_evidence', False)}")
                print(f"   Has Summary: {metadata.get('has_summary', False)}")
                print(f"   Content Length: {metadata.get('content_length', 0)} chars")

                if show_full:
                    print(f"   Content:\n{doc}\n")
                else:
                    preview = doc[:300] + "..." if len(doc) > 300 else doc
                    print(f"   Preview: {preview}\n")

        except Exception as e:
            print(f"Search error: {str(e)}")

    def get_by_filename(self, filename: str) -> None:
        if not self.collection:
            print("Database not connected!")
            return

        try:
            results = self.collection.query(
                query_texts=[filename],
                n_results=20,
                include=['documents', 'metadatas', 'distances']
            )

            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                if metadata.get('filename') == filename:
                    print(f"\n=== Analysis: {filename} ===")
                    print(f"File path: {metadata.get('file_path', 'Unknown')}")
                    print(f"Has Evidence: {metadata.get('has_evidence', False)}")
                    print(f"Has Summary: {metadata.get('has_summary', False)}")
                    print(f"Content Length: {metadata.get('content_length', 0)} chars")
                    print(f"\nContent:\n{doc}")
                    return

            print(f"Analysis file '{filename}' not found!")

        except Exception as e:
            print(f"Retrieval error: {str(e)}")

    def list_all_analyses(self) -> None:
        if not self.collection:
            print("Database not connected!")
            return

        try:
            results = self.collection.get(
                include=['metadatas']
            )

            print(f"\n=== All Analyses ({len(results['metadatas'])} total) ===")

            for i, metadata in enumerate(results['metadatas']):
                filename = metadata.get('filename', 'Unknown')
                has_evidence = metadata.get('has_evidence', False)
                has_summary = metadata.get('has_summary', False)
                content_length = metadata.get('content_length', 0)

                status = []
                if has_evidence:
                    status.append("Evidence")
                if has_summary:
                    status.append("Summary")
                status_str = ", ".join(status) if status else "No sections"

                print(f"{i+1:3d}. {filename:<50} [{status_str}] ({content_length} chars)")

        except Exception as e:
            print(f"Listing error: {str(e)}")

    def get_stats(self) -> None:
        if not self.collection:
            print("Database not connected!")
            return

        try:
            results = self.collection.get(include=['metadatas'])
            total_docs = len(results['metadatas'])

            has_evidence = sum(1 for m in results['metadatas'] if m.get('has_evidence', False))
            has_summary = sum(1 for m in results['metadatas'] if m.get('has_summary', False))
            total_chars = sum(m.get('content_length', 0) for m in results['metadatas'])

            print(f"\n=== Database Statistics ===")
            print(f"Total documents: {total_docs}")
            print(f"Documents with Evidence: {has_evidence} ({has_evidence/total_docs*100:.1f}%)")
            print(f"Documents with Summary: {has_summary} ({has_summary/total_docs*100:.1f}%)")
            print(f"Total content: {total_chars:,} characters")
            print(f"Average document size: {total_chars/total_docs:.0f} characters")

        except Exception as e:
            print(f"Stats error: {str(e)}")


def interactive_query():
    interface = AnalysesQueryInterface()

    if not interface.collection:
        return

    print("\n=== Medical Analyses Query Interface ===")
    print("Commands:")
    print("  search <query>         - Search analyses")
    print("  get <filename>         - Get specific analysis by filename")
    print("  list                   - List all analyses")
    print("  stats                  - Show database statistics")
    print("  quit                   - Exit")
    print()

    while True:
        try:
            cmd = input("\nanalyses> ").strip()
            if not cmd:
                continue
            cmd_lower = cmd.lower()

            if cmd_lower in ("quit", "exit"):
                break
            elif cmd_lower == "list":
                interface.list_all_analyses()
            elif cmd_lower == "stats":
                interface.get_stats()
            elif cmd_lower.startswith("search "):
                query = cmd[7:]
                interface.search(query, n_results=5)
            elif cmd_lower.startswith("get "):
                filename = cmd[4:]
                interface.get_by_filename(filename)
            elif cmd_lower == "help":
                print("Commands: search <query>, get <filename>, list, stats, quit")
            else:
                print("Unknown command. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    interactive_query()
