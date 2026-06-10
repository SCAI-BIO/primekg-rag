from pathlib import Path
from typing import Dict, List

import chromadb

from .config import log


def connect_chroma(chroma_dir: Path, collection_name: str):
    if not chroma_dir.exists():
        log.warning(f"ChromaDB dir not found: {chroma_dir} - PMID fidelity will be skipped.")
        return None

    try:
        c = chromadb.PersistentClient(path=str(chroma_dir))
        coll = c.get_collection(name=collection_name)
        log.info(f"ChromaDB '{collection_name}': {coll.count():,} documents")
        return coll
    except Exception as e:
        log.warning(f"ChromaDB connection failed: {e}")
        return None


def retrieve_abstracts_by_pmids(pmids: List[str], collection) -> Dict[str, Dict]:
    if not collection or not pmids:
        return {}

    abstract_map: Dict[str, Dict] = {}

    try:
        str_pmids = [str(p) for p in pmids]
        results = collection.get(
            where={"pmid": {"$in": str_pmids}},
            include=["documents", "metadatas"],
        )

        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            pmid = str(meta.get("pmid", ""))

            if not pmid:
                continue

            full_text = doc or ""
            parts = full_text.split("\n\n", 1)

            title = parts[0] if parts else ""
            abstract = parts[1] if len(parts) > 1 else full_text

            abstract_map[pmid] = {
                "title": title.strip(),
                "abstract": abstract.strip(),
            }

    except Exception as e:
        log.warning(f"Abstract re-retrieval failed: {e}")

    return abstract_map

