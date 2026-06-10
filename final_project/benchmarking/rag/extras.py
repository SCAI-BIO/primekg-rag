from pymed import PubMed
import json


# Fetch a focused set of TRD/MDD papers for the genes used in the benchmark.
pubmed = PubMed(
    tool="RAG_Thesis_Tool",
    email="ahmedmekkawi97@gmail.com",
)

query = (
    "("
    "MAOA OR DRD4 OR HTR2C OR ADRA1A OR ADRA2A OR "
    "OPRM1 OR OPRK1 OR KCND2 OR KCND3 OR "
    "KCNQ2 OR KCNQ3 OR CHRNA3 OR CHRNB4 OR "
    "CHRM1 OR ABAT OR GPT2"
    ") AND ("
    "antidepressant response OR antidepressant non-response OR "
    "treatment resistant OR treatment resistance OR refractory"
    ") AND ("
    "major depressive disorder OR MDD OR depression"
    ") NOT ("
    "bipolar OR cancer OR carcinoma OR tumor OR oncology"
    ")"
)


results = pubmed.query(query, max_results=2000)

with open("pubmed_trd_extras.jsonl", "w", encoding="utf-8") as f:
    for article in results:
        pmid_raw = article.pubmed_id

        if not pmid_raw:
            continue

        pmid = str(pmid_raw).strip().split()[0]

        if not pmid.isdigit():
            continue

        record = {
            "pmid": pmid,
            "title": article.title or "",
            "abstract": article.abstract or "",
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Done")
