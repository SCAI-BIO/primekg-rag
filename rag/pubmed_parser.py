from pymed import PubMed
import json


#from the docs of pymed on github
pubmed = PubMed(
    tool="My_Tool",
    email="ahmedmekkawi97@gmail.com"  
)

#I tried it more than once to get the best results, it is consistent of roughly 20k articles
query = (
    '("inadequate response" OR "non-response" OR refractory OR "failed treatment") '
    'AND ("major depressive disorder" OR MDD OR depression) '
    'AND (antidepressant OR psychiatric) '
    'NOT (cancer OR carcinoma OR tumor OR oncology) '
    'NOT bipolar'
)

results = pubmed.query(query, max_results=2000)
# Saving corpus
with open("pubmed_trd_new.jsonl", "w", encoding="utf-8") as f:
    for article in results:
        pmid_raw = article.pubmed_id

        if not pmid_raw:
            continue

        pmid = str(pmid_raw).strip().split()[0]

        # optional but recommended safety check
        if not pmid.isdigit():
            continue


        pmid = pmid_raw.splitlines()[0]

        record = {
            "pmid": pmid,
            "title": article.title,
            "abstract": article.abstract or "",
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Done")

