improving_matching.py

Embeds nodes from nodes.csv using a SentenceTransformer model (all-MiniLM-L6-v2) and stores them in a Chroma vector database for semantic search.

Matches each question from questions_for_mapping.csv to the most semantically similar node from the database, based on text similarity, and saves the best match and its similarity score to a new CSV file (qa_to_node_matches_improved.csv).

retriever.py

Reads the Excel File: It opens your Codebook opzet.xlsx file and looks specifically for the sheet named MINI.

Finds the Right Column: It then searches for the column named Item + question within that sheet.

Extracts the Data: It copies all the text from that single column.

Cleans the Data: It removes any empty rows to make sure the list is clean.

Saves to a New File: Finally, it saves this clean list of questions into a new file named questions_for_mapping.csv.

it then maps it with the kg using cosine sim and keep the matched with the best match with the percentage of match and store them in a db

and creates subgraphs to each node from the kg itself 

# This is a two-stage pipeline to prepare data for AI analysis.
# 1. MAPPING: Maps questions to the most semantically similar nodes in the `node_db`.
# 2. SUBGRAPH EXTRACTION: Extracts focused subgraphs for each matched node from `kg.csv`.