import duckdb
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

# --- Configuration ---

import os

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Configuration ---
KG_CSV_PATH = os.path.join(BASE_DIR, 'primekg-rag', 'kg.csv')
MATCH_FILE_PATH = os.path.join(BASE_DIR, 'primekg-rag', 'qa_to_node_matches_improved.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'primekg-rag', 'subgraphs')

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)



def create_fully_deduplicated_subgraphs():
    
    # --- NEW: Dynamically get all unique relation types ---
    print(f"Finding all unique relation types from '{KG_CSV_PATH}'...")
    try:
        all_relations_df = duckdb.query(f"""
            SELECT DISTINCT display_relation FROM read_csv_auto('{KG_CSV_PATH}', ignore_errors=true)
        """).to_df()
        SYMMETRICAL_RELATIONS = all_relations_df['display_relation'].tolist()
        print(f"Found {len(SYMMETRICAL_RELATIONS)} unique relations. All will be treated as symmetrical.")
    except Exception as e:
        print(f"Error finding relations: {e}")
        SYMMETRICAL_RELATIONS = [] # Fallback to an empty list on error
    # --- END NEW ---

    matches_df = pd.read_csv(MATCH_FILE_PATH)
    nodes_to_process = matches_df['best_match_node'].unique()
    
    print(f"Starting clean subgraph creation for {len(nodes_to_process)} nodes.")

    for node_name in tqdm(nodes_to_process, desc="Processing Nodes"):
        try:
            safe_node_name = node_name.replace("'", "''")
            
            subgraph_df = duckdb.query(f"""
                SELECT x_name, x_type, display_relation, y_name, y_type 
                FROM read_csv_auto('{KG_CSV_PATH}', ignore_errors=true) 
                WHERE x_name = '{safe_node_name}' OR y_name = '{safe_node_name}';
            """).to_df()
            
            if not subgraph_df.empty:
                # Standardize all relationships before de-duplicating
                is_symmetrical = subgraph_df['display_relation'].isin(SYMMETRICAL_RELATIONS)
                needs_swap = is_symmetrical & (subgraph_df['x_name'] > subgraph_df['y_name'])
                
                x_name_orig, y_name_orig = subgraph_df.loc[needs_swap, 'x_name'], subgraph_df.loc[needs_swap, 'y_name']
                x_type_orig, y_type_orig = subgraph_df.loc[needs_swap, 'x_type'], subgraph_df.loc[needs_swap, 'y_type']
                
                subgraph_df.loc[needs_swap, 'x_name'] = y_name_orig
                subgraph_df.loc[needs_swap, 'y_name'] = x_name_orig
                subgraph_df.loc[needs_swap, 'x_type'] = y_type_orig
                subgraph_df.loc[needs_swap, 'y_type'] = x_type_orig
                
                subgraph_df.drop_duplicates(inplace=True)

                safe_filename = "".join(x for x in node_name if x.isalnum() or x in " _-").rstrip()
                output_path = os.path.join(OUTPUT_DIR, f"{safe_filename}_subgraph.csv")
                subgraph_df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"An error for node '{node_name}': {e}")
            continue
            
    print(f"\n✅ Process Complete. Fully de-duplicated subgraph files created in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    create_fully_deduplicated_subgraphs()