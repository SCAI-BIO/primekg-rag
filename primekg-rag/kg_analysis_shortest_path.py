import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Optional

def load_kg_data() -> Tuple[pd.DataFrame, nx.Graph]:
    """Load and prepare the knowledge graph data."""
    # Read the node matches file
    matches_df = pd.read_csv("C:/Users/aemekkawi/Documents/GitHub/primekg-rag/primekg-rag/node_matches.csv")
    
    # Load the KG
    kg_path = "C:/Users/aemekkawi/Documents/GitHub/primekg-rag/primekg-rag/kg.csv"
    kg_df = pd.read_csv(kg_path, low_memory=False)
    
    # Create a mapping of node names to their types
    node_types = {}
    for _, row in kg_df.iterrows():
        node_types[row['x_name']] = row['x_type']
        node_types[row['y_name']] = row['y_type']
    
    # Create a graph with edge attributes
    G = nx.from_pandas_edgelist(
        kg_df, 
        source='x_name', 
        target='y_name', 
        edge_attr=['relation', 'display_relation', 'x_type', 'y_type'],
        create_using=nx.Graph()
    )
    
    # Add node types as node attributes
    for node, node_type in node_types.items():
        if node in G:
            G.nodes[node]['type'] = node_type
    
    return matches_df, G, kg_df

def get_path_with_details(G: nx.Graph, path: List[str]) -> str:
    """Generate a detailed path string with node types and edge relations."""
    detailed_path = []
    
    for i in range(len(path) - 1):
        node_a = path[i]
        node_b = path[i + 1]
        
        # Get node types
        node_a_type = G.nodes[node_a].get('type', 'unknown')
        node_b_type = G.nodes[node_b].get('type', 'unknown')
        
        # Get edge data
        edge_data = G.get_edge_data(node_a, node_b, {})
        
        # Get relation type (prefer display_relation if available)
        relation = edge_data.get('display_relation', edge_data.get('relation', 'related_to'))
        
        # Format the path segment
        detailed_path.append(f"{node_a} [{node_a_type}] --{relation}--> {node_b} [{node_b_type}]")
    
    return "\n  • ".join([""] + detailed_path)

def main():
    # Load data
    print("Loading knowledge graph data...")
    matches_df, G, kg_df = load_kg_data()
    
    results = []
    total_pairs = len(matches_df) * (len(matches_df) - 1) // 2
    processed_pairs = 0
    
    print(f"Analyzing paths for {total_pairs} node pairs...")
    
    for i, row_a in matches_df.iterrows():
        for j, row_b in matches_df.iterrows():
            if i >= j:
                continue
                
            node_a = row_a['Node_Name']
            node_b = row_b['Node_Name']
            
            processed_pairs += 1
            if processed_pairs % 100 == 0:
                print(f"Processed {processed_pairs}/{total_pairs} pairs...")
            
            if node_a not in G or node_b not in G:
                results.append({
                    "Source_Query": row_a['Query'],
                    "Target_Query": row_b['Query'],
                    "Source_Node": node_a,
                    "Target_Node": node_b,
                    "Status": "Node not found in graph",
                    "Path": None,
                    "Path_Length": None,
                    "Detailed_Path": None
                })
                continue
                
            try:
                # Get the shortest path
                path = nx.shortest_path(G, source=node_a, target=node_b)
                path_length = len(path) - 1
                path_str = " → ".join(path)
                
                # Get detailed path with types and relations
                detailed_path = get_path_with_details(G, path)
                
                results.append({
                    "Source_Query": row_a['Query'],
                    "Target_Query": row_b['Query'],
                    "Source_Node": node_a,
                    "Target_Node": node_b,
                    "Status": "Path found",
                    "Path": path_str,
                    "Path_Length": path_length,
                    "Detailed_Path": detailed_path
                })
                
            except nx.NetworkXNoPath:
                results.append({
                    "Source_Query": row_a['Query'],
                    "Target_Query": row_b['Query'],
                    "Source_Node": node_a,
                    "Target_Node": node_b,
                    "Status": "No path exists",
                    "Path": None,
                    "Path_Length": None,
                    "Detailed_Path": None
                })

    # Save results
    df_results = pd.DataFrame(results)
    output_path = "kg_shortest_paths_fixed.csv"
    
    # Reorder columns for better readability
    column_order = [
        'Source_Query', 'Target_Query', 'Status', 'Path_Length',
        'Source_Node', 'Target_Node', 'Path', 'Detailed_Path'
    ]
    df_results = df_results[column_order]
    
    df_results.to_csv(output_path, index=False)
    print(f"\nAnalysis complete! Results saved to {output_path}")

    # Print summary
    connected_pairs = df_results[df_results['Status'] == 'Path found']
    print(f"\nFound paths for {len(connected_pairs)} out of {len(df_results)} pairs")
    if not connected_pairs.empty:
        print(f"Average path length: {connected_pairs['Path_Length'].mean():.2f}")
        print("\nTop 5 shortest paths:")
        for _, row in connected_pairs.nsmallest(5, 'Path_Length').iterrows():
            print(f"\n{row['Source_Query']} -> {row['Target_Query']}:")
            print(f"  Path: {row['Path']}")
            if pd.notna(row['Detailed_Path']):
                print(f"  Detailed Path:\n{row['Detailed_Path']}")
            print(f"  Length: {row['Path_Length']} steps")

if __name__ == "__main__":
    main()