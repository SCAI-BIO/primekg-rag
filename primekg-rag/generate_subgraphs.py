import pandas as pd
import networkx as nx
import os
from tqdm import tqdm
from typing import Dict, List, Set, Tuple

class SubgraphGenerator:
    def __init__(self, kg_path: str, matches_path: str, output_dir: str = "subgraphs"):
        """
        Initialize the SubgraphGenerator.
        
        Args:
            kg_path: Path to the knowledge graph CSV file
            matches_path: Path to the CSV file containing best matches
            output_dir: Directory to save the generated subgraphs
        """
        self.kg_path = kg_path
        self.matches_path = matches_path
        self.output_dir = output_dir
        self.node_mapping = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Graph cache file
        self.graph_cache_file = os.path.join(output_dir, "graph_cache.gpickle")
        self.graph = self._load_or_build_graph()
    
    def _load_or_build_graph(self) -> nx.Graph:
        """Load graph from cache if available, otherwise build and cache it."""
        if os.path.exists(self.graph_cache_file):
            print("Loading graph from cache...")
            try:
                return nx.read_gpickle(self.graph_cache_file)
            except Exception as e:
                print(f"Error loading graph from cache, rebuilding... ({str(e)})")
                
        # If we get here, we need to build the graph
        print("Building graph...")
        graph = self._build_graph()
        
        # Cache the graph for future use
        try:
            nx.write_gpickle(graph, self.graph_cache_file)
            print(f"Graph cached to {self.graph_cache_file}")
        except Exception as e:
            print(f"Warning: Could not cache graph: {str(e)}")
            
        return graph
    
    def _build_graph(self) -> nx.Graph:
        """Build the knowledge graph from CSV."""
        # Read the knowledge graph
        df = pd.read_csv(self.kg_path)
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add edges to the graph
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graph"):
            source = str(row['x_index'])
            target = str(row['y_index'])
            relation = row['relation']
            
            # Add nodes if they don't exist
            if source not in G:
                G.add_node(source, label=row['x_type'], name=row['x_name'])
            if target not in G:
                G.add_node(target, label=row['y_type'], name=row['y_name'])
                
            # Add edge with relation as an attribute
            G.add_edge(source, target, relation=relation)
        
        print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def load_kg(self) -> nx.Graph:
        """Load the knowledge graph (returns the cached graph)."""
        return self.graph
    
    def get_subgraph(self, node_name: str, hops: int = 2) -> nx.Graph:
        """
        Get a subgraph containing all nodes within n hops of the target node.
        
        Args:
            node_name: Name of the target node
            hops: Number of hops to include in the subgraph
            
        Returns:
            A NetworkX subgraph
        """
        # Find the node ID by name
        target_node = None
        for node_id, data in self.graph.nodes(data=True):
            if data.get('name', '').lower() == node_name.lower():
                target_node = node_id
                break
        
        if target_node is None:
            print(f"Node '{node_name}' not found in the graph")
            return None
        
        # Get all nodes within n hops
        nodes_in_subgraph = set()
        nodes_in_subgraph.add(target_node)
        
        # Get nodes at each hop level
        current_level = {target_node}
        for _ in range(hops):
            next_level = set()
            for node in current_level:
                # Add all neighbors (both predecessors and successors)
                neighbors = set(self.graph.predecessors(node)) | set(self.graph.successors(node))
                next_level.update(neighbors)
            
            # Add new nodes to the subgraph
            new_nodes = next_level - nodes_in_subgraph
            nodes_in_subgraph.update(new_nodes)
            current_level = new_nodes
        
        # Create the subgraph
        subgraph = self.graph.subgraph(nodes_in_subgraph).copy()
        return subgraph
    
    def save_subgraph(self, subgraph: nx.Graph, node_name: str):
        """Save the subgraph to a CSV file."""
        # Create a list of edges with their attributes
        edges_data = []
        for u, v, data in subgraph.edges(data=True):
            edges_data.append({
                'source': u,
                'source_name': subgraph.nodes[u].get('name', ''),
                'source_type': subgraph.nodes[u].get('label', ''),
                'target': v,
                'target_name': subgraph.nodes[v].get('name', ''),
                'target_type': subgraph.nodes[v].get('label', ''),
                'relation': data.get('relation', '')
            })
        
        # Create a DataFrame and save to CSV
        if edges_data:
            df = pd.DataFrame(edges_data)
            safe_name = "".join([c if c.isalnum() else "_" for c in node_name]).strip("_")
            output_path = os.path.join(self.output_dir, f"{safe_name}_subgraph.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved subgraph for '{node_name}' to {output_path}")
            return output_path
        return None
    
    def process_matches(self, max_matches: int = None):
        """Process all matches and generate subgraphs."""
        # Load the matches
        print("\nLoading matches...")
        matches_df = pd.read_csv(self.matches_path)
        
        # Process each match
        results = []
        for idx, row in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Processing matches"):
            if max_matches and idx >= max_matches:
                break
                
            node_name = row['best_match']
            question = row['question']
            score = row['score']
            
            print(f"\nProcessing match {idx+1}/{len(matches_df)}: {node_name} (score: {score:.4f})")
            
            # Get the subgraph
            subgraph = self.get_subgraph(node_name, hops=2)
            
            if subgraph:
                # Save the subgraph
                output_path = self.save_subgraph(subgraph, node_name)
                
                # Save results
                results.append({
                    'question': question,
                    'matched_node': node_name,
                    'similarity_score': score,
                    'nodes_in_subgraph': subgraph.number_of_nodes(),
                    'edges_in_subgraph': subgraph.number_of_edges(),
                    'output_path': output_path
                })
        
        # Save summary of all processed matches
        if results:
            summary_df = pd.DataFrame(results)
            summary_path = os.path.join(self.output_dir, 'subgraphs_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSaved summary of all subgraphs to {summary_path}")

if __name__ == "__main__":
    # Initialize the generator
    generator = SubgraphGenerator(
        kg_path="kg.csv",
        matches_path="best_question_matches.csv",
        output_dir="generated_subgraphs"
    )
    
    # Process all matches
    generator.process_matches()
    
    print("\nSubgraph generation complete!")
