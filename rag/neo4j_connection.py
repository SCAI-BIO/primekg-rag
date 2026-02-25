"""
Neo4j Database Connection Module

This module provides functions to connect to and interact with Neo4j database.
Reads connection parameters from environment variables.
"""
import os
from typing import Dict, Any, Optional, List
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_neo4j_connection_params():
    """
    Get Neo4j connection parameters from environment variables.
    
    Returns:
        Tuple of (uri, auth) where auth is (username, password)
    """
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    return uri, (username, password)


def create_driver():
    """
    Create and return a Neo4j driver instance.
    
    Returns:
        Neo4j driver instance or None if connection fails
    """
    uri, auth = get_neo4j_connection_params()
    username, password = auth
    
    if not password:
        return None
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j: {str(e)}")
        return None


def execute_query(query: str, parameters: Dict[str, Any] = None) -> Optional[List[Dict[str, Any]]]:
    driver = create_driver()
    if driver is None:
        return None
    
    try:
        with driver.session() as session:
            if parameters is None:
                parameters = {}
            result = session.run(query, parameters)
            return [record.data() for record in result]
    except Exception as e:
        print(f"Error executing Neo4j query: {str(e)}")
        return None
    finally:
        driver.close()


def test_connection() -> bool:
    """
    Test the Neo4j connection.
    
    Returns:
        True if connection is successful, False otherwise
    """
    driver = create_driver()
    if driver is None:
        return False
    
    try:
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


def get_connection_info() -> Dict[str, str]:
    """
    Get Neo4j connection information.
    
    Returns:
        Dictionary with connection info (uri, user, connected status)
    """
    uri, auth = get_neo4j_connection_params()
    username, _ = auth
    is_connected = test_connection()
    
    return {
        "uri": uri,
        "user": username,
        "connected": is_connected
    }

def get_driver():
    """
    Get Neo4j driver instance (for use with Streamlit's @st.cache_resource).
    Use this function when you need a persistent driver instance.
    
    Returns:
        Neo4j driver instance or None
    """
    return create_driver()


# Example usage
if __name__ == "__main__":
    print("Testing Neo4j connection...")
    
    # Test connection
    if test_connection():
        print("Connection successful!")
        
        # Example query
        results = execute_query("MATCH (n) RETURN count(n) as count LIMIT 1")
        if results:
            print(f"Database contains {results[0]['count']} nodes")
    else:
        print("Connection failed!")
        print("\nPlease set the following environment variables:")
        print("  - NEO4J_URI (default: bolt://localhost:7687)")
        print("  - NEO4J_USER (default: neo4j)")
        print("  - NEO4J_PASSWORD (required)")
