# PrimeKG-RAG: Knowledge Graph RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system built on PrimeKG (Prime Knowledge Graph) for medical research analysis. This system combines knowledge graphs with AI-powered analysis to provide insights into medical conditions, treatments, and research relationships.

## 🎯 Project Overview

This system processes medical knowledge graphs, retrieves relevant research papers, and generates AI-powered analyses to help researchers and clinicians understand complex medical relationships and evidence.

## 🏗️ System Architecture

The system consists of several interconnected components:

1. **Knowledge Graph Processing** - Extracts and processes medical relationships
2. **Paper Retrieval** - Finds relevant research papers from PubMed
3. **AI Analysis** - Generates insights using dynamic analysis
4. **Interactive Interface** - Streamlit web application for exploration

## 📁 File Structure & Documentation

### 🚀 **Core Application**

#### `app.py` - Main Streamlit Application
- **Purpose**: Primary web interface for the entire system
- **Features**:
  - Interactive subgraph selection and analysis
  - Dynamic AI-powered analysis generation
  - Real-time paper retrieval from PubMed
  - Interactive knowledge graph visualization
  - Path finding between medical concepts
  - Chat interface with AI analyst
- **Key Functions**:
  - `retrieve_similar_papers_dynamic()` - Retrieves relevant papers
  - `generate_dynamic_analysis()` - Creates AI analysis
  - `get_analysis_collection()` - Manages analysis database
- **Dependencies**: Streamlit, ChromaDB, OpenAI/Gemini APIs

### 📊 **Data Processing & Generation**

#### `generate_subgraphs.py` - Subgraph Generation Engine
- **Purpose**: Extracts focused subgraphs from the main knowledge graph
- **Features**:
  - Creates targeted subgraphs for specific medical conditions
  - Handles node mapping and relationship extraction
  - Generates CSV files for each subgraph
- **Key Classes**:
  - `SubgraphGenerator` - Main processing class
- **Output**: Individual CSV files in `new_subgraphs/` directory

#### `kg_analysis_shortest_path.py` - Knowledge Graph Analysis
- **Purpose**: Analyzes shortest paths between medical concepts
- **Features**:
  - Calculates shortest paths in the knowledge graph
  - Identifies key relationships between medical entities
  - Generates path analysis reports
- **Dependencies**: NetworkX, Pandas

### 🗄️ **Database Management**

#### `pmc_embed.py` - PubMed Database Creator
- **Purpose**: Creates and populates the PubMed papers database
- **Features**:
  - Processes PubMed JSON files
  - Creates embeddings for paper abstracts
  - Stores papers in ChromaDB for semantic search
- **Database**: `pubmed_db/` (ChromaDB)
- **Collection**: `pubmed_abstracts`
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`

#### `path_paper_downloader.py` - Path-Based Paper Retrieval
- **Purpose**: Downloads and stores papers related to shortest paths
- **Features**:
  - Finds papers for specific node relationships
  - Downloads full-text papers when available
  - Stores path-paper associations
- **Database**: `shortest_path_db/` (ChromaDB)
- **Collection**: `shortest_path_papers`

#### `retrieve_node_papers.py` - Node-Specific Paper Retrieval
- **Purpose**: Retrieves papers for specific medical nodes/concepts
- **Features**:
  - Semantic search for node-related papers
  - Paper ranking and filtering
  - Integration with PubMed database
- **Database**: Uses existing `pubmed_db/`

#### `knowledge-graph.py` - Unified Knowledge Database
- **Purpose**: Creates a unified knowledge database from multiple sources
- **Features**:
  - Combines nodes and Q&A facts
  - Creates embeddings for unified search
  - Stores in ChromaDB for retrieval
- **Database**: `primekg_unified_db_asis/` (ChromaDB)
- **Collection**: `unified_knowledge_asis`

### 🔧 **Utility & Support Files**

#### `export_utils.py` - Path Export Utilities
- **Purpose**: Exports detailed path information to CSV files
- **Features**:
  - Converts path data to structured CSV format
  - Handles timestamped exports
  - Processes relationship data
- **Output**: CSV files in `exported_paths/` directory

#### `pmc_parser.py` - PubMed Parser
- **Purpose**: Parses and processes PubMed data files
- **Features**:
  - Extracts abstracts and metadata
  - Handles batch processing
  - Updates ChromaDB with new data
- **Integration**: Works with `pmc_embed.py`

#### `repository.py` - Data Repository Utilities
- **Purpose**: Converts knowledge graph data to natural language format
- **Features**:
  - Verbalizes KG relationships into sentences
  - Creates metadata for each relationship
  - Exports to Parquet format
- **Status**: Legacy file, not actively used

#### `retriever.py` - Legacy Retrieval System
- **Purpose**: Two-stage pipeline for question-to-node mapping and subgraph extraction
- **Features**:
  - Maps questions to knowledge graph nodes
  - Extracts 2-hop subgraphs
  - Uses ChromaDB and DuckDB
- **Status**: Mostly commented out, legacy code

## 🗃️ **Database Structure**

The system uses several ChromaDB databases:

- **`pubmed_db/`** - PubMed papers with embeddings
- **`shortest_path_db/`** - Papers related to shortest paths
- **`node_db/`** - Node embeddings for semantic search
- **`question_db/`** - Question embeddings
- **`mappings_db/`** - Question-to-node mappings

## 🚀 **Getting Started**

### Prerequisites
```bash
pip install streamlit pandas chromadb sentence-transformers
pip install networkx matplotlib pyvis openai
```

### Running the Application
   ```bash
   streamlit run app.py
   ```

### Data Preparation
1. **PubMed Data**: Run `pmc_embed.py` to create the papers database
2. **Subgraphs**: Run `generate_subgraphs.py` to create subgraph files
3. **Knowledge Graph**: Ensure `kg.csv` and `nodes.csv` are available

## 🔄 **Workflow**

1. **Data Ingestion**: Papers and knowledge graph data are processed and stored
2. **Subgraph Generation**: Focused subgraphs are extracted for specific conditions
3. **Analysis**: Users select subgraphs and generate AI-powered analyses
4. **Retrieval**: Relevant papers are retrieved using semantic similarity
5. **Visualization**: Results are displayed through interactive interfaces

## 🎯 **Key Features**

- **Dynamic Analysis**: Real-time AI analysis generation
- **Semantic Search**: Advanced paper retrieval using embeddings
- **Interactive Visualization**: Network graphs and path exploration
- **Multi-Modal Interface**: Web app with chat and analysis capabilities
- **Scalable Architecture**: Modular design for easy extension

## 📊 **Data Sources**

- **PrimeKG**: Medical knowledge graph
- **PubMed**: Research paper abstracts and metadata
- **PMC**: Full-text papers (when available)

## 🔧 **Configuration**

Key configuration files and settings:
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **AI Models**: OpenAI GPT-4, Google Gemini
- **Database**: ChromaDB with cosine similarity
- **File Paths**: Configurable in each script

## 📈 **Performance**

- **Real-time Analysis**: Dynamic generation without pre-computation
- **Efficient Retrieval**: Optimized semantic search
- **Scalable Storage**: ChromaDB for large-scale data
- **Interactive UI**: Responsive Streamlit interface

## 🤝 **Contributing**

This system is designed for medical research applications. Key areas for extension:
- Additional data sources
- Enhanced visualization features
- Improved AI analysis capabilities
- Performance optimizations

## 📄 **License**

See LICENSE file for details.

---

**Note**: This system is designed for research purposes and should not be used for clinical decision-making without proper validation.