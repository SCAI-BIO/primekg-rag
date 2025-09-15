# 🐳 Docker Setup for PrimeKG RAG

This guide explains how to run the PrimeKG RAG clinical decision support system using Docker with databases from the [Zenodo repository](https://zenodo.org/records/17119877).

## 📋 Prerequisites

1. **Docker and Docker Compose** installed on your system
2. **API Keys** for the AI services:
   - OpenAI API key (required)
   - Google Gemini API key (optional)
   - Google API key (optional)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd primekg-rag
```

### 2. Create Environment File

Create a `.env` file in the root directory:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
API_KEY=your_google_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Build and Run

```bash
# Build the Docker image (downloads databases automatically)
docker-compose build

# Run the application
docker-compose up
```

The application will be available at: http://localhost:8501

## 🗄️ Database Integration

### Automatic Database Download

The Docker setup automatically:

1. **Downloads** the 467.6 MB `databases.7z` file from [Zenodo](https://zenodo.org/records/17119877)
2. **Extracts** all databases to the container
3. **Verifies** that all required databases are present
4. **Configures** the application to use the downloaded data

### Included Databases

- **`pubmed_db/`** - PubMed abstracts database with embeddings
- **`node_db/`** - Knowledge graph nodes with embeddings  
- **`mappings_db/`** - Question to node mappings
- **`shortest_path_db/`** - Precomputed shortest paths
- **`analyses_db/`** - AI-generated analyses storage
- **`new_subgraphs/`** - Extracted knowledge graph subgraphs
- **`nodes.csv`** - Knowledge graph nodes metadata
- **`best_question_matches.csv`** - Question mappings
- **`kg_shortest_paths.csv`** - Shortest paths data

## 🔧 Configuration

### Environment Variables

The application uses these environment variables for database paths:

```bash
SUBGRAPHS_DIR=/app/primekg-rag/new_subgraphs
NODES_CSV_PATH=/app/primekg-rag/nodes.csv
ANALYSIS_DB_PATH=/app/primekg-rag/analyses_db
QUESTION_NODE_MAPPING_PATH=/app/primekg-rag/best_question_matches.csv
PUBMED_DB_PATH=/app/primekg-rag/pubmed_db
NODE_DB_PATH=/app/primekg-rag/node_db
MAPPINGS_DB_PATH=/app/primekg-rag/mappings_db
SHORTEST_PATH_DB_PATH=/app/primekg-rag/shortest_path_db
```

### Volume Mounts

The Docker setup includes volume mounts for data persistence:

```yaml
volumes:
  - ./primekg-rag/data:/app/primekg-rag/data
  - ./primekg-rag/pubmed_db:/app/primekg-rag/pubmed_db
  - ./primekg-rag/node_db:/app/primekg-rag/node_db
  - ./primekg-rag/question_db:/app/primekg-rag/question_db
  - ./primekg-rag/shortest_path_db:/app/primekg-rag/shortest_path_db
  - ./primekg-rag/new_subgraphs:/app/primekg-rag/new_subgraphs
  - ./primekg-rag/analyses_db:/app/primekg-rag/analyses_db
```

## 🛠️ Development

### Local Development

For local development without Docker:

```bash
# Install dependencies
pip install -r primekg-rag/requirements.txt

# Download databases
cd primekg-rag
python setup_databases.py

# Verify setup
python verify_databases.py

# Run the application
streamlit run app.py
```

### Testing Database Setup

```bash
# Test if databases are present
python test_databases.py

# Verify all databases
cd primekg-rag
python verify_databases.py
```

## 🔍 Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```
   Error: OPENAI_API_KEY not found
   Solution: Set OPENAI_API_KEY in your .env file
   ```

2. **Database Download Failed**
   ```
   Error: Database setup failed
   Solution: Check internet connection and try rebuilding
   ```

3. **Database Verification Failed**
   ```
   Error: Database verification failed
   Solution: Run `python setup_databases.py` manually
   ```

### Logs and Debugging

```bash
# View container logs
docker-compose logs -f

# Access container shell
docker-compose exec primekg-rag bash

# Check database status
docker-compose exec primekg-rag python verify_databases.py
```

## 📊 Application Features

Once running, the application provides:

- **🔬 AI Expert Analysis** - Generate clinical insights from knowledge graphs
- **📊 Subgraph Summary** - View relationship evidence and statistics  
- **💬 Chat with AI Analyst** - Interactive Q&A about medical conditions
- **📈 Interactive Graph** - Visualize knowledge graph relationships
- **📄 Raw Data** - Access underlying CSV data
- **🔍 Path Finder** - Find connections between medical concepts

## 🎯 Usage

1. **Select a Subgraph** - Choose a medical condition from the dropdown
2. **Generate Analysis** - Click "Generate Dynamic Analysis" for AI insights
3. **Chat** - Ask questions about the selected condition
4. **Explore** - Use the interactive graph to visualize relationships
5. **Find Paths** - Discover connections between medical concepts

## 📚 References

- **Zenodo Repository**: [https://zenodo.org/records/17119877](https://zenodo.org/records/17119877)
- **PrimeKG**: Knowledge graph for precision medicine
- **PubMed**: Medical literature database
- **ChromaDB**: Vector database for embeddings

## 🤝 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all environment variables are set
3. Ensure databases are downloaded correctly
4. Check Docker logs for detailed error messages

---

**Note**: The first build may take several minutes as it downloads the 467.6 MB database archive from Zenodo.
