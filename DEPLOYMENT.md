# 🐳 Docker Deployment Guide

This guide explains how to containerize and deploy the PrimeKG-RAG system using Docker.

## 📋 Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)
- API keys for OpenAI/Gemini (if using AI features)

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Clone and navigate to the project:**
   ```bash
   cd primekg-rag
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Build and run:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   - Open your browser to `http://localhost:8501`

### Option 2: Using Docker directly

1. **Build the image:**
   ```bash
   docker build -t primekg-rag .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 \
     -v $(pwd)/primekg-rag/data:/app/primekg-rag/data \
     -v $(pwd)/primekg-rag/pubmed_db:/app/primekg-rag/pubmed_db \
     -e OPENAI_API_KEY=your_key_here \
     primekg-rag
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Required for AI features
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
API_KEY=your_general_api_key_here

# Optional
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Volume Mounts

The following directories should be mounted for data persistence:

- `./primekg-rag/data` - General data files
- `./primekg-rag/pubmed_db` - PubMed database
- `./primekg-rag/node_db` - Node embeddings database
- `./primekg-rag/question_db` - Question database
- `./primekg-rag/shortest_path_db` - Shortest path database
- `./primekg-rag/new_subgraphs` - Generated subgraphs

## 📊 Data Setup

### Initial Data Preparation

Before running the container, you need to prepare your data:

1. **Knowledge Graph Data:**
   ```bash
   # Ensure these files exist in primekg-rag/
   - kg.csv (main knowledge graph)
   - nodes.csv (node information)
   - questions_for_mapping.csv (questions for analysis)
   ```

2. **PubMed Data (Optional):**
   ```bash
   # Run data preparation scripts
   python primekg-rag/pmc_parser.py
   python primekg-rag/pmc_embed.py
   ```

3. **Generate Subgraphs:**
   ```bash
   python primekg-rag/generate_subgraphs.py
   ```

## 🏗️ Production Deployment

### Using Docker Compose for Production

1. **Create production docker-compose.yml:**
   ```yaml
   version: '3.8'
   services:
     primekg-rag:
       build: .
       ports:
         - "80:8501"  # Map to port 80
       volumes:
         - ./data:/app/primekg-rag/data
         - ./databases:/app/primekg-rag/databases
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
       restart: always
       deploy:
         resources:
           limits:
             memory: 4G
             cpus: '2.0'
   ```

2. **Deploy with:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Using Kubernetes

1. **Create deployment.yaml:**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: primekg-rag
   spec:
     replicas: 2
     selector:
       matchLabels:
         app: primekg-rag
     template:
       metadata:
         labels:
           app: primekg-rag
       spec:
         containers:
         - name: primekg-rag
           image: primekg-rag:latest
           ports:
           - containerPort: 8501
           env:
           - name: OPENAI_API_KEY
             valueFrom:
               secretKeyRef:
                 name: api-secrets
                 key: openai-key
   ```

## 🔍 Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "8502:8501"  # Use different host port
   ```

2. **Missing data files:**
   ```bash
   # Ensure data files exist before building
   ls primekg-rag/kg.csv
   ls primekg-rag/nodes.csv
   ```

3. **API key issues:**
   ```bash
   # Check environment variables
   docker exec -it container_name env | grep API
   ```

4. **Memory issues:**
   ```bash
   # Increase Docker memory limit
   # In Docker Desktop: Settings > Resources > Memory
   ```

### Logs and Debugging

```bash
# View container logs
docker-compose logs -f primekg-rag

# Access container shell
docker exec -it container_name /bin/bash

# Check container status
docker ps
```

## 📈 Performance Optimization

### Resource Allocation

```yaml
# In docker-compose.yml
services:
  primekg-rag:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

### Caching

- Use Docker layer caching for faster builds
- Mount data volumes for persistence
- Use multi-stage builds for smaller images

## 🔒 Security Considerations

1. **API Keys:**
   - Never commit API keys to version control
   - Use Docker secrets or environment files
   - Rotate keys regularly

2. **Network Security:**
   - Use reverse proxy (nginx) for production
   - Enable HTTPS
   - Restrict container network access

3. **Data Security:**
   - Encrypt sensitive data volumes
   - Use read-only mounts where possible
   - Regular backups

## 📝 Maintenance

### Updates

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```

### Backups

```bash
# Backup data volumes
docker run --rm -v primekg-rag_data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup.tar.gz -C /data .
```

## 🎯 Next Steps

1. **Set up monitoring** (Prometheus, Grafana)
2. **Implement CI/CD** pipeline
3. **Add health checks** and auto-restart
4. **Scale horizontally** with load balancer
5. **Set up logging** aggregation

---

For more detailed information, see the main README.md file.
