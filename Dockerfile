FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY primekg-rag/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir \
    openai \
    google-generativeai \
    networkx \
    matplotlib \
    tqdm \
    duckdb

# Copy the entire project
COPY . .

# Set working directory to primekg-rag
WORKDIR /app/primekg-rag

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]