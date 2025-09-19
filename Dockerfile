FROM python:3.10-slim

# Set working directory
WORKDIR /app/primekg-rag

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy application code and requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY primekg-rag .

# Run database setup and then start app
CMD python setup_databases.py && streamlit run app.py