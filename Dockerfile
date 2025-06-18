FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY primekg-rag/ primekg-rag/

# Run the client
CMD ["python", "primekg-rag/client.py"]
