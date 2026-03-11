FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts
COPY model/bpe_merges.json ./model/bpe_merges.json
COPY model/bpe_vocabulary.txt ./model/bpe_vocabulary.txt
COPY model/trigram_model.pkl ./model/trigram_model.pkl

# Copy source
COPY src/serve.py ./src/serve.py

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
