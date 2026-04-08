FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app

# Expose Hugging Face Space default port
EXPOSE 7860

# Run FastAPI server via Uvicorn
CMD ["uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "7860"]
