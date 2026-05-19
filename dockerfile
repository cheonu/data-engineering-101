FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY artifacts/ ./artifacts/

# Expose port
EXPOSE 8000

# Run the service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]