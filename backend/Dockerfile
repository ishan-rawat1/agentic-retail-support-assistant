FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY evals ./evals
COPY pytest.ini .

# Seed DB and build the local vector index at image build time so the
# container is demo-ready without external services.
RUN python -m app.db.seed && python -m app.rag.ingest --clear

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
