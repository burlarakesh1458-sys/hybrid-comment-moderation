# Serving image: rules + calibrated TF-IDF/LR + optional LLM band + SQLite review queue.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
# models/tfidf_lr.joblib is committed (11.5 MB); rebuild with `make train` if you change data/params
COPY models ./models

RUN useradd -m -u 10001 app && mkdir -p /app/results && chown -R app /app
USER app

EXPOSE 8000
# GROQ_API_KEY is optional: without it the LLM band falls back to the ML threshold + human review
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"
CMD ["uvicorn", "hybridmod.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
