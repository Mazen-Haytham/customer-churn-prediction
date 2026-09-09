# ---- Build stage: slim Python image ----
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY app/ app/
COPY model/ model/
COPY data/ data/
COPY train_and_export.py .

# If model files don't exist yet, train and export them
RUN if [ ! -f model/adaboost_model.joblib ]; then python train_and_export.py; fi

# Expose the port (overridden by $PORT at runtime)
EXPOSE 8000

# Start the API — uses $PORT if set, defaults to 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
