# Multi-stage Dockerfile for LangGraph Multi-Utility Chatbot
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt pytest

# Final lightweight runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

# Non-root user for security compliance
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/vectorstores /app/data && \
    chown -R appuser:appuser /app

COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . /app

USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV CHATBOT_DB_PATH=/app/data/chatbot.db

EXPOSE 8501

# Healthcheck to verify Streamlit responsiveness
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "final_frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
