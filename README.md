# Multi-Utility Agentic Chatbot

A Streamlit chatbot with LangGraph, PDF question answering (RAG), web search,
stock quotes, and a calculator.

## Requirements

- Python 3.11 or later
- An OpenAI API key
- Docker Desktop (for the containerized version)

## Run locally with Python

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and enter your API keys. `OPENAI_API_KEY` is
   required for chat and document embeddings. `ALPHA_VANTAGE_API_KEY` enables
   stock quotes. LangSmith settings are optional.
4. Start the app:

   ```powershell
   streamlit run final_frontend.py
   ```

## Run with Docker Desktop

1. Copy `.env.example` to `.env` and set the keys you want to use.
2. In the project directory, build and start the app:

   ```powershell
   docker compose up --build
   ```

3. Open <http://localhost:8501>.
4. Stop the app with `Ctrl+C`; run `docker compose down` to remove the
   container. Conversation data and vector indexes are stored in Docker named
   volumes and remain available when the container is recreated.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub without committing `.env` or API keys.
2. In Streamlit Community Cloud, create an app from the repository, select
   `final_frontend.py` as the entry point, and deploy.
3. In the app's **Settings → Secrets**, add the keys needed by the app using
   TOML format:

   ```toml
   OPENAI_API_KEY = "your-openai-api-key"
   ALPHA_VANTAGE_API_KEY = "your-alpha-vantage-api-key"
   LANGSMITH_TRACING = "true"
   LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
   LANGSMITH_API_KEY = "your-langsmith-api-key"
   LANGSMITH_PROJECT = "multi-utility-agentic-rag"
   ```

   Only `OPENAI_API_KEY` is required. Omit optional keys if you do not use those
   services. Streamlit Secrets are loaded into the environment before the
   backend and LangSmith client are initialized.
4. Wait for the app to finish building. Streamlit Community Cloud will provide
   its shareable `*.streamlit.app` URL.

Community Cloud's local filesystem is not durable storage. Conversation and
uploaded-document data may be lost when the app restarts or is redeployed.
Docker Compose, by contrast, keeps this data in local Docker named volumes.
