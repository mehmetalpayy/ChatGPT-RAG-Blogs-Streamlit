<h1 align="center"><strong>CrewAI Coder System</strong></h1>

# ChatGPT RAG Blogs Streamlit

## Overview

This project is a Streamlit app that builds a Retrieval-Augmented Generation (RAG) pipeline from Medium articles and lets you ask questions over the collected content. It fetches article text, splits it into chunks, embeds the chunks, stores them in a local vector database, and answers questions with a LangChain RAG chain.

## What It Does

- Loads a curated set of Medium articles (default list) or user-provided URLs.
- Builds embeddings and a vector store (Chroma).
- Uses a multi-query retriever to improve recall.
- Generates answers through a RAG prompt and an OpenAI chat model.

## Requirements

- Python 3.9+
- An OpenAI API key (for embeddings and chat)
- Dependencies listed in `requirements.txt`

## Repository Structure

- `streamlit_app.py`: Streamlit UI and app flow.
- `rag.py`: Default URL list and a ready-to-use RAG chain.
- `helper.py`: Fetching, preprocessing, vector store creation, and chain assembly.
- `requirements.txt`: Python dependencies.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Run The App

```bash
streamlit run streamlit_app.py
```

## Usage

1. Click `Load Example Documents` in the sidebar to load the default Medium articles.
2. Optionally paste additional Medium article URLs (one per line) and click `Load Optional Documents`.
3. Enter a question and click `Submit` to receive a response grounded in the loaded content.

## Design Notes

- `MultiQueryRetriever` generates multiple query variants to improve retrieval coverage.
- Chunking is handled by `RecursiveCharacterTextSplitter`.
- The app stores the vector store and chain in `st.session_state` to avoid rebuilding on every query.

## Limitations

- Fetching relies on Medium’s HTML structure and can break if the site changes.
- Only Medium URLs are assumed by default; other sites may require custom parsing.
- The vector store is local and in-memory for the Streamlit session.

## Future Improvements

- Add persistent storage for the vector database.
- Support non-Medium sources with configurable loaders.
- Add UI controls for model selection and chunking parameters.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request.
