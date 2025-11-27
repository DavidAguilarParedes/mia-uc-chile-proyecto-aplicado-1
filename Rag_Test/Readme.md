# 🧪 RAG Hito 1: Clean Architecture

This project implements a Retrieval-Augmented Generation (RAG) system using **Hexagonal Architecture**. It uses **Streamlit** for the UI and **Qdrant** for vector storage.

## ⚡️ Quick Start

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.

### 1. Prerequisites
Ensure you have `uv` installed:
```bash
# MacOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"





.
├── .env                   # Environment variables (API Keys)
├── data/                  # Place your PDF documents here
├── app.py                 # Streamlit Entry Point (UI)
├── main.py                # CLI Entry Point
├── pyproject.toml         # Dependencies (optional if using uv directly)
└── src/
    ├── domain/            # Entities (ProcessedChunk, etc.)
    ├── application/       # Ports (Interfaces) & Services (Use Cases)
    └── infrastructure/    # Adapters (PDFLoader, Qdrant, HuggingFace)


To run use:

uv sync

uv run streamlit run app.py

query ejemplo :

what is the m/z of  4-Dihydroxyacetophenone
