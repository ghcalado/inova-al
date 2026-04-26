<img width="800" height="455" alt="GravacaodeTela2026-04-26as16 43 33-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/b58ad917-aec9-4504-8ecd-aae29d7ff2df" />




https://github.com/user-attachments/assets/41db5480-7639-4399-9222-b38161cb7746





# InovaAL — Legal Assistant

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat&logo=flask&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.x-green?style=flat&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3-orange?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-purple?style=flat)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

AI-powered legal assistant for Alagoas Inovação (Brazil). Answers questions about state innovation legislation using RAG architecture with LangChain, Groq (LLaMA 3.3) and ChromaDB. Responses are grounded exclusively on the loaded legal documents.

## Overview

Legal consultation on innovation law is often inaccessible. This project builds an end-to-end RAG pipeline that allows anyone to query Alagoas state legislation in natural language — getting accurate, source-backed answers instantly.

The pipeline covers document ingestion, vector embedding, semantic retrieval, and a conversational interface with session memory and source citation.

## Features

| Feature | Description |
|---|---|
| Document Ingestion | Automated loading and chunking of PDF and TXT legal documents |
| Vector Store | ChromaDB with persistent storage — processes once, loads instantly |
| LLM | Groq API with LLaMA 3.3 70B for fast, high-quality responses |
| RAG Pipeline | LangChain retrieval chain with semantic search over legislation |
| Session Memory | Conversation history kept across turns within the same session |
| Source Citation | Every response shows which documents were used as reference |

## Tech Stack

- **Language:** Python 3.12
- **Backend:** Flask + Flask-CORS
- **LLM:** Groq API (LLaMA 3.3 70B Versatile)
- **RAG:** LangChain 1.x
- **Vector Store:** ChromaDB
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **Frontend:** Vanilla HTML, CSS, JavaScript

## Project Structure

```
inova-al/
├── docs/
│   ├── 237_texto_integral.pdf
│   ├── Lcp-182.pdf
│   ├── legislacao_federal_TIC.txt
│   ├── legislacao_tic_alagoas.pdf
│   ├── lei_no_8.956_de_4_de_setembro_de_2023.pdf
│   ├── lei_no_9.095_de_11_de_dezembro_de_2023.pdf
│   └── lei_no_9.272_de_11_de_junho_de_2024.pdf
├── app.py               # Flask backend — RAG pipeline and API endpoints
├── index.html           # Frontend — chat interface
├── Procfile             # Railway deployment config
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ghcalado/inova-al.git
cd inova-al
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root folder:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key at [console.groq.com](https://console.groq.com).

### 5. Run the server

```bash
python3 app.py
```

Open `http://localhost:8001` in your browser.

## How It Works

**Document ingestion** — on first run, all PDFs and TXT files in `./docs` are loaded, split into 1,000-character chunks with 200-character overlap, and embedded using `all-MiniLM-L6-v2`. The resulting vector store is saved to `./chroma_db`.

**Semantic retrieval** — each user query is embedded and compared against the vector store. The 4 most semantically similar chunks are retrieved as context.

**Response generation** — the retrieved chunks, conversation history, and user query are passed to LLaMA 3.3 70B via Groq. The model is instructed to answer only based on the provided context and cite relevant articles.

**Session memory** — the last 5 exchanges are appended to each prompt, allowing the assistant to handle follow-up questions naturally.

## Roadmap

- [ ] Multi-user session isolation
- [ ] SQLite for persistent conversation history
- [ ] REST API with FastAPI
- [ ] Docker containerization
- [ ] Support for more document formats (DOCX, HTML)

## Author

**Ghabriel Calado**
Computer Science Student | Python & AI

[GitHub](https://github.com/ghcalado) · [LinkedIn](https://www.linkedin.com/in/ghabriel-calado-7132a33b6/)
