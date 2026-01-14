# Agentic RAG System (AI Academy Final Project)

This project implements an autonomous **Agentic RAG (Retrieval-Augmented Generation) System** designed to intelligently answer user queries. 
It moves beyond simple "retrieval" by incorporating self-reflection, autonomous decision-making, and tool usage.

## 🚀 Key Features (Assignment Objectives)

* **RAG Pipeline:** Ingests PDF and Audio files into a Qdrant vector database for local context retrieval.
* **Reasoning & Reflection:**
    * **Document Grading:** The agent "reads" retrieved documents and decides if they are actually relevant before answering.
    * **Self-Evaluation:** After answering, the agent rates its own answer (1-5 score).
* **Tool-Calling Mechanisms:**
    * **Web Fallback:** Automatically switches to **DuckDuckGo** if local documents are insufficient.
    * **File Logging:** Uses a dedicated tool to save Q&A pairs to `agent_output.txt`.
* **Agentic Loop:** If a RAG answer receives a low evaluation score, the system automatically triggers a "Retry".

## 🛠️ Technology Stack

* **Language:** Python 3.10+
* **LLM & Embeddings:** OpenAI-compatible API (configured for **LM Studio**)
* **Vector Database:** Qdrant (Local instance)
* **Search Tool:** DuckDuckGo Search
* **Libraries:** `langchain`, `qdrant-client`, `pdfplumber`, `whisper`, `nltk`

## 📋 Prerequisites

1.  **Python 3.10+** installed.
2.  **Qdrant** running locally:
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```
3.  **LM Studio** (or compatible Local LLM server) running:
    * **Base URL:** `http://localhost:1234/v1`
    * **LLM:** Loaded (e.g., Llama-3.2-1B-Instruct)
    * **Embeddings:** Loaded (e.g., All-MiniLM-L6-v2)

## 📦 Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

### 1. Ingest Data (Data Preparation)
Load your specific documents or audio recordings into the vector database.
```bash
python agent_rag.py --pdf documents/manual.pdf
# OR
python agent_rag.py --audio recordings/meeting.mp3
```

### 2. Query a question
```bash
python agent_rag.py --query "What is the return policy for online purchases?"
```