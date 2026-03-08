# 📚 Lecture PDF Summarizer (Multimodal RAG)

An intelligent application that processes lecture PDFs to provide structured, academic summaries using **Retrieval-Augmented Generation (RAG)**.

## 🚀 Features
* **PDF Ingestion:** High-speed extraction of text and document structure using `PyMuPDF4LLM`.
* **Vector Storage:** Persistent local knowledge base using `ChromaDB`.
* **AI Engine:** Uses `Llama-3.1-8b` via Groq for lightning-fast summary generation.
* **Formatted Output:** Professional Markdown-styled reports displayed directly in the terminal.

## 🛠️ Tech Stack
* **Framework:** LangChain
* **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
* **Database:** ChromaDB
* **LLM Provider:** Groq (Llama 3.1)

## 📦 Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd rag_project