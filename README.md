🎓 AI-Powered Lecture PDF Summarizer (Multimodal RAG)
A professional Retrieval-Augmented Generation (RAG) application built to automate the summarization of complex lecture notes and technical documents.

This project was developed as part of an IT Internship, focusing on building private, local knowledge bases for sensitive documents.

Key Features
Multimodal Extraction: Uses PyMuPDF4LLM to convert PDFs into structured Markdown, preserving headers and layout.

Local Vector Store: Implements ChromaDB as a local database to store document "embeddings" privately.

Semantic Search: Uses HuggingFace all-MiniLM-L6-v2 to mathematically understand and retrieve relevant text chunks.

High-Speed Inference: Powered by Llama 3.1-8b via the Groq API for lightning-fast academic summaries.

Interactive UI: A modern web interface built with Streamlit for seamless file uploads and report generation.

Tech Stack
Orchestration: LangChain

LLM Provider: Groq (Llama 3.1)

Embeddings: HuggingFace Transformers

Database: ChromaDB (Vector Store)

Frontend: Streamlit

PDF Engine: PyMuPDF4LLM

The RAG Pipeline
Ingestion: The system parses the PDF and breaks it into 1000-character chunks with a 150-character overlap to preserve context.

Vectorization: Text chunks are converted into numerical vectors (embeddings).

Storage: Vectors are indexed in a local chroma_db folder.

Retrieval: When a summary is requested, the system retrieves the most relevant context from the database.

Generation: The AI synthesizes a structured summary based only on the retrieved facts.

How to Run
1. Setup Environment
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt

2. Configure API Key
    Create a .env file in the root directory and add:
    GROQ_API_KEY=your_actual_key_here

3. Launch the Application
    streamlit run web_app.py

Security & Privacy
    Zero Hardcoding: All API credentials are managed via environment variables.
    Data Privacy: The knowledge base remains local on the machine in the ./chroma_db directory.ss