import os
import pymupdf4llm
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. SETUP: Load keys and initialize the "Brain"
load_dotenv()
llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def process_lecture_pdf(file_path):
    print(f"--- Processing: {file_path} ---")
    
    # 2. EXTRACTION: Convert PDF to Markdown (extracts text and image descriptions)
    # PyMuPDF4LLM is smart—it identifies where images are located
    md_text = pymupdf4llm.to_markdown(file_path)
    
    # 3. CHUNKING: Break it into 1000-character pieces with overlap
    # Overlap ensures we don't cut a sentence in half
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.create_documents([md_text])
    
    # 4. STORAGE: Save to ChromaDB using free HuggingFace embeddings
    print("Building the knowledge base...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    return vector_db.as_retriever()

def generate_lecture_summary(retriever):
    # 5. GENERATION: Ask the AI to summarize based ONLY on the retrieved context
    query = "Provide a comprehensive summary of this lecture, including key points and any visual diagrams mentioned."
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"Using only the following lecture content, create a professional summary:\n\n{context}"
    summary = llm.invoke(prompt)
    return summary.content

# RUN THE APP
if __name__ == "__main__":
    # Ensure you have a file named 'lecture.pdf' in your folder
    lecture_retriever = process_lecture_pdf("lecture.pdf")
    final_summary = generate_lecture_summary(lecture_retriever)
    
    print("\n--- LECTURE SUMMARY ---")
    print(final_summary)