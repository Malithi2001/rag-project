import os
import pymupdf4llm
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from rich.console import Console
from rich.markdown import Markdown

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
    query = "Provide a comprehensive summary of this lecture."
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # We define a strict Markdown template here
    template = f"""
    You are a professional academic assistant. Use the following context to create a summary.
    FORMATTING RULES:
    1. Use a clear # Title.
    2. Use ## for main sections.
    3. Use bullet points for key takeaways.
    4. Use **bold** for important terms.
    5. If there are diagrams described, mention them in a > blockquote.

    CONTEXT:
    {context}

    FINAL SUMMARY:
    """
    
    summary = llm.invoke(template)
    return summary.content

# RUN THE APP
if __name__ == "__main__":
    # Ensure you have a file named 'lecture.pdf' in your folder
    lecture_retriever = process_lecture_pdf("lecture.pdf")
    final_summary = generate_lecture_summary(lecture_retriever)
    console = Console()
    console.print(Markdown(final_summary))
    
    print("\n--- LECTURE SUMMARY ---")
    print(final_summary)