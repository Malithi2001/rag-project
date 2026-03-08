import streamlit as st
import os
import pymupdf4llm
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 1. Setup the Page Layout
st.set_page_config(page_title="Lecture Summarizer AI", layout="centered")
st.title("🎓 Lecture PDF Summarizer")
st.markdown("Upload your lecture notes and get an AI-powered summary in seconds.")

load_dotenv()

# 2. Sidebar for Configuration
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload Lecture PDF", type="pdf")
    
    if uploaded_file:
        # Save the uploaded file locally so our parser can read it
        with open("temp_lecture.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("PDF Uploaded!")

# 3. Processing & Summarization Logic
if uploaded_file:
    if st.button("✨ Generate Summary"):
        with st.status("Analyzing document...", expanded=True) as status:
            # Step A: Extract
            st.write("Extracting text and images...")
            md_text = pymupdf4llm.to_markdown("temp_lecture.pdf")
            
            # Step B: Chunk
            st.write("Splitting content into chunks...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.create_documents([md_text])
            
            # Step C: Embed & Store
            st.write("Building knowledge base...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_db = Chroma.from_documents(chunks, embeddings)
            
            # Step D: Summarize
            st.write("Generating summary with Llama 3.1...")
            llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
            context = "\n\n".join([doc.page_content for doc in vector_db.as_retriever().invoke("summary")])
            
            prompt = f"Create a structured academic summary of this lecture:\n\n{context}"
            response = llm.invoke(prompt)
            
            status.update(label="Process Complete!", state="complete", expanded=False)

        # 4. Display the Final Result
        st.divider()
        st.markdown("### 📝 Lecture Summary")
        st.markdown(response.content)
else:
    st.info("👈 Please upload a PDF in the sidebar to get started.")