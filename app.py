from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the PDF
print("Loading PDF...")
loader = PyPDFLoader("data.pdf")
pages = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

# 3. Create Free Local Embeddings (No API Key needed!)
print("Initializing free HuggingFace embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Create the Vector Database
print("Building the database...")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

# 5. Test a Search (Retrieval)
query = "What is this document about?"
docs = vector_db.similarity_search(query)

print("\n--- Top Relevant Result Found ---")
print(docs[0].page_content)