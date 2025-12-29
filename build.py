import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # make sure env set hai

def get_pdf_text(pdf_paths):
    text = ""
    for path in pdf_paths:
        pdf_reader = PdfReader(path)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)

def build_faiss_index(pdf_paths):
    print("Reading PDFs...")
    raw_text = get_pdf_text(pdf_paths)
    print(f"Total text length: {len(raw_text)}")

    print("Splitting into chunks...")
    chunks = get_text_chunks(raw_text)
    print(f"Total chunks: {len(chunks)}")

    print("Creating embeddings and FAISS index...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    vector_store.save_local("faiss_index")
    print("Index saved to ./faiss_index")

if __name__ == "__main__":
    # yahan apne PDF paths daal
    pdf_files = [
        "docs/manual1.pdf",
        "docs/manual2.pdf"
    ]
    build_faiss_index(pdf_files)
