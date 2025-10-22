from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
from typing import List
from sqlite_db import insert_document
from fastapi import UploadFile # Ensure UploadFile is imported

def load_and_chunk_uploaded_pdfs(uploaded_files: List[UploadFile], chunk_size: int = 500, chunk_overlap: int = 50):
    all_docs = []

    for uploaded_file in uploaded_files:
        # Read the file content ONCE
        file_content_bytes = uploaded_file.file.read()

        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content_bytes) # Use the stored content
            tmp_path = tmp_file.name

        # Load the temporary PDF
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()
        content = "\n".join([doc.page_content for doc in documents])
        insert_document(uploaded_file.filename, content)
        print(f"📄 Stored {uploaded_file.filename} in database")
        all_docs.extend(documents)

        # Remove the temporary file after loading
        os.remove(tmp_path)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs_chunks = text_splitter.split_documents(all_docs)

    return docs_chunks, documents

def load_and_chunk_local_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs_chunks = text_splitter.split_documents(documents)
    return docs_chunks
