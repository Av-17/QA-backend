from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from fastapi.concurrency import run_in_threadpool
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from vectorDB import vectorstore
import vectorDB
from load import load_and_chunk_uploaded_pdfs  # use the upload-based version we discussed earlier
from main import graph  # import your workflow graph defined above
from sqlite_db import init, insert_message, fetch_messages, clear_messages, insert_document, clear_documents
from fastapi import Request
api = FastAPI(title="RAG Chatbot API")
origins = [  # React dev server
    "https://qa-frontend-iuiag85dt-avdhesh-prajapatis-projects.vercel.app/",
    "http://localhost:3000",
]


api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

global_docs = [] # Add this global variable

@api.on_event("startup")
async def startup_event():
    init()
    print("✅ SQLite database initialized")

@api.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        # Load and chunk uploaded PDFs
        docs, documents = load_and_chunk_uploaded_pdfs(files)
        print(f"✅ Loaded and chunked {len(docs)} documents")

        # Update global_docs
        global global_docs
        global global_documents
        global_documents = documents
        global_docs = docs
        # Create or update the vectorstore retriever
        vectorstore(docs)
    
        print("✅ Vector store updated")

        return {"status": "success", "chunks_stored": len(docs)}
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        return {"status": "error", "message": str(e)}


@api.post("/ask")
async def ask_question(question: str = Form(...)):
    if not vectorDB.global_retriever:
        return {"error": "Please upload PDFs first before asking questions."}

    try:
        insert_message("User", question)
        print(f"❓ Question received: {question}")

        input_data = {"question": question, "docs": global_documents}

        # Run blocking graph.invoke in threadpool
        response = graph.invoke(input=input_data, config={"configurable": {"thread_id": 1}})

        answer = response["ai_response"].content
        return {"question": question, "answer": answer}
    except Exception as e:
        print(f"❌ Error during question answering: {e}")
        return {"error": str(e)}


@api.get("/")
def home():
    return {"message": "Welcome to the RAG Chatbot API!"}

# @api.on_event("shutdown")
# def shutdown_event():
#     print("Shutting down API...")
#     clear_documents()
#     clear_messages()
#     print("✅ Cleared chat history and documents from database")

