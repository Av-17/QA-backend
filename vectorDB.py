from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings



global_retriever = None
def vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="mmr", search_kwargs = {"k": 10})
    global global_retriever
    global_retriever = retriever   
    return retriever

# retriever = vectorstore()
# print(retriever.invoke("who was jinger"))