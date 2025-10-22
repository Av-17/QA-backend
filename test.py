from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

# llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
result = llm.invoke("Hello, iam avdhesh")
print(result.content)