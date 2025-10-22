from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
from langchain_core.messages import  HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.summarize import load_summarize_chain
from sqlite_db import fetch_messages, insert_message, fetch_last_user_question
import vectorDB
from load import load_and_chunk_local_pdf
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- SETUP ----------
# pdf_path = "story.pdf"
# docs = load_and_chunk_local_pdf(pdf_path)
# retriever = vectorstore(docs)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
template = """
You are an intelligent and reliable QA assistant designed to answer questions using all available information from the provided document context and chat history.

Your goals:
1. Carefully review both the context and the chat history before generating an answer.
2. Use all relevant information from the context and chat history to provide accurate, clear, and complete responses.
3. Never fabricate information. If the answer cannot be found or reasonably inferred from the context or chat history, reply exactly with:
   "I'm sorry, I don't know."
4. Maintain a polite, professional, and helpful tone in all responses.

Guidelines:
- Keep answers extremely brief, ideally **2–3 lines maximum**.
- Summarize multiple relevant points into **one short paragraph**; avoid bullet points unless explicitly asked.
- Incorporate prior user interactions from the chat history only if it improves clarity.
- Avoid repeating the question or obvious context.
- Do not provide extra details beyond what is necessary to answer the question.

---

Chat History:
{history}

Context (from documents):
{context}

User Question:
{question}

"""

prompt = ChatPromptTemplate.from_template(template)
rag_chain = prompt | llm

summarize_prompt_template = """Write a detailed summary of the following text, aiming for approximately 400 words.

Text:
{text}

Summary:"""
summarize_prompt = ChatPromptTemplate.from_template(summarize_prompt_template)

# ---------- STATE ----------
class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    docs: List[Document]
    # question: str
    proceed_to_generate: bool
    question: str
    on_topic : str
    ai_response : AIMessage


class GradeQuestion(BaseModel):# pydentic model to get structure output 
            score: str = Field(
                description="Question is about the specified topics? If yes -> 'Yes' if not -> 'No'"
            )

        # this function refine user question
# def question_rewriter(state: AgentState):
#             # print(f"Entering question_rewriter with following state: {state}")

#             # Reset state variables except for 'question' and 'messages'
#             state["documents"] = []
#             state["question"] = ""
#             state["proceed_to_generate"] = False
#             state["on_topic"] = ""
#             state["ai_response"] = None

#             if "messages" not in state or state["messages"] is None:
#                 state["messages"] = []

#             if state["question"] not in state["messages"]:
#                 state["messages"].append(state["question"])

#             if len(state["messages"]) > 1:
#                 conversation = state["messages"][:-1]
#                 current_question = state["question"].content
#                 messages = [
#                     SystemMessage(
#                         content="You are a helpful assistant that rephrases the user's qustion."
#                     )
#                 ]
#                 messages.extend(conversation)
#                 messages.append(HumanMessage(content=current_question))
#                 rephrase_prompt = ChatPromptTemplate.from_messages(messages)
#                 # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
#                 prompt = rephrase_prompt.format()
#                 response = llm.invoke(prompt)
#                 better_question = response.content.strip()
#                 # print(f"question_rewriter: Rephrased question: {better_question}")
#                 state["question"] = better_question
#             else:
#                 state["question"] = state["question"].content
#             return state
        # this function classify wheather the user question is related to pdf context or not
def question_classifier(state: AgentState):
            # print("Entering question_classifier")
            state["ai_response"] = None
            system_message = SystemMessage(
            content="""You are a precise and helpful assistant. Your task is to classify the user's question into **one of three categories**:

- **on_topic**: The question clearly pertains to the main content, topics, or context of the provided PDF document. Examples: "Summarize the PDF", "What are the key points of the document?", "Explain the main story in the PDF."

- **off_topic**: The question is unrelated to the main content of the PDF. Examples: "Tell me a joke", "What is the weather today?", "How are you?"


**Instructions:**  
1. Consider only the user's question and the PDF content context.  
2. Respond strictly with **one word**: `on_topic` or `off_topic`.  
3. Do not provide explanations or additional text.
"""

)

            doc = vectorDB.global_retriever.invoke(state["question"])
            # print(f"Rephrased Question: {state['question']}")
            # print(f"Question Classifier: Retrieved Document Content: {[d.page_content for d in doc]}")
            human_message = HumanMessage(
                    content=f"User question: {state['question']}\n\nRetrieved document:\n{[d.page_content for d in doc]}"
                )
            grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
            llm_classifier = llm | StrOutputParser()
            classification = llm_classifier.invoke(grade_prompt.format_messages())
            state["on_topic"] = classification.strip().lower()
            print(f"Question Classifier: Classification result: {state['on_topic']}")
            return state
        # this function will router the to retrieve node if question is related to pdf or return to off topic node
def on_topic_router(state: AgentState):
    print("Entering on_topic_router")
    category = state.get("on_topic", "").strip().lower()
    if category == "on_topic":
        print("Routing to retrieve")
        return "retrieve"
    else :
        print("Routing to off_topic_response")
        return "off_topic_response"


        # this funtion or node will get all the chunks from vectore database and store into the State
def retrieve(state: AgentState):
            question = state["question"]

            # Step 1: Let LLM classify the query type
            classifier_prompt = """
        You are a smart query router. Based on the question, classify it into:
        - summary: for full document explanations or overviews.
        - ending : for questions about the end/final part of the document.
        - chunk : for specific or factual questions needing retrieval.

        Return only one of: summary, ending, or chunk.
        Question: {}
        """.format(question)
            # groqllm = ChatGroq(model="llama-3.1-8b-instant") 
            classification = llm.invoke(classifier_prompt).content.strip().lower()
            print(f"Retrieve Function: Query Classification: {classification}")
            if classification == "summary":
                print(f"Retrieve Function: Entering summary branch.")
                summary_prompt = """
                    You are a helpful and concise assistant. Your task is to summarize the following text clearly and accurately.
                    - Keep the summary brief but informative.
                    - Focus on the main points, key events, or important details.
                    - Avoid unnecessary repetition.
                    - If the text is from a story or narrative, keep the tone engaging and easy to understand.
                    - Write the summary in 2-4 sentences (or a few short paragraphs if needed for clarity).

                    Text to summarize:
                    {page_text}
                    """
                all_summaries = []
                # print(len(state["docs"]))
                for doc in state["docs"]:
                    page_text = doc.page_content
                    prompt = summary_prompt.format(page_text=page_text)
                    page_summary = llm.invoke(prompt).content
                    all_summaries.append(page_summary)

                # Combine all page summaries
                # combined_summary_text = "\n\n".join(all_summaries)

                # Optional: final refinement summary
                final_prompt = """
                You are a helpful assistant. Combine the following summaries into a single concise summary.
                Ensure clarity, coherence, and focus on the main points relevant to the user's question.

                User's question: "{user_question}"

                Summaries to combine:
                {combined_summaries}
"""
                prompt = final_prompt.format(
                    user_question=question, 
                    combined_summaries="\n\n".join(all_summaries)
                )
                final_summary = llm.invoke(prompt).content
                insert_message("AI", final_summary)
                state["ai_response"] = AIMessage(content=final_summary)
                # state["documents"] = [Document(page_content=final_summary)]

                print(f"Retrieve Function: Summarized {len(state["documents"])} documents.")
                return state

            elif classification == "ending":
                # print("-----ending------")
                # ending_text = " ".join([doc.page_content for doc in state["docs"][-5:]])
                # Wrap string in Document
                state["documents"] = state["docs"][-5:]
                print(f"Retrieve Function: Extracted ending from {len(state["documents"])} documents.")
                return state
            else:
                # Default chunk retrieval from FAISS
                documents = vectorDB.global_retriever.invoke(state["question"])
                state["documents"] = documents
                print(f"Retrieve Function: Retrieved {len(state["documents"])} documents.")
                return state


def generate_answer(state: AgentState):# this node will generate answer to the refine user question
            # print("Entering generate_answer")
            # if "messages" not in state or state["messages"] is None:
            #     raise ValueError("State must include 'messages' before generating an answer.")

            history = fetch_messages(limit=3)
            documents = state["documents"]
            question = state["question"]
            question_content = question
            context_text = "\n\n".join([doc.page_content for doc in documents])
            # print(f"Context passed to LLM: \n{context_text}\n")
            response = rag_chain.invoke(
                {"history": history, "context": context_text, "question": question_content}
            )
            # response = llm.invoke(prompt_text)

            generation = response.content.strip()
            insert_message("AI", generation)
            print(f" Generated response: {generation}")
            state["ai_response"] = AIMessage(content=generation)
            # print(f"generate_answer: Generated response: {generation}")
            return state

def off_topic_response(state: AgentState):# if the question is off topic then this node will run
            # print("Entering off_topic_response")
            # if "messages" not in state or state["messages"] is None:
            #     raise ValueError("State must include 'messages' before generating an answer.")

            # history = fetch_messages(limit=3)
            # documents = state["documents"]
#             question = state["question"]
#             prompt_text =f"""- you are a simple chatting AI.
#             - chat with user in a friendly manner.
# - Keep your answer concise (2-3 sentences) unless the context clearly requires more detail.
# - Conversation history:\n{history}
# - If u not get good context from history then you can reply accordingly.
# - User: {question}"""

#             response = llm.invoke(
#                 prompt_text
#             )

#             generation = response.content.strip()
#             insert_message("AI", generation)
            state["ai_response"] = AIMessage(content="I'm sorry, I don't know. I only have information related to the uploaded documents. I not able to answer off-topic questions.")
            # print(f"generate_answer: Generated response: {generation}")
            return state

def after_retrieve_router(state: AgentState):
    if "ai_response" in state and state["ai_response"]:
        return "end"
    else:
        return "generate_answer"

checkpointer = MemorySaver()
        #structure of graph
workflow = StateGraph(AgentState)
        # added all nodes
# workflow.add_node("question_rewriter", question_rewriter)
workflow.add_node("question_classifier", question_classifier)
workflow.add_node("off_topic_response", off_topic_response)
workflow.add_node("retrieve", retrieve)
# workflow.add_node("after_retrieve_router", after_retrieve_router) # This was incorrect
# workflow.add_node("retrieval_grader", retrieval_grader)
workflow.add_node("generate_answer", generate_answer)
# workflow.add_node("refine_question", refine_question)
# workflow.add_node("cannot_answer", cannot_answer)


# workflow.add_edge("question_rewriter", "question_classifier")
workflow.add_conditional_edges(
            "question_classifier",
            on_topic_router,
            {
                "retrieve": "retrieve",
                "off_topic_response": "off_topic_response",
            },
        )
workflow.add_conditional_edges(
    "retrieve",
    after_retrieve_router,
    {
        "end": END,
        "generate_answer": "generate_answer",
    }
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("off_topic_response", END)
# workflow.set_entry_point("question_rewriter")
workflow.set_entry_point("question_classifier")
graph = workflow.compile(checkpointer=checkpointer)

# Example usage:
# input_data = {"question": HumanMessage(content="hiii")}
# response = graph.invoke(input=input_data, config={"configurable": {"thread_id": 1}})
# print(response["messages"][-1].content)
