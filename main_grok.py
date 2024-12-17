import logging
import os
import glob
import time
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings as HFEmbeddings
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from langchain.llms import HuggingFacePipeline


# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables if needed
load_dotenv()

DATA_PATH = "./data"
DB_FAISS_PATH = "v_database"
SYSTEM_PROMPT_PATH = "system_prompts/response_generation_prompt.txt"

# Step 1: Load the CSV files
file_list = glob.glob(f"{DATA_PATH}/*.csv")
print(f"Files to process: {file_list}")

loader = DirectoryLoader(
    DATA_PATH,
    glob="*.csv",
    loader_cls=CSVLoader,
    loader_kwargs={
        "encoding": "utf-8",  # adjust if needed
    },
)

documents = loader.load()

print(f"Number of files loaded: {len(file_list)}")
print(f"Total records loaded: {len(documents)}")

# Step 2: Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
texts = text_splitter.split_documents(documents)

# Step 3: Create Embeddings & Vector Store
embeddings = HuggingFaceEmbeddings(
    model_name="omarelshehy/arabic-english-sts-matryoshka-v2.0"
)

if not os.path.exists(DB_FAISS_PATH):
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
else:
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever()

groq_api_key = ('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')

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="Llama3-8b-8192")

# Step 5: Prompt Template for the retrieval step
retrieval_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {input}
Only return the helpful answer below and nothing else.
Helpful answer:"""
retrieval_prompt = ChatPromptTemplate.from_template(retrieval_prompt_template)

# Create the retrieval chain (RAG step)
document_chain = create_stuff_documents_chain(llm, retrieval_prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Step 6: Define the final response chain that incorporates system prompt and chat history
def final_response_chain(system_prompt):
    # Creates a chain that uses system prompt, chat history, and user input
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm

    demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: demo_ephemeral_chat_history_for_chain,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return chain_with_message_history, demo_ephemeral_chat_history_for_chain

# Step 7: Load system prompt
def load_system_prompt():
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as file:
            system_prompt = file.read()
        return system_prompt
    except FileNotFoundError:
        print(f"Error: System prompt file '{SYSTEM_PROMPT_PATH}' not found.")
        sys.exit(1)

# Step 8: Chatbot workflow (RAG pipeline)
def chatbot_workflow(user_query: str) -> str:
    print(f"************** Invoking the Retrieval Chain! ************")
    try:
        print(f"User query received: {user_query}")
        input_payload = {"input": user_query}
        print(f"Payload to retrieval chain: {input_payload}")
        response = retrieval_chain.invoke(input_payload)
        print("************** Response from Chain:",response["answer"], "************")
        print(f"************** Context Retrieved:", response["context"], "************")
        
        # Ensure 'answer' exists in the response
        if 'answer' not in response:
            raise ValueError("Key 'answer' not found in response.")
        
        return response['answer']
    
    except Exception as e:
        print(f"Error during retrieval_chain.invoke: {e}")
        return "Error occurred while processing the query."


# Step 9: Process user query (combines pipeline and final response)
def process_user_query(user_query: str, chat_chain, chat_history) -> str:
    # Step 1: Get pipeline (RAG) response
    pipeline_response = chatbot_workflow(user_query)
    print(f"Response from RAG Pipeline: {pipeline_response}")


    # Step 2: Use the final response chain with chat history and pipeline response
    # Here we treat pipeline_response as the "input" to the final chain
    final_response = chat_chain.invoke(
        {"input": pipeline_response},
        {"configurable": {"session_id": "unused"}}
    )

    print(f"Final Response from process_user_query function: {final_response.content}")

    # Step 3: Update chat history
    chat_history.add_user_message(user_query)
    chat_history.add_ai_message(final_response.content)

    return final_response

# Step 10: Conversation entry point
def conversation(user_query):
    # Initialize the system prompt and final response chain
    system_prompt = load_system_prompt()
    chat_chain, chat_history = final_response_chain(system_prompt)

    # Process the user query
    final_response = process_user_query(user_query, chat_chain, chat_history)

    # Return the final content
    print("\nFinal Response Done")
    return final_response.content


# # Example usage (uncomment to run interactively):
# while True:
#     user_input = input("You: ")
#     if user_input.lower().strip() in ["exit", "quit"]:
#         break
#     answer = conversation(user_input)
#     print("Bot:", answer)
