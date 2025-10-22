import logging
import os
import glob
import time
import sys
from llama_index.core import VectorStoreIndex, Document
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface.llms import HuggingFacePipeline
from utils.classifier import classify_query
from utils.postprocessors import extract_response, extract_combined_content, format_llama_prompt
import pandas as pd

from langchain_groq import ChatGroq

groq_api_key = "<Add Groq API Key here>"

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",  # or "llama-3.3-70b-versatile"
    temperature=0.1
)
# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables if needed
load_dotenv()

DATA_PATH = "./data"
DB_FAISS_PATH = "v_database"
SYSTEM_PROMPT_PATH_FOR_RETRIEVAL = "system_prompts/response_generation_prompt.txt"
SYSTEM_PROMPT_PATH_FOR_GENERAL_RESPONSE="system_prompts/handling_general_responses.txt"
# Step 7: Load system prompt
def load_system_prompt(system_prompt_path):
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as file:
            system_prompt = file.read()
        return system_prompt
    except FileNotFoundError:
        print(f"Error: System prompt file '{system_prompt_path}' not found.")
        sys.exit(1)

# Step 1: Load the CSV files
file_list = glob.glob(f"{DATA_PATH}/*.csv")
print(f"Files to process: {file_list}")

# Loop through the CSV and create Document objects
documents = []
for file_name in os.listdir(DATA_PATH):
    if file_name.endswith(".csv"):  # Only process CSV files
        file_path = os.path.join(DATA_PATH, file_name)
        print(f"Processing file: {file_path}")
        
        # Load the CSV file into a DataFrame
        try:
            data = pd.read_csv(file_path, encoding="utf-8")
            for index, row in data.iterrows():
                title = row['title']
                langauge = row['language']  
                keywords = row['keywords']
                category = row['category']
                content = row['content']  # Assuming tags are comma-separated in the CSV

                    # Create a Document with metadata
                doc = Document(
                        text=content,
                        metadata={"title": title, "language": langauge, "keywords": keywords, "category": category}
                    )

                    # Add the document to the list
                documents.append(doc)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

print(f"Processed {len(documents)} documents from all CSV files.")



print(f"Number of files loaded: {len(file_list)}")
print(f"Total records loaded: {len(documents)}")


embeddings = HuggingFaceEmbeddings(
    model_name="omarelshehy/arabic-english-sts-matryoshka-v2.0"
)

if not os.path.exists(DB_FAISS_PATH):
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(DB_FAISS_PATH)
else:
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

system_prompt_for_general_response = load_system_prompt(SYSTEM_PROMPT_PATH_FOR_GENERAL_RESPONSE)

def final_response_chain(system_prompt):
    # Creates a chain that uses system prompt, chat history, and user input
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the user query based on the provided system prompt, context and chat history."),
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

def chatbot_workflow(user_query: str, chat_chain) -> str:
    classification = classify_query(user_query, llm)
    user_query = classification['Query']
    query_class = classification['Class']

    if query_class == "class_1":
        try:
            print(f"User query received: {user_query}")
            
            # Retrieve relevant documents
            retrieved_documents = db.similarity_search(user_query)
            print(f"Retrieved Documents: {retrieved_documents}")
            
            # Load system prompt and prepare context
            system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH_FOR_RETRIEVAL)
            postprocessed_documents = extract_combined_content(retrieved_documents)
            
            # Format input for ChatGroq
            llama_prompt = format_llama_prompt(
                system_prompt, user_query, postprocessed_documents
            )
            print(f"Formatted LLaMA Prompt:\n{llama_prompt}")
            
            # Use ChatGroq LLM directly with the chain
            final_response = chat_chain.invoke(
                {"input": f"{llama_prompt}"},
                {"configurable": {"session_id": "unused"}}
            )
            
            # Extract response content
            if hasattr(final_response, 'content'):
                processed_rag_response = final_response.content
            else:
                processed_rag_response = str(final_response)
                
            print(f"Post Processed RAG Response: {processed_rag_response}")
            
            return processed_rag_response
        except Exception as e:
            print(f"Error during retrieval: {e}")
            import traceback
            traceback.print_exc()
            return "Error occurred while processing the query."
        
    else:
        # For general queries (class_2)
        try:
            print(f"User Query (General): {user_query}")
            
            # Use ChatGroq directly for general responses
            messages = [
                {"role": "system", "content": system_prompt_for_general_response},
                {"role": "user", "content": user_query}
            ]
            
            # Invoke the LLM
            response = llm.invoke(user_query)
            
            # Extract content
            if hasattr(response, 'content'):
                processed_response = response.content
            else:
                processed_response = str(response)
                
            print(f"General Response: {processed_response}")
            return processed_response
            
        except Exception as e:
            print(f"Error during general response generation: {e}")
            import traceback
            traceback.print_exc()
            return "Error occurred while processing the query."


# Step 9: Process user query (combines pipeline and final response)
def process_user_query(user_query: str, chat_chain, chat_history) -> str:
    # Step 1: Get pipeline (RAG) response
    pipeline_response = chatbot_workflow(user_query, chat_chain)
    print(f"Response from Pipeline: {pipeline_response}")


    # Step 2: Use the final response chain with chat history and pipeline response
    # Here we treat pipeline_response as the "input" to the final chain
    

    print(f"Final Response from process_user_query function: {pipeline_response}")

    # Step 3: Update chat history
    chat_history.add_user_message(user_query)
    chat_history.add_ai_message(pipeline_response)

    return pipeline_response

# Step 10: Conversation entry point
def conversation(user_query):
    # Initialize the system prompt and final response chain
    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH_FOR_RETRIEVAL)
    chat_chain, chat_history = final_response_chain(system_prompt)

    # Process the user query
    final_response = process_user_query(user_query, chat_chain, chat_history)

    # Return the final content
    print("\nFinal Response Done")
    return final_response.content

