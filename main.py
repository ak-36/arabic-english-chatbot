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
from utils.classifier import classify_query
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM


# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables if needed
load_dotenv()

DATA_PATH = "./data"
DB_FAISS_PATH = "v_database"
SYSTEM_PROMPT_PATH_FOR_RETRIEVAL = "system_prompts/response_generation_prompt.txt"
SYSTEM_PROMPT_PATH_FOR_GENERAL_RESPONSE="system_prompts/handling_general_responses.txt"


def load_system_prompt(system_prompt_path):
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as file:
            system_prompt = file.read()
        return system_prompt
    except FileNotFoundError:
        print(f"Error: System prompt file '{system_prompt_path}' not found.")
        sys.exit(1)

system_prompt_for_general_response = load_system_prompt(SYSTEM_PROMPT_PATH_FOR_GENERAL_RESPONSE)

#*********************RAG Set-up****************************
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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name="omarelshehy/arabic-english-sts-matryoshka-v2.0"
)

if not os.path.exists(DB_FAISS_PATH):
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
else:
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever()

LOCAL_MODEL_PATH = "./local_models/meta-llama/Llama-3.1-8B-Instruct"

# Load model and tokenizer from the local directory
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, use_fast=False)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

llm = HuggingFacePipeline(pipeline=pipe)

retrieval_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {input}
Only return the helpful answer below and nothing else.
Helpful answer:"""
retrieval_prompt = ChatPromptTemplate.from_template(retrieval_prompt_template)

document_chain = create_stuff_documents_chain(llm, retrieval_prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


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

def chatbot_workflow(user_query: str, chat_chain) -> str:
    classification = classify_query(user_query)
    user_query = classification['Query']
    query_class = classification['Class']

    if query_class == "class_1":
        try:
            print(f"User query received: {user_query}")
            input_payload = {"input": user_query}
            print(f"Payload to retrieval chain: {input_payload}")
            response = retrieval_chain.invoke(input_payload)
            print(f"************** Response from Chain: {response} ************")
            
            
            # Ensure 'answer' exists in the response
            if 'answer' not in response:
                raise ValueError("Key 'answer' not found in response.")
            
            final_response = chat_chain.invoke(
                {"input": response['answer']},
                {"configurable": {"session_id": "unused"}}
            )
            return final_response.content
        except Exception as e:
            print(f"Error during retrieval_chain.invoke: {e}")
            return "Error occurred while processing the query."
      
    else:
        try:
            print(f"User Query: {user_query}")
            combined_input = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

            {system_prompt_for_general_response}<|eot_id|><|start_header_id|>user<|end_header_id|>

            {user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            print(f"Combined Input Sent to Pipeline:\n{combined_input}")
            response = pipe(
                combined_input,
                truncation=True,
                max_length=900,  # Set a reasonable max length for output
                pad_token_id=tokenizer.eos_token_id,  # Explicitly set pad_token_id
                eos_token_id=tokenizer.eos_token_id,  # Set stopping criteria
                do_sample=True,  # Enable sampling for diverse responses
                temperature=0.7  # Control randomness (lower for more focused output)
            )
            print("General Response from LLama: ", response)  


            return response[0]['generated_text']
            
        except Exception as e:
            print(f"Error during general response generation: {e}")
            return "Error occurred while processing the query."
        

def process_user_query(user_query: str, chat_chain, chat_history) -> str:
  
    pipeline_response = chatbot_workflow(user_query, chat_chain)
    print(f"Response from Pipeline: {pipeline_response}")

    print(f"Final Response from process_user_query function: {pipeline_response}")

 
    chat_history.add_user_message(user_query)
    chat_history.add_ai_message(pipeline_response)

    return pipeline_response



# For testing main.py
#def conversation(user_query):
    # Initialize the system prompt and final response chain
#    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH_FOR_RETRIEVAL)
 #   chat_chain, chat_history = final_response_chain(system_prompt)

    # Process the user query
#    final_response = process_user_query(user_query, chat_chain, chat_history)

    # Return the final content
#    print("\nFinal Response Done")
 #   return final_response.content


# # Example usage (uncomment to run interactively):
# while True:
#     user_input = input("You: ")
#     if user_input.lower().strip() in ["exit", "quit"]:
#         break
#     answer = conversation(user_input)
#     print("Bot:", answer)