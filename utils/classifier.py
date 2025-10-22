from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import re
from main import llm as chat

def query_classification_chain(system_prompt):
    """Set up a classification chain with system prompt."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    chain = prompt | chat
    return chain


def chain_setup():
    """Load system prompt and set up the chain."""
    try:
        # Load the system prompt
        with open('system_prompts/new_classifier.txt', 'r', encoding='utf-8') as file:
            system_prompt = file.read()
        print("System Prompt Loaded")

        # Set up the chain
        chain = query_classification_chain(system_prompt)
        return chain

    except Exception as e:
        print(f"Error occurred during chain setup: {str(e)}")
        return None
def output_parser(response_text: str) -> dict:
    """
    Parse the response text to extract the Query and Class.
    Expected format:
    Query: "...."
    Class: "...."
    """
    try:
        query_pattern = r'Query:\s*"(.*?)"'
        class_pattern = r'Class:\s*"(.*?)"'

        query_match = re.search(query_pattern, response_text)
        class_match = re.search(class_pattern, response_text)

        if query_match and class_match:
            return {
                "Query": query_match.group(1),
                "Class": class_match.group(1)
            }
        else:
            raise ValueError("Response format is incorrect.")
    except Exception as e:
        print(f"Error during parsing: {e}")
        return {"Query": None, "Class": None}


def classify_query(user_query: str):
    """Classify the user query using the classification chain."""
    try:
        # Set up the chain if not already initialized
        if 'chat_chain' not in globals():
            global chat_chain
            chat_chain = chain_setup()
            if not chat_chain:
                raise ValueError("Chain setup failed.")

        # Invoke the chain with the user query
        print("Classifying query...")
        response = chat_chain.invoke({"input": user_query})
        print(f"LLM Response: {response.content.strip()}")

        

        parsed_output = output_parser(response.content.strip())
        print("parsed output:", parsed_output)
        return parsed_output
        

    except Exception as e:
        print(f"Error occurred during classify_query: {str(e)}")
        return "Error: Classification failed."


# Example Usage
if __name__ == "__main__":
    # Example queries
    test_queries = [
        "What are the environmental policies on waste management?",
        "Hello!",
        "Thank you for your help!",
        "I hate this chatbot.",
        "How can I apply for an environmental permit?"
    ]

    for query in test_queries:
        print(f"User Query: {query}")
        result = classify_query(query)
        print(f"Classification: {result}\n")
