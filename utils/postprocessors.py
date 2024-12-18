import re

def extract_response(output):
    """
    Extracts the 'Response' text after the last occurrence of the 'Response:' keyword.
    This ensures we ignore any earlier 'Response' mentions in the system prompt.
    """
    try:
        # Find all matches for 'Response: "..."'
        matches = re.findall(r'Response:\s*"(.*?)"', output, re.DOTALL)
        
        # If matches exist, return the last one
        if matches:
            return matches[-1].strip()
        else:
            return "No valid response found in the output."
    except Exception as e:
        print(f"Error during response extraction: {e}")
        return "Error processing the output."

def extract_combined_content(retrieved_documents):
    """
    Extracts the 'page_content' from a list of Document objects and combines them into a single text block.

    Args:
        retrieved_documents (list): List of Document objects.

    Returns:
        str: Combined content from all retrieved documents.
    """
    combined_content = []

    for doc in retrieved_documents:
        try:
            # Extract 'page_content' if it exists
            content = doc.page_content
            if content:
                combined_content.append(content.strip())
        except AttributeError:
            print(f"Document missing 'page_content': {doc}")

    # Combine all the content into a single string
    return "\n\n".join(combined_content)


def format_llama_prompt(system_prompt, user_query, context):
    """
    Formats the input using the LLaMA-3.1 prompt template.

    Args:
        system_prompt (str): The system instruction.
        user_query (str): The user input/query.
        context (str): Retrieved and combined document content.

    Returns:
        str: Formatted input string for LLaMA-3.1.
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

Context:
{context}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
