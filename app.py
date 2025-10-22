from flask import Flask, request, jsonify, render_template
from main import final_response_chain, process_user_query, load_system_prompt
import logging
from utils.classifier import classify_query
import os

app = Flask(__name__)

# Get port from environment variable (Cloud Run compatibility)
PORT = int(os.environ.get('PORT', 5002))

SYSTEM_PROMPT_PATH_FOR_RETRIEVAL = "system_prompts/response_generation_prompt.txt"

# In-memory storage for chat sessions
chat_sessions = {}

logging.basicConfig(level=logging.DEBUG)

# Home route for frontend
@app.route('/')
def home():
    return render_template('index.html')

# Health check endpoint for Cloud Run
@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

# API route to handle chatbot queries
@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    # Get the user's message and session ID from the request
    user_query = request.json.get("message")
    session_id = request.json.get("session_id")

    if not user_query:
        return jsonify({"error": "No message provided."}), 400

    print(f"User Message: {user_query}, Session ID: {session_id}")

    # Check if session_id exists in chat_sessions (in-memory storage)
    if session_id and session_id in chat_sessions:
        logging.info(f"Session found with session_id: {session_id}")
        chat_data = chat_sessions.get(session_id)
        chat_chain = chat_data['chat_chain']
        chat_history = chat_data['chat_history']
    else:
        # If session not found, initialize a new session
        logging.info(f"Session ID not found, initializing new session for session_id: {session_id}")
        system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH_FOR_RETRIEVAL)
        chat_chain, chat_history = final_response_chain(system_prompt)
        
        # Store the new session data in memory
        chat_sessions[session_id] = {
            "chat_chain": chat_chain,
            "chat_history": chat_history
        }

    try:
        # Process the user query and generate a response
        final_response = process_user_query(user_query, chat_chain, chat_history)
        print(f"************Final Response - {final_response}**************")
        
        # Update the session with the latest chat history
        chat_sessions[session_id] = {
            "chat_chain": chat_chain,
            "chat_history": chat_history
        }

        return jsonify({"response": final_response})
    except Exception as e:
        # Log any error and send a response
        logging.error(f"Error processing user query: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Use PORT from environment variable for Cloud Run
    app.run(host="0.0.0.0", port=PORT, debug=False)
