# Arabic-English RAG Chatbot POC

A proof-of-concept chatbot that provides intelligent responses based on Royal Decree documents in both Arabic and English. The system uses Retrieval-Augmented Generation (RAG) to answer domain-specific queries and handles general conversations.

## 🌟 Features

- **Bilingual Support**: Handles queries in both Arabic and English
- **Intelligent Query Classification**: Automatically routes queries to appropriate handlers
- **RAG Pipeline**: Retrieves relevant context from document database for accurate responses
- **Conversational Memory**: Maintains chat history for contextual conversations
- **GPU Acceleration**: Leverages NVIDIA GPUs for embeddings and processing
- **Web Interface**: Flask-based API with session management

## 🏗️ Architecture

```
User Query
    ↓
Query Classifier (Groq LLM)
    ↓
├─→ Class 1 (Domain-Specific)      ├─→ Class 2 (General Chat)
    ↓                                   ↓
    FAISS Vector Search                 Direct LLM Response
    ↓
    Context Retrieval
    ↓
    RAG Response Generation (Groq LLM)
    ↓
Response to User
```

## 📋 Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support
- CUDA 12.1+ installed
- 8GB+ GPU memory recommended

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd compressed_arabic_chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv env
env\Scripts\activate  # Windows
# source env/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install flask python-dotenv
pip install langchain langchain-community langchain-core langchain-groq
pip install sentence-transformers faiss-cpu
pip install transformers huggingface-hub
pip install pandas
```

### 4. Set Up API Keys

In the main.py, add your Groq API key:
```
groq_api_key = "<Add Groq API Key here>"
```

Get your Groq API key from: https://console.groq.com/

## 📁 Project Structure

```
compressed_arabic_chatbot/
│
├── app.py                          # Flask application entry point
├── main.py                         # Core chatbot logic and RAG pipeline
│
├── data/                           # Document storage
│   ├── Arabic(Royal_Decree).csv    # Arabic documents
│   └── English(Royal_Decree).csv   # English documents
│
├── system_prompts/                 # System prompts for LLM behavior
│   ├── new_classifier.txt          # Query classification prompt
│   ├── response_generation_prompt.txt  # RAG response prompt
│   └── handling_general_responses.txt  # General chat prompt
│
├── utils/                          # Utility modules
│   ├── classifier.py               # Query classification logic
│   └── postprocessors.py           # Response extraction utilities
│
├── v_database/                     # FAISS vector store (auto-generated)
│
├── templates/                      # HTML templates for web interface
│   └── index.html
│
└── .env                           # Environment variables (not in repo)
```

## 🗄️ Data Format

CSV files should have the following columns:

```csv
title,language,keywords,category,content
"Document Title","English","keyword1, keyword2","Policy","Full document content..."
```

## ⚙️ Configuration

### Key Parameters in `main.py`:

```python
DATA_PATH = "./data"                    # Document folder
DB_FAISS_PATH = "v_database"           # Vector database path
EMBEDDING_MODEL = "omarelshehy/arabic-english-sts-matryoshka-v2.0"
LLM_MODEL = "llama-3.3-70b-versatile"     # Groq model
```

### Groq API Configuration:
```python
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)
```

## 🐳 Docker Setup
### Build Docker Image
```
docker build -t arabic-chatbot .
```

### Run Docker Container Locally
```
docker run -p 8080:8080 -e GROQ_API_KEY=your_groq_api_key arabic-chatbot
```

Access the chatbot at: http://localhost:8080
Test Docker Container
```
curl -X POST http://localhost:8080/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "session_id": "test"}'
```


## Google Cloud Deployment
### Prerequisites
Google Cloud Account: Create at https://console.cloud.google.com
Install gcloud CLI: https://cloud.google.com/sdk/docs/install
Enable Billing: Required for Cloud Run

#### Step 1: Initialize gcloud
```
gcloud auth login
```
```
gcloud config set project YOUR_PROJECT_ID
```

```
gcloud config set run/region us-central1
```

### Step 2: Enable Required APIs
```
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com
```

### Step 3: Build and Push Docker Image
```
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/arabic-chatbot
```

### Or use Docker
```
docker build -t gcr.io/YOUR_PROJECT_ID/arabic-chatbot .
```
```
docker push gcr.io/YOUR_PROJECT_ID/arabic-chatbot
```

### Step 4: Deploy to Cloud Run
```
gcloud run deploy arabic-chatbot --image gcr.io/YOUR_PROJECT_ID/arabic-chatbot --platform managed --region us-central1 --allow-unauthenticated --memory 4Gi --cpu 2 --timeout 300 
```

Step 5: Get Service URL
```
gcloud run services describe arabic-chatbot \
  --region us-central1 \
  --format 'value(status.url)'
```
  
## 🎯 Usage

### Run the Web Application

```bash
python app.py
```

Access the chatbot at: `http://localhost:5002`

### Test via Command Line

Add this to the bottom of `main.py`:

```python
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in ["exit", "quit"]:
            break
        answer = conversation(user_input)
        print("Bot:", answer)
```

Then run:
```bash
python main.py
```

### Test via API

```bash
curl -X POST http://localhost:5002/chatbot \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What are the environmental policies?\", \"session_id\": \"test123\"}"
```

## 🔍 How It Works

### 1. Query Classification
- User query is sent to Groq LLM with classification prompt
- Returns `class_1` (domain-specific) or `class_2` (general chat)

### 2. Class 1 Flow (RAG)
```python
User Query 
  → FAISS Similarity Search 
  → Retrieve Top-K Documents 
  → Format Context with System Prompt 
  → Generate Response with LLM 
  → Return Answer
```

### 3. Class 2 Flow (General)
```python
User Query 
  → Direct LLM Call with General Prompt 
  → Generate Response 
  → Return Answer
```

### 4. Session Management
- Each user gets a unique `session_id`
- Chat history maintained per session
- Context preserved across conversations

## 🧪 Example Queries

**Domain-Specific (Class 1):**
- "What are the environmental policies on waste management?"
- "ما هي السياسات البيئية؟" (Arabic)
- "Tell me about Royal Decree regulations"

**General Chat (Class 2):**
- "Hello, how are you?"
- "Thank you for your help"
- "What can you do?"

## 🛠️ Troubleshooting

### PyTorch DLL Error
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CUDA Out of Memory
Reduce batch size or use CPU for embeddings:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="omarelshehy/arabic-english-sts-matryoshka-v2.0",
    model_kwargs={'device': 'cpu'}
)
```

### FAISS Not Loading
```bash
pip install faiss-cpu --force-reinstall
```

## 📊 Performance

- **Embedding Model**: ~500ms per query (GPU)
- **Vector Search**: ~50ms for similarity search
- **LLM Response**: ~2-5s (Groq API)
- **Total Response Time**: ~3-6s per query

## 🔐 Security Notes

- **API Keys**: Never commit `.env` file to version control
- **Input Validation**: Add sanitization for production use
- **Rate Limiting**: Implement rate limits for API endpoints
- **CORS**: Configure properly for production deployment

## 🚧 Known Limitations

1. **Local Model Support**: Currently using Groq API instead of local Llama model
2. **Single Language Output**: Responses are primarily in English
3. **Context Window**: Limited to retrieved documents (no full conversation context in RAG)
4. **Scalability**: In-memory session storage (use Redis for production)


## 📝 API Reference

### POST /chatbot

**Request:**
```json
{
  "message": "Your query here",
  "session_id": "unique-session-id"
}
```

**Response:**
```json
{
  "response": "Chatbot response here"
}
```

**Error Response:**
```json
{
  "error": "Error message"
}
```

