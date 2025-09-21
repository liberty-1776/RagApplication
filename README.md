# 📄 Chat with PDF (RAG Application)

A Streamlit-based web app that lets you **chat with your PDFs** using **Retrieval-Augmented Generation (RAG)**.  
It integrates multiple LLM providers (AWS Bedrock, OpenAI, Groq) and FAISS vector search for efficient question answering over documents.  

---

## 🚀 Features
- 📂 Upload one or multiple PDF documents  
- 🧩 Document chunking & vectorization with **Amazon Titan Embeddings (Bedrock)**  
- 🔍 Fast retrieval with **FAISS** vector store  
- 💬 Conversational interface powered by:  
  - **AWS Bedrock** (Claude 3 Sonnet, Llama 3, Nova Pro, DeepSeek V3)  
  - **OpenAI** (GPT-5, GPT-5 nano)  
  - **Groq** (Llama 3, GPT-OSS 120B)  
- 🖥️ Deployed on **AWS Elastic Beanstalk** with Docker  
- 🔐 Secure configuration with environment variables  

---

## 🛠️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)  
- **Backend**: Python, LangChain  
- **Vector Store**: FAISS  
- **LLMs**: AWS Bedrock, OpenAI, Groq  
- **Deployment**: AWS Elastic Beanstalk (Docker)  

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/liberty-1776/RagApplication.git
cd RagApplication

### 2. Install dependencies
```bash
pip install -r requirements.txt

### 3. Add environment variables
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
AWS_REGION=ap-south-1


### 4. Run locally
```bash
streamlit run app.py
The app will be available at