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
- **Frontend**: Streamlit
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
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add environment variables
Create a `.env` file in the root directory with the following keys:

```env
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
AWS_REGION=ap-south-1
```

### 4. Run locally
```bash
streamlit run app.py
```

The app will be available at <a href="http://localhost:8501" target="_blank">http://localhost:8501</a>.


### 5. Deploy on AWS Elastic Beanstalk

1. Initialize Elastic Beanstalk (only once):
   ```bash
   eb init
   ```
   - Select your region (e.g., `us-east-1`)  
   - Choose platform: **Docker running on 64bit Amazon Linux 2023**

2. Create an environment:
   ```bash
   eb create rag-env --single
   ```

3. Add environment variables:
   ```bash
   eb setenv OPENAI_API_KEY=sk-xxxx GROQ_API_KEY=gsk-xxxx AWS_REGION=ap-south-1
   ```

4. Deploy updates:
   ```bash
   eb deploy
   ```

Your app will be live at the Elastic Beanstalk URL
