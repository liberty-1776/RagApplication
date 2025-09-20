import os
import boto3
import streamlit as st
import shutil
from dotenv import load_dotenv
load_dotenv()

# API key fields (only show if needed)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

#To import langchain aws models
from langchain_aws import BedrockEmbeddings,ChatBedrock
from custom_models import DeepSeekChatModel


##Importing langchain groq models 
from langchain_groq import ChatGroq

##iporting langchain OpenAi model
from langchain_openai import ChatOpenAI


##import langchain Ollama models
from langchain_ollama import ChatOllama

## Data Ingestion and Transformatioin
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader

# Vector Embedding And Vector Store
from langchain_community.vectorstores import FAISS

## LLm Models prompts and chians
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA,ConversationalRetrievalChain

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime",region_name='ap-south-1')
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)



##Definging function to import all the diferetn different models
def get_openai_gpt5_llm(openai_api_key):
    ##returning the OPENAI GPT-5 Model
    return ChatOpenAI(model="gpt-5-2025-08-07",api_key=openai_api_key)

def get_openai_gpt5nano_llm(openai_api_key):
    ##returning the OPENAI GPT-5 nano Model
    return ChatOpenAI(model="gpt-5-nano-2025-08-07",api_key=openai_api_key)

def get_ollama_gemma3_llm():
    ##returning the Ollama Gemma3 Model
    return ChatOllama(model="gemma3:27b")

def get_groq_llama3_llm(groq_api_key):
    ##returning the Groq Llama 3 70b Model
    return ChatGroq(model_name="llama-3.3-70b-versatile",streaming=True,api_key=groq_api_key)


def get_groq_gpt_oss_llm(groq_api_key):
    ##returning the Groq GPT OSS 120b Model
    return ChatGroq(model_name="openai/gpt-oss-120b",streaming=True,api_key=groq_api_key)

def get_bedrock_nova_pro_llm():
    ##returning the Bedrock Amazon Nova Pro Model
    return ChatBedrock(client=bedrock,model_id="apac.amazon.nova-pro-v1:0",model_kwargs={"maxTokens": 512,'model_provider':"amazon"})


def get_bedrock_mistral_llm():
    ##returning the Bedrock Mistral Model
    return ChatBedrock(model_id="mistral.mistral-7b-instruct-v0:2",client=bedrock,max_tokens=512)

def get_bedrock_deepseek_llm():
    return DeepSeekChatModel(model_id="deepseek.v3-v1:0", max_tokens=512)

def get_bedrock_claude_llm():
    ##returning the Bedrock Anthropic Model
    return ChatBedrock(model="anthropic.claude-3-sonnet-20240229-v1:0",client=bedrock,model_kwargs={'max_tokens':512})

def get_bedrock_llama3_llm():
     ##returning the Bedrock Llama3 70b Model
    return ChatBedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock,model_kwargs={'max_gen_len':512})
    

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but atleast summarize with 
250 words with detailed explaantions.
If the provided context is not enough, just say you dont know, dont try to cook answer up

<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def data_ingestion(file_paths):
    documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    docs = text_splitter.split_documents(documents)
    return docs


def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")


def get_conversational_chain(llm, vectorstore_faiss):
    retriever = vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return qa_chain

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
    )
    answer=qa({"query":query})
    return answer['result']



def process_uploaded_files():
    """
    Callback invoked when the file_uploader value changes.
    Saves uploaded files to data/, deletes old files, then runs ingestion+FAISS build once.
    """
    uploaded = st.session_state.get("uploaded_files")
    if not uploaded:
        # Deleting everything in data/
        for old_file in os.listdir("data"):
            path = os.path.join("data", old_file)
            if os.path.isfile(path):
                os.remove(path)

        # Deleting FAISS index folder if exists
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")

        st.session_state["processed"] = False
        st.session_state["uploaded_filenames"] = []
        with st.sidebar:
            st.success("All files deleted. Vector store cleared.")
        return

    # normalize to list
    files = uploaded if isinstance(uploaded, list) else [uploaded]

    os.makedirs("data", exist_ok=True)

    # delete old files in data/
    for old_file in os.listdir("data"):
        path = os.path.join("data", old_file)
        if os.path.isfile(path):
            os.remove(path)

    # save new uploads to disk
    file_paths = []
    for uf in files:
        dest = os.path.join("data", uf.name)
        with open(dest, "wb") as out:
            out.write(uf.getbuffer())
        file_paths.append(dest)

    # process (showing spinner only during this upload/change event)
    with st.sidebar:
        with st.spinner("Processing uploaded files..."):
            docs = data_ingestion(file_paths)
            get_vector_store(docs)

    # mark state so we don't re-process unless the uploader changes
    st.session_state["processed"] = True
    st.session_state["uploaded_filenames"] = [f.name for f in files]
    with st.sidebar:
        st.success(f"{len(files)} file(s) uploaded and processed.")


def main():
    st.set_page_config(page_title="Chat PDF", page_icon="📄", layout="wide")


    with st.sidebar:
        st.subheader("⚙️ Model Selection")
        model_choice = st.selectbox(
            "Choose Model",
            ["Bedrock - Claude 3 Sonnet", 
            "Bedrock - Llama3-70B",
            "Bedrock - DeepSeek-V3",
            "Bedrock - Nova Pro",
            "Groq - GPT OSS 120b",
            "Groq - Llama3",
            "OpenAI - GPT 5",
            "OpenAI - GPT 5-nano",
            "Ollama - Gemma3"],
            key="model_choice"
        )

    st.header("Chat with PDF using AWS Bedrock")

    with st.sidebar:
        st.title("Upload PDF Files")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = None

        os.makedirs("data", exist_ok=True)

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="uploaded_files",
            on_change=process_uploaded_files
        )

        if st.session_state.get("uploaded_filenames"):
            st.markdown("**Processed files:**")
            for name in st.session_state["uploaded_filenames"]:
                st.write("- " + name)
        else:
            st.info("No files uploaded yet.")

    # Showing chat history using st.chat_message
    for q, a,m in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(f"**Model: {m}**\n\n{a}")

    #Input bar stays at the bottom
    user_question = st.chat_input("Ask a question from the PDFs...")
    if user_question:
        if not st.session_state.get("uploaded_filenames"):  # no files uploaded
            with st.sidebar:
                st.warning("⚠️ Please upload at least one PDF before asking questions.")
        else:
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.spinner(f"Processing your question with {model_choice}..."):
                faiss_index = FAISS.load_local(
                    "faiss_index",
                    bedrock_embeddings,
                    allow_dangerous_deserialization=True
                )
                if model_choice == "Bedrock - Claude 3 Sonnet":
                    llm = get_bedrock_claude_llm()
                elif model_choice=="Bedrock - Llama3-70B":
                    llm = get_bedrock_llama3_llm()
                elif model_choice=="Bedrock - DeepSeek-V3":
                    llm = get_bedrock_deepseek_llm()
                elif model_choice=="Bedrock - Nova Pro":
                    llm = get_bedrock_nova_pro_llm()
                elif model_choice=="Groq - GPT OSS 120b":
                    llm = get_groq_gpt_oss_llm(GROQ_API_KEY)
                elif model_choice=="Groq - Llama3":
                    llm = get_groq_llama3_llm(GROQ_API_KEY)
                elif model_choice=="OpenAI - GPT 5":
                    llm = get_openai_gpt5_llm(OPENAI_API_KEY)
                elif model_choice=="OpenAI - GPT 5-nano":
                    llm = get_openai_gpt5nano_llm(OPENAI_API_KEY)
                else:
                    llm = get_ollama_gemma3_llm()

                st.session_state.qa_chain = get_conversational_chain(llm, faiss_index)

                result = st.session_state.qa_chain({
                    "question": user_question,
                    "chat_history": st.session_state.chat_history
                })

                answer = result["answer"]

            with st.chat_message("assistant"):
                st.markdown(f"**Model: {model_choice}**\n\n{answer}")

            # Save into chat history
            st.session_state.chat_history.append((user_question, answer,model_choice)) 

if __name__ == "__main__":
    main()