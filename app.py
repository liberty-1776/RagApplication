import os
import boto3
import streamlit as st
from langchain_ollama import ChatOllama
import shutil
## We will be suing Titan Embeddings Model To generate Embeddings
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock
from custom_models import DeepSeekChatModel,ChatGptModel

## Data Ingestion and Transformatioin
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader


# Vector Embedding And Vector Store
from langchain_community.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA,ConversationalRetrievalChain



## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime",region_name='ap-south-1')
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)


def get_gpt_oss_llm():
    return ChatBedrock(
    client=bedrock,
    model_id="openai.gpt-oss-120b-1:0", 
    model_kwargs={"max_completion_tokens": 512})

def get_bedrock_nova_pro_llm():
    return ChatBedrock(client=bedrock,model_id="apac.amazon.nova-pro-v1:0",model_kwargs={"maxTokens": 512,'model_provider':"amazon"})


def get_bedrock_mistral_llm():
    return ChatBedrock(model_id="mistral.mistral-7b-instruct-v0:2",client=bedrock,max_tokens=512)

def get_bedrock_deepseek_llm():
    return DeepSeekChatModel(model_id="deepseek.v3-v1:0", max_tokens=512)

def get_bedrock_claude_llm():
    ##create the Anthropic Model
    return ChatBedrock(model="anthropic.claude-3-sonnet-20240229-v1:0",client=bedrock,
                model_kwargs={'max_tokens':512})


def get_bedrock_llama3_llm():
    ##create the Anthropic Model
    return ChatBedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock,
                model_kwargs={'max_gen_len':512})
    

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
        # Delete everything in data/
        for old_file in os.listdir("data"):
            path = os.path.join("data", old_file)
            if os.path.isfile(path):
                os.remove(path)

        # Delete FAISS index folder if exists
        if os.path.exists("faiss_index"):
            import shutil
            shutil.rmtree("faiss_index")

        st.session_state["processed"] = False
        st.session_state["uploaded_filenames"] = []
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

    # process (show spinner only during this upload/change event)
    with st.spinner("Processing uploaded files..."):
        docs = data_ingestion(file_paths)
        get_vector_store(docs)

    # mark state so we don't re-process unless uploader changes
    st.session_state["processed"] = True
    st.session_state["uploaded_filenames"] = [f.name for f in files]
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
            "Bedrock - GPT OSS 120b",
            "Groq - GPT OSS 20b",
            "Groq - "
            "OpenAI - gpt-5o"],
            key="model_choice"
        )
    # API key fields (only show if needed)
    openai_api_key = None
    groq_api_key = None

    if "OpenAI" in model_choice:
        openai_api_key = st.text_input("Enter OpenAI API Key", type="password")

    elif "Groq" in model_choice:
        groq_api_key = st.text_input("Enter Groq API Key", type="password")

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

    # ✅ Show chat history using st.chat_message
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    # ✅ Input bar stays at the bottom
    user_question = st.chat_input("Ask a question from the PDFs...")
    if user_question:
        if not st.session_state.get("uploaded_filenames"):  # no files uploaded
            with st.chat_message("assistant"):
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
                else:
                    llm = get_bedrock_deepseek_llm()

                st.session_state.qa_chain = get_conversational_chain(llm, faiss_index)

                result = st.session_state.qa_chain({
                    "question": user_question,
                    "chat_history": st.session_state.chat_history
                })

                answer = result["answer"]

            with st.chat_message("assistant"):
                st.markdown(answer)

            # Save into chat history
            st.session_state.chat_history.append((user_question, answer)) 

if __name__ == "__main__":
    main()