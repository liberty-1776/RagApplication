import os
import boto3
import streamlit as st
from langchain_ollama import ChatOllama

## We will be suing Titan Embeddings Model To generate Embeddings
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock

## Data Ingestion and Transformatioin
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader


# Vector Embedding And Vector Store
from langchain_community.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA,ConversationalRetrievalChain

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)


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

def get_claude_llm():
    ##create the Anthropic Model
    llm=llm=ChatOllama(model="gemma2:2b")
    
    return llm

def get_llama2_llm():
    ##create the Anthropic Model
    llm=ChatBedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock,
                model_kwargs={'max_gen_len':512})
    
    return llm

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but atleast summarize with 
250 words with detailed explaantions.
If the provided context is not enough, use your own knowledge to answer the question to 
the best of you ability.

<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


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

    model_choice = st.selectbox(
        "Model",
        ["Claude 3 Sonnet", "Llama3-70B Instruct"],
        key="model_choice"
    )

    st.header("Chat with PDF using AWS Bedrock")

    with st.sidebar:
        st.title("Upload your PDF files here")

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

    # ✅ Show chat history using st.chat_message
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    # ✅ Input bar stays at the bottom
    user_question = st.chat_input("Ask a question from the PDFs...")
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner(f"Processing your question with {model_choice}..."):
            faiss_index = FAISS.load_local(
                "faiss_index",
                bedrock_embeddings,
                allow_dangerous_deserialization=True
            )
            if model_choice == "Claude 3 Sonnet":
                llm = get_claude_llm()
            else:
                llm = get_llama2_llm()

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