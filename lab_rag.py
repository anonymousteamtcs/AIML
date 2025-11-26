import streamlit as st
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tempfile
import os
import httpx
import tiktoken
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Please modify the "tiktoken_cache_dir" to the directory wherever you are placing your "tiktoken_cache" folder
tiktoken_cache_dir = "tiktoken_cache"

os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

client = httpx.Client(verify=False)


# LLM setup (IMPORTANT)
# Please update the base_url, model, api_key as specified below.
llm = ChatOpenAI(
    base_url = os.getenv("api_endpoint"),
    api_key = os.getenv("api_key"),
    model="<get your model name from .env file from your respective team's .env file>",
    http_client=client
)


# EMBEDDING setup (IMPORTANT)
# Please update the base_url, model, api_key as specified below.
embedding_model = OpenAIEmbeddings(
    base_url = os.getenv("api_endpoint"),
    api_key = os.getenv("api_key"),
    model="<get your model name from .env file from your respective team's .env file>",
    http_client=client
)

st.set_page_config(page_title="RAG PDF Summarizer")
st.title(" RAG-powered PDF Summarizer")
upload_file = st.file_uploader("Upload a PDF", type="pdf")

if upload_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(upload_file.read())
        temp_file_path = temp_file.name
        
    # Step 1: Extract text
    raw_text = extract_text(temp_file_path)

    # Step 2: Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

    # Step 3: Embed and store in Chroma
    with st.spinner("Indexing document..."):
        vectordb = Chroma.from_texts(chunks, embedding_model, persist_directory="./chroma_index")
        vectordb.persist()

    # Step 4: RAG QA Chain
    #retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    retriever = vectordb.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    # Step 5: Ask summarization prompt
    summary_prompt = "Please summarize this document based on the key topics:"
    with st.spinner("Running RAG summarization..."):
        # result = rag_chain.run(summary_prompt)
        result = rag_chain.invoke(summary_prompt)

    st.subheader(" Summary")
    st.write(result)
