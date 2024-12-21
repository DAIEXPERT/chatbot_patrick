import os
import shelve
from PyPDF2 import PdfReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# App Title
st.title("Mounaim's Resume Explorer")
st.subheader("Ask anything you want from your policies")

# Constants
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"
client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

# Ensure the OpenAI model is initialized in session state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Function to load and process multiple PDF files
def load_multiple_pdfs(file_paths):
    combined_text = ""
    for file_path in file_paths:
        try:
            pdf_reader = PdfReader(file_path)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    combined_text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading {file_path}: {str(e)}")
    return combined_text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to create a FAISS vectorstore from text chunks
def create_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(text_chunks, embeddings)

# Function to load chat history from shelve
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

# Function to save chat history to shelve
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# List of predefined PDF files
pdf_files = ["docs/file1.pdf", "docs/file2.pdf", "docs/file3.pdf"]

# Load and process PDFs into a combined vectorstore on app startup
if "vectorstore" not in st.session_state:
    combined_text = load_multiple_pdfs(pdf_files)
    if combined_text.strip():  # Ensure text is not empty
        text_chunks = get_text_chunks(combined_text)
        st.session_state.vectorstore = create_vectorstore(text_chunks)
    else:
        st.error("No valid text found in the provided PDF files.")

# Function to retrieve context from the vectorstore
def retrieve_context(query):
    retriever = st.session_state.vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    return context

# Sidebar for informational purposes
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("Chat with your policies !"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # Retrieve context from vectorstore
    context = retrieve_context(prompt)

    # Combine user query with retrieved context
    full_prompt = f"Context: {context}\n\nQuestion: {prompt}"

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[{"role": "user", "content": full_prompt}],
            stream=True,
        ):
            full_response += response.choices[0].delta.content or ""
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)
