import os
import streamlit as st
import pandas as pd
from rag import query_rag, reset_chromadb, get_embedding
import chromadb
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHROMA_PATH = config["chroma_path"]

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("rag_collection")

st.title("Chalapathi Chatbot")

# File Upload
st.sidebar.header("Upload Excel Files")
uploaded_files = st.sidebar.file_uploader("Upload Excel file(s)", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    st.sidebar.success("Processing knowledge...")

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        df_dict = pd.read_excel(uploaded_file, sheet_name=None)  # Read all sheets

        for sheet_name, df in df_dict.items():
            df.fillna("", inplace=True)

            for idx, row in df.iterrows():
                text = ", ".join([f"{col}:{row[col]}" for col in df.columns])
                embedding = get_embedding(text)

                collection.add(
                    ids=[f"{file_name}_{sheet_name}_{idx}"],
                    embeddings=[embedding],
                    metadatas=[{"text": text, "file": file_name}]
                )

    st.sidebar.success("Knowledgebase created/updated!")

# Chat Interface
st.header("Ask your query related to Chalapathi College!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something about your document...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    response = query_rag(query)
    
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": response})

# Reset Button
if st.sidebar.button("Reset Knowledge Base"):
    reset_chromadb()
    st.sidebar.warning("Knowledge Base Reset!")
