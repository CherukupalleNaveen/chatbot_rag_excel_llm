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
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("rag_collection")

st.set_page_config(page_title="Chalapathi Admissions Bot", page_icon="🎓")
st.title("🎓 Chalapathi Admissions Chatbot")

# File upload
st.sidebar.header("📂 Upload Excel Files")
uploaded_files = st.sidebar.file_uploader("Upload Excel file(s)", type=["xlsx"], accept_multiple_files=True)

for uploaded_file in uploaded_files:
    file_name = uploaded_file.name
    df_dict = pd.read_excel(uploaded_file, sheet_name=None, header=0)

    for sheet_name, df in df_dict.items():
        df.fillna("", inplace=True)
        df.columns = [str(col).strip() for col in df.columns]

        for idx, row in df.iterrows():
            text = ", ".join([f"{col}:{row[col]}" for col in df.columns])
            embedding = get_embedding(text)
            collection.add(
                ids=[f"{file_name}_{sheet_name}_{idx}"],
                embeddings=[embedding],
                metadatas=[{"text": text, "file": file_name}]
            )

    st.sidebar.success("✅ Knowledgebase created/updated!")

# Reset DB
if st.sidebar.button("🗑️ Reset Knowledge Base"):
    reset_chromadb()
    st.sidebar.warning("⚠️ Knowledge Base Reset!")

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.name = ""
    st.session_state.mobile = ""
    st.session_state.email = ""
    st.session_state.messages = []

# Render previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Conversation logic
if st.session_state.step == 0:
    greeting = "👋 Hi! Welcome to **Chalapathi College Admissions** chatbot. I'm here to help you with any information you need. Let's get started!"
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    st.session_state.step += 1
    st.rerun()

elif st.session_state.step == 1:
    with st.chat_message("assistant"):
        st.markdown("Can I know your **name**?")
    name = st.chat_input("Enter your name...")
    if name:
        st.session_state.name = name
        st.session_state.messages.append({"role": "user", "content": name})
        st.session_state.messages.append({"role": "assistant", "content": f"Nice to meet you, {name}!"})
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == 2:
    with st.chat_message("assistant"):
        st.markdown("What's your **mobile number**?")
    mobile = st.chat_input("Enter your mobile number...")
    if mobile:
        st.session_state.mobile = mobile
        st.session_state.messages.append({"role": "user", "content": mobile})
        st.session_state.messages.append({"role": "assistant", "content": "Thanks! Got your number."})
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == 3:
    with st.chat_message("assistant"):
        st.markdown("Great! And your **email address**?")
    email = st.chat_input("Enter your email...")
    if email:
        st.session_state.email = email
        st.session_state.messages.append({"role": "user", "content": email})
        st.session_state.messages.append({"role": "assistant", "content": "Perfect! You're all set."})
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == 4:
    with st.chat_message("assistant"):
        st.markdown(f"Thanks **{st.session_state.name}**! What would you like to know about Chalapathi College?")
    st.session_state.step += 1

# Main Chat Interface
faq_options = ["Courses Offered", "Fee Structure", "Hostel Facilities", "Placements", "Faculty", "Contact Info"]

if st.session_state.step >= 5:
    st.write("### 🤖 Choose a topic or ask your own question:")
    cols = st.columns(3)
    selected = None
    for i, option in enumerate(faq_options):
        if cols[i % 3].button(option):
            selected = option

    user_query = st.chat_input("Type your question here...")

    if selected or user_query:
        final_query = selected or user_query

        # Show user message
        with st.chat_message("user"):
            st.markdown(final_query)
        st.session_state.messages.append({"role": "user", "content": final_query})

        # Generate response
        response = query_rag(final_query)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
