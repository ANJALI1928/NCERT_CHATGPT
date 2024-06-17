import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage
import random
import json
#import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
#nltk.download('punkt')
#nltk.download('stopwords')

#loading the environment variable
load_dotenv()

# defining a function to automate and add documents
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
def load_embeddings(model_name = "text-embedding-3-large"):
    embeddings = OpenAIEmbeddings(model=model_name, api_key=OPENAI_API_KEY)
    return embeddings

def load_vectorstore(name_of_vectorstore):
    embeddings = load_embeddings()
    chroma_db = Chroma(embedding_function=embeddings, persist_directory= name_of_vectorstore)
    return chroma_db

def load_llm(model = "gpt-3.5-turbo"):
    llm = OpenAI()
    return llm

def response_generator(question, docs, llm):
    messages = [HumanMessage(content = f"Context:\n{docs}\n\n Question:\n{question}")]
    s = {"role":"assistant","content":messages[0].content}
    answer = llm.chat.completions.create(model = "gpt-3.5-turbo", messages = [s], stream = True)
    answer = st.write_stream(answer)
    return answer

# Suggested questions
suggested_questions = [
    "What is Indian National Movement?",
    "What is civil disobedience movement?",
    "What is the constitution of India?","What is quit India movement?","When did India got its Independence?"
]

def fetch_response(question, vectorstore, llm):
    with st.chat_message("assistant"):
        docs = vectorstore.similarity_search(question)
        if docs:
            source_info = []
            for doc in docs:
                metadata = doc.metadata
                st.write(metadata)  # Debug print statement to check the metadata structure
                pdf_name = metadata.get('source', 'Unknown PDF')
                page_number = metadata.get('page', 'Unknown Page')
                source_info.append(f"PDF: {pdf_name}, Page: {page_number}")
            
            answer = response_generator(question, docs, llm)
            source_text = "\n\nSources:\n" + "\n".join(source_info)
            answer_with_sources = answer + source_text

            st.session_state.messages.append({"role": "assistant", "content": answer_with_sources})
        else:
            answer = "I don't know"
            st.session_state.messages.append({"role": "assistant", "content": answer})

def save_chat_history():
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.chat_history, f)

def load_chat_history():
    try:
        if os.path.exists("chat_history.json"):
            with open("chat_history.json", "r") as f:
                st.session_state.chat_history = json.load(f)
        else:
            st.session_state.chat_history = []
    except json.JSONDecodeError:
        st.session_state.chat_history = []

def delete_chat_history(index):
    del st.session_state.chat_history[index]
    save_chat_history()
    st.rerun()

def move_chat_to_top(index):
    chat = st.session_state.chat_history.pop(index)
    st.session_state.chat_history.insert(0, chat)
    save_chat_history()

def edit_chat_title(index, new_title):
    st.session_state.chat_history[index]["title"] = new_title
    save_chat_history()
    st.session_state.edit_index = None
    st.rerun()

def generate_chat_title(question):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(question)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    title = " ".join(filtered_words[:2])
    return title

def main():
    st.set_page_config(page_title="NCERT_GPT")
    #st.title("NCERT_GPT")

    # Load vectorstore and LLM once
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_vectorstore("./vectorstore")
    if "llm" not in st.session_state:
        st.session_state.llm = load_llm()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "question_asked" not in st.session_state:
        st.session_state.question_asked = False
    if "chat_history" not in st.session_state:
        load_chat_history()
    if "current_chat_index" not in st.session_state:
        st.session_state.current_chat_index = None
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = None

    # Save chat history if a new chat has been initiated
    if st.session_state.current_chat_index is None and st.session_state.messages:
        new_chat = {
            "title": generate_chat_title(st.session_state.messages[0]["content"]),
            "messages": st.session_state.messages
        }
        st.session_state.chat_history.insert(0, new_chat)  # Add new chat to the top
        st.session_state.current_chat_index = 0
        save_chat_history()


    # Display chat history in the sidebar
    with st.sidebar:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown('<p style="font-size:24px;">Chat History</p>', unsafe_allow_html=True)
        with col2:
            st.button("🆕", key="new_chat", help="New Chat", on_click=lambda: st.session_state.update({"messages": [], "question_asked": False, "current_chat_index": None}) and st.rerun())

        for i, chat in enumerate(st.session_state.chat_history):
            col1, col2, col3 = st.columns([3, 1, 1])
            if st.session_state.edit_index == i:
                new_title = col1.text_input("Edit Title", value=chat["title"], key=f"edit_{i}")
                if col2.button("Save", key=f"save_{i}"):
                    edit_chat_title(i, new_title)
            else:
                if col1.button(chat["title"], key=f"chat_{i}"):
                    st.session_state.messages = chat["messages"]
                    st.session_state.question_asked = True
                    st.session_state.current_chat_index = i
                    st.rerun()
                if col2.button("🗑️", key=f"delete_{i}"):
                    delete_chat_history(i)
                if col3.button("✏️", key=f"edit_{i}"):
                    st.session_state.edit_index = i
                    st.rerun()


    # Display suggested questions after clicking the start button
    if not st.session_state.question_asked:
        random_questions = random.sample(suggested_questions, 4)
        cols = st.columns(2)
        st.image("static/ncert.png", width=50)
        st.markdown('<p style="text-align:center; font-weight:bold; font-style:italic; margin-top:15px;">Welcome to NCERT_GPT!:</p>', unsafe_allow_html=True)
        st.markdown('<p style="text-align:center; font-weight:bold; font-style:italic; margin-top:1px;color: #888888;">Generates answer based on the questions that you have asked. NCERT_GPT is your interactive tool for exploring and understanding the NCERT textbooks. Whether you have specific questions or need summaries of the content, this app is here to help.</p>', unsafe_allow_html=True)
        for i, q in enumerate(random_questions):
            col = cols[i % 2]  # Alternate between columns
            if col.button(q):
                st.session_state.messages.append({"role": "user", "content": q})
                fetch_response(question=q, vectorstore=st.session_state.vectorstore, llm=st.session_state.llm)
                st.session_state.question_asked = True
                st.rerun()

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the PDFs")
    if question:  # Mark as started if a question is typed
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        fetch_response(question=question, vectorstore=st.session_state.vectorstore, llm=st.session_state.llm)
        st.session_state.question_asked = True
        if st.session_state.current_chat_index is not None:
            st.session_state.chat_history[st.session_state.current_chat_index]["messages"] = st.session_state.messages
            move_chat_to_top(st.session_state.current_chat_index)
        st.rerun()
if __name__ == "__main__":
    main()