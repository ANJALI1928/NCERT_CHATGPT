import os
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import HumanMessage

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

def main():
    st.set_page_config(page_title="NCERT_GPT")
    st.title("NCERT_GPT")

    name_of_vectorstore = "./vectorstore"
    vectorstore = load_vectorstore(name_of_vectorstore)
    llm = load_llm()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the PDFs")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Load the vector store
        with st.chat_message("assistant"):
            docs = vectorstore.similarity_search(question)
            if docs:        
                answer = response_generator(question, docs, llm)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                answer = "I don't know"
                st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
