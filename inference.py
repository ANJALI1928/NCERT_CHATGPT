import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain


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

def response_generator(response_text):
    for word in response_text.split():
        yield word + " "
        time.sleep(0.1)
def main():
    st.set_page_config(page_title="NCERT_GPT")
    st.title("NCERT_GPT")
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
        name_of_vectorstore = "./vectorstore"
        vectorstore = load_vectorstore(name_of_vectorstore)
        docs = vectorstore.similarity_search(question)
        if docs:        
            qa_chain = load_qa_chain(llm = OpenAI())
            answer = qa_chain.run(input_documents=docs, question=question)
            response_text = ""
            response_placeholder = st.empty()
            if answer:
                response_stream = response_generator(answer)
                for chunk in response_stream:
                    response_text += chunk
                    response_placeholder.markdown(f"**assistant**: {response_text}")
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                answer = "I don't know"
                response_placeholder.markdown(f"**assistant**: {answer}")
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            # If no relevant documents are found
            answer = "None"
            st.markdown(f"**assistant**: {answer}")
            st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
