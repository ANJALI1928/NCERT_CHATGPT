from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

#loading the environment variable
load_dotenv()

# defining a function to automate and add documents
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
def load_embeddings(model_name = "text-embedding-3-large"):
    embeddings = OpenAIEmbeddings(model = model_name, api_key = OPENAI_API_KEY)
    return embeddings

def load_vectorstore(name_of_vectorstore):
    embeddings = load_embeddings()
    chroma_db = Chroma(embedding_function=embeddings, persist_directory = name_of_vectorstore)
    return chroma_db

def run_training():
    folder_path = "./100_pdfs/"           # Give your folder path where all all your pdfs are saved
    name_of_vectorstore = "./vectorstore"
    vectorstore = load_vectorstore(name_of_vectorstore)
    for file_name in tqdm(os.listdir(folder_path)):
        file_name = folder_path + file_name
        loader = PDFMinerLoader(file_name, concatenate_pages = False)
        pages = loader.load()
        vectorstore.add_documents(pages)
        print(f"{file_name} updated successfully")
    vectorstore.persist()


def main():
    run_training()


            
if __name__ == "__main__":
    main()