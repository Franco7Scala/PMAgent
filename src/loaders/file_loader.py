import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from src.utils import cprint, Color, get_device


def build_vector_store(file_paths, model_name, chunk_size=700, chunk_overlap=50):
    # reading documents from the file paths provided
    docs_to_be_splitted = []
    for path in file_paths:
        if not os.path.exists(path):
            cprint(f"The file '{path}' not exists!", color=Color.WARNING)
            continue

        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)

        elif ext == ".txt":
            loader = TextLoader(path)

        elif ext == ".csv":
            loader = CSVLoader(path)

        else:
            cprint(f"Unsupported file extension for file '{path}'!", color=Color.WARNING)
            continue

        docs_to_be_splitted.extend(loader.load())

    # splitting documents into chunks of text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs_splitted = text_splitter.split_documents(docs_to_be_splitted)
    cprint(f"Loaded {len(docs_to_be_splitted)} files, splitted in {len(docs_splitted)} chunks!", color=Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    huggingface_embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": get_device()}, encode_kwargs={"normalize_embeddings": True})
    return FAISS.from_documents(docs_splitted, huggingface_embeddings)
