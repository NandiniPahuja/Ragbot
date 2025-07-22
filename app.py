#!/usr/bin/env python3
"""
RAG Law Student Helper: a retrieval-augmented generation chatbot using LangChain, OpenAI,
and FAISS or Chroma (ChromaDB) for vector storage.
"""
import os
import argparse

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI  # community chat model (if desired)
from langchain.chains import ConversationalRetrievalChain


def build_vectorstore(docs_path: str, persist_dir: str, embeddings, store_type: str):
    """
    Load documents, split into chunks, create a vectorstore (FAISS or Chroma), and persist it.
    """
    # Load text files and PDF documents
    text_loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = text_loader.load() + pdf_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)
    if store_type == "faiss":
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(persist_dir)
    else:
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="RAG Law Student Helper Chatbot")
    parser.add_argument(
        "--docs", type=str, default="knowledge_base", help="Path to the knowledge base directory"
    )
    parser.add_argument(
        "--persist_dir", type=str, default="vector_index",
        help="Directory to store the vectorstore index"
    )
    parser.add_argument(
        "--vectorstore", choices=["faiss", "chroma"], default="chroma",
        help="Which vector store backend to use (default: chroma)"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild the vectorstore even if an index exists"
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Please set it in environment or .env file.")
        return

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    if args.rebuild or not os.path.isdir(args.persist_dir):
        print(f"Building '{args.vectorstore}' vectorstore from knowledge base in '{args.docs}'...")
        vectorstore = build_vectorstore(
            args.docs, args.persist_dir, embeddings, args.vectorstore
        )
    else:
        print(f"Loading existing '{args.vectorstore}' vectorstore from '{args.persist_dir}'...")
        if args.vectorstore == "faiss":
            vectorstore = FAISS.load_local(args.persist_dir, embeddings)
        else:
            vectorstore = Chroma(
                persist_directory=args.persist_dir,
                embedding_function=embeddings
            )

    llm = ChatOpenAI(openai_api_key=api_key, temperature=0)
    # use top-3 chunks for context and enable source tracking
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        return_source_documents=True,
    )

    chat_history = []
    print("\nRAG Law Student Helper is ready! Type 'exit' or 'quit' to stop.")
    while True:
        query = input("You: ")
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        result = qa_chain({"question": query, "chat_history": chat_history})
        # display retrieved context for transparency
        docs = result.get("source_documents", [])
        if docs:
            print("\n--- Relevant Context (top 3) ---")
            for i, doc in enumerate(docs, 1):
                print(f"[{i}] {doc.page_content}" )
            print("--- End Context ---\n")
        answer = result.get("answer", "")
        chat_history.append((query, answer))
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
