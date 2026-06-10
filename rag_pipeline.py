"""
rag_pipeline.py
Core RAG logic: document loading, chunking, embedding, and retrieval.
"""

import os
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ── Prompt template ──────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions based strictly
on the provided document context. If the answer cannot be found in the context,
say "I couldn't find that information in the uploaded document."

Context:
{context}

Question: {question}

Answer:""",
)


# ── Document processing ───────────────────────────────────────────────────────

def load_and_split(pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List:
    """Load a PDF and split it into overlapping text chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    return chunks


# ── Vector store ──────────────────────────────────────────────────────────────

def build_vectorstore(chunks: List, api_key: str) -> FAISS:
    """Embed chunks and build an in-memory FAISS index."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ── QA chain ──────────────────────────────────────────────────────────────────

def build_qa_chain(vectorstore: FAISS, api_key: str, model: str = "gpt-3.5-turbo") -> RetrievalQA:
    """Wire retriever + LLM into a RetrievalQA chain."""
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_key=api_key,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    return chain


# ── Query helper ──────────────────────────────────────────────────────────────

def query(chain: RetrievalQA, question: str) -> Tuple[str, List]:
    """
    Run a question through the chain.
    Returns (answer_text, source_documents).
    """
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = result.get("source_documents", [])
    return answer, sources
