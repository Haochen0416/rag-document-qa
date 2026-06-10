"""
rag_pipeline.py
Core RAG logic using LangChain 0.2+ LCEL.
"""

from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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


def load_and_split(pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)


def build_vectorstore(chunks: List, api_key: str) -> FAISS:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )
    return FAISS.from_documents(chunks, embeddings)


def build_qa_chain(vectorstore: FAISS, api_key: str, model: str = "gpt-3.5-turbo"):
    llm = ChatOpenAI(model=model, temperature=0, openai_api_key=api_key)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def query(chain_and_retriever, question: str) -> Tuple[str, List]:
    chain, retriever = chain_and_retriever
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return answer, sources
