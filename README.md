# 📄 RAG Document Q&A

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=flat&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI_API-GPT--3.5%2F4o-412991?style=flat&logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-00599C?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

An AI-powered document question-answering system built with **Retrieval-Augmented Generation (RAG)**. Upload any PDF, ask questions in plain English, and get grounded answers with cited source passages — all in a clean Streamlit interface.

---

## ✨ Features

- 📤 **PDF upload** — drag-and-drop any PDF document
- 🔍 **Semantic search** — FAISS vector index for fast similarity retrieval
- 🤖 **GPT-powered answers** — responses grounded strictly in your document
- 📎 **Source citations** — every answer links back to the exact page and passage
- 💬 **Multi-turn chat** — conversation history preserved within the session
- ⚡ **Smart caching** — document is only re-embedded when a new file is uploaded
- 🔒 **Key stays local** — your OpenAI key never leaves the browser session

---

## 🏗️ Architecture

```
┌──────────────┐    ┌───────────────────────────────────┐
│  PDF Upload  │───▶│  PyPDFLoader → RecursiveTextSplitter│
└──────────────┘    └──────────────┬────────────────────┘
                                   │ chunks
                    ┌──────────────▼────────────────────┐
                    │  OpenAI text-embedding-3-small     │
                    │  → FAISS in-memory vector store    │
                    └──────────────┬────────────────────┘
                                   │ retriever (top-4)
┌──────────────┐    ┌──────────────▼────────────────────┐
│  User query  │───▶│  RetrievalQA chain (LangChain)    │
└──────────────┘    │  + custom RAG prompt               │
                    │  → ChatOpenAI (GPT-3.5 / 4o)      │
                    └──────────────┬────────────────────┘
                                   │
                    ┌──────────────▼────────────────────┐
                    │  Answer + source passages → UI    │
                    └───────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Haochen0416/rag-document-qa.git
cd rag-document-qa
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`, paste your OpenAI API key in the sidebar, upload a PDF, and start asking questions.

> **Note:** You need a valid [OpenAI API key](https://platform.openai.com/api-keys). The key is only used within your local session and is never stored.

---

## 📂 Project Structure

```
rag-document-qa/
├── app.py              # Streamlit UI (upload, chat, source display)
├── rag_pipeline.py     # Core RAG logic (load → split → embed → retrieve → answer)
├── requirements.txt    # Python dependencies
└── README.md
```

### `rag_pipeline.py` — core logic

| Function | Description |
|----------|-------------|
| `load_and_split(pdf_path)` | Load PDF with PyPDFLoader, split with RecursiveCharacterTextSplitter (800 chars, 150 overlap) |
| `build_vectorstore(chunks, api_key)` | Embed chunks with `text-embedding-3-small`, build FAISS index |
| `build_qa_chain(vectorstore, api_key, model)` | Wire retriever + ChatOpenAI into a RetrievalQA chain |
| `query(chain, question)` | Run a question, return `(answer, source_documents)` |

---

## 🔧 Configuration

All settings are controlled from the **sidebar** in the UI:

| Setting | Options | Default |
|---------|---------|---------|
| OpenAI API Key | Any valid key | — |
| GPT Model | `gpt-3.5-turbo`, `gpt-4o-mini`, `gpt-4o` | `gpt-3.5-turbo` |

To change chunking parameters, edit constants at the top of `rag_pipeline.py`:

```python
chunk_size    = 800   # characters per chunk
chunk_overlap = 150   # overlap between chunks
k             = 4     # number of chunks retrieved per query
```

---

## 💡 Example Usage

```
Upload: company_annual_report_2024.pdf

Q: What was the total revenue in Q4?
A: According to the document, Q4 revenue was $2.3 billion, representing
   a 12% year-over-year increase driven by cloud services growth.
   [Source: page 14, Financial Highlights section]

Q: Who is the current CEO?
A: The document identifies Jane Smith as the Chief Executive Officer,
   appointed in March 2022.
   [Source: page 3, Executive Leadership]
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-3.5-turbo / GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | FAISS (in-memory) |
| RAG Framework | LangChain 0.2 |
| PDF Parsing | PyPDF |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## 📈 Potential Improvements

- [ ] Support for `.txt`, `.docx`, and web URL inputs
- [ ] Persistent vector store with ChromaDB
- [ ] Multi-document Q&A across a collection
- [ ] Streaming responses for lower perceived latency
- [ ] Deployment to Streamlit Cloud

---

## 👤 Author

**Haochen Li** — M.S. Computer Engineering, SMU Dallas (2026)  
[GitHub](https://github.com/Haochen0416) · [LinkedIn](https://linkedin.com/in/haochenli)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
