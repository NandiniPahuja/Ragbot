# RAG Law Student Helper Chatbot

This is a simple Retrieval-Augmented Generation (RAG) chatbot tailored for law students, built with LangChain, OpenAI GPT, and FAISS or ChromaDB for vector storage.

## Features
- Load local documents (plain text and PDF) and index them with embeddings (PDF support via PyPDFLoader).
- Perform semantic search over documents to provide context for chat queries.
- Maintain conversational history for follow-up questions.

## Requirements
See [requirements.txt](requirements.txt) (includes pypdf for PDF support).

## Installation
```bash
# (Optional) If you want to pin your Python version via pyenv:
# curl https://pyenv.run | bash
# pyenv install 3.11.9
# pyenv local 3.11.9

# Create & activate a virtual environment:
python3 -m venv venv
source venv/bin/activate

# Install dependencies:
pip install -r requirements.txt
```

## Setup
1. Create a `.env` file in this directory or set the `OPENAI_API_KEY` environment variable:
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```
2. Prepare a `knowledge_base/` directory with your `.txt` and/or `.pdf` documents to use as the knowledge base.

## Usage
```bash
# First time (or rebuild index):
python app.py \
  --docs knowledge_base \
  --persist_dir vector_index \
  --vectorstore chroma \
  --rebuild

# Load existing index and chat:
python app.py \
  --docs knowledge_base \
  --persist_dir vector_index \
  --vectorstore chroma

## Utilities

### PDF upload script: upload_pdf.py

A helper to extract text from a single PDF and upload it to a ChromaDB collection.

```bash
python upload_pdf.py path/to/file.pdf
```

## Document Loading & Preprocessing

- **Diverse Loaders**: To ingest formats beyond plain text and PDFs (e.g., DOCX, HTML, CSV, Markdown), leverage the corresponding loaders in `langchain_community.document_loaders` such as `Docx2txtLoader`, `UnstructuredHTMLLoader`, `CSVLoader`, or `MarkdownLoader`.
- **OCR for scanned PDFs**: If your PDFs are scanned images, integrate an OCR step (e.g., Tesseract or Google Cloud Vision API) before or during loading. PyMuPDF works well on extractable text but may miss text in poor-quality scans.
- **Layout-aware parsing**: For documents with complex layouts (tables, multi-column text), consider using layout-capable parsers like the `unstructured` or `nougat` libraries for more accurate extraction.
- **Metadata extraction**: After loading, each `Document` may carry metadata (title, author, sections). You can customize your loader logic (in `app.py` or a preprocessing script) to populate or filter on `doc.metadata` for richer retrieval and context filtering.

## Chunk Splitting & Embedding Considerations

### Adaptive Chunking Strategies
- **Semantic/Contextual Chunking**: Instead of fixed character counts, use header-aware splitters (e.g., `MarkdownHeaderTextSplitter`, `HTMLHeaderTextSplitter`) or token-based splitters (`TokenTextSplitter`) to respect semantic boundaries. For documents with clear sections—like Constitution.pdf—you may find `MarkdownHeaderTextSplitter` yields more coherent chunks aligned with headings.
- **Sentence Boundary Awareness**: Ensure chunks do not cut off mid-sentence; many splitters handle this (e.g., `RecursiveCharacterTextSplitter(respect_sentence_boundaries=True)`).
- **Chunk Size & Overlap Tuning**: Experiment with `chunk_size` and `chunk_overlap` in `RecursiveCharacterTextSplitter` to balance retrieval granularity versus context completeness. Lower sizes and higher overlap improve recall but cost more tokens.
- **Small-to-Big Retrieval**: Retrieve many small, focused chunks (e.g., `k=5` with smaller size) and then combine the top results into a larger context chunk for the LLM (often called "Small-to-Big Retrieval").

### Embedding Model Choices
- **OpenAI Embeddings**: Default (`text-embedding-ada-002`) via `OpenAIEmbeddings`.
- **Alternative/Open-Source Models**: For privacy, cost, or domain-specific needs, consider Hugging Face models with `HuggingFaceEmbeddings`:
  ```python
  # from langchain_openai import OpenAIEmbeddings
  from langchain_community.embeddings import HuggingFaceEmbeddings

  # embeddings = OpenAIEmbeddings(openai_api_key=...)
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  ```

## Vector Store & Retrieval Enhancements

- **Hybrid Search (Sparse + Dense)**: Combine classic keyword-based sparse retrieval (e.g., BM25, TF-IDF) with Chroma’s vector similarity search to catch both lexical matches and semantic relevance. You can merge a `KeywordRetriever` with Chroma’s retriever for hybrid results.
- **Advanced Dense Retrieval (ANCE/ColBERT)**: For state-of-the-art dense search, consider ANCE or ColBERT (e.g., via RAGatouille), which may require a separate service but can boost retrieval accuracy.
- **Re-Ranking**: After initial retrieval (e.g., top 20 chunks), re-rank them using a Cross-Encoder (from `sentence-transformers`) or the LLM itself to identify the most relevant snippets before generation.
- **Multi-Query Retrieval**: Use LangChain’s `MultiQueryRetriever` to generate and aggregate multiple query reformulations, improving recall across variations of the user’s question.
- **Parent Document Retrieval**: With `ParentDocumentRetriever`, automatically fetch larger parent sections or documents for any highly relevant chunk, providing richer context to the LLM.
- **Metadata Filtering**: Leverage document metadata (e.g., date, author, section) to pre-filter the search space using Chroma’s `where` clauses or `collection_metadata` parameters for targeted retrieval.

**Example (Hybrid + Re-Rank):**
```python
from langchain.retrievers import KeywordRetriever, MultiRetriever
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

# sparse keyword lookup
sparse = KeywordRetriever(directory="knowledge_base")
# dense semantic lookup (k=5)
dense = Chroma(persist_directory="vector_index",
                embedding_function=embeddings).as_retriever(search_kwargs={"k": 5})
# combine
retriever = MultiRetriever([sparse, dense])

# re-rank top-20 via Cross-Encoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# wrap MultiRetriever to re-rank
retriever = CrossRanker(retriever=retriever, cross_encoder=cross_encoder, top_k=20)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever, return_source_documents=True
)
```

## Generation (LLM Integration)

- **Prompt Engineering**: Provide clear, specific instructions so the LLM leverages the retrieved context, avoids hallucination, and responds in the desired format (e.g., bullet points, Q&A style, summaries).
- **LLM Choice**: Select the model that fits your needs:
  - **GPT-4 / Claude Opus**: Highest quality and reasoning capability.
  - **Fine-tuned Models**: Customize to domain‑specific data for better accuracy.
  - **Local LLMs** (Llama.cpp, Ollama, CTransformers): Cost‑effective and privacy‑preserving; integrate via `langchain_community.llms.Ollama` or `CTransformers`.
- **Guardrails & Fact‑Checking**: Implement post‑generation verification—e.g., use a smaller LLM or rule‑based checks to ensure answers are supported by source chunks.
- **Chat History Management**: For coherent multi‑turn dialogue, pass truncated `chat_history` to the LLM or use `ConversationalRetrievalChain` to maintain context across turns.

## Example
```text
$ python app.py --docs knowledge_base --persist_dir vector_index --vectorstore chroma --rebuild
Building 'chroma' vectorstore from knowledge base in 'knowledge_base'...

RAG Law Student Helper is ready! Type 'exit' or 'quit' to stop.
You: What is habeas corpus?
Bot: Habeas corpus is a legal action or writ by means of which detainees can seek relief...
``` 
