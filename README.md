# Policy RAG – document-based Q&A

Ask questions about company policy documents; answers come strictly from those documents.

| Requirement | Where |
|---|---|
| Document input | `rag.load_document` (PDF/TXT/MD) + Streamlit uploader |
| Chunking | `rag.chunk_text` (150 words, 30 overlap) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, local) |
| Local vector store | FAISS `IndexFlatIP`, saved to `./index` |
| Semantic retrieval | `rag.retrieve` (top-k cosine) |
| LLM generation | Claude via Anthropic API (`LLM_MODEL` env var to change) |
| Strict RAG prompt | `rag.SYSTEM_PROMPT` + similarity threshold refusal |
| Simple UI | `app.py` (Streamlit) |
| 5+ test questions | `run_tests.py` (6 questions incl. 1 out-of-scope) |

## Run
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # Windows: set ANTHROPIC_API_KEY=...
python run_tests.py                      # builds index from ./docs and runs tests
streamlit run app.py                     # UI; click "Index bundled sample docs"
```
To use your own domain, drop files in `docs/` or upload them in the UI and edit the questions in `run_tests.py`.

## How hallucination is limited
1. If the best chunk's similarity is below `MIN_SCORE`, the LLM is never called.
2. The system prompt forbids outside knowledge and requires a fixed refusal sentence and citations.
3. Temperature is 0.
