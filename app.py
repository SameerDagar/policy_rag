"""Simple UI:  streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

import rag

st.set_page_config(page_title="Policy Q&A", page_icon="📄")
st.title("📄 Company Policy Assistant")
st.caption("Answers come only from your documents. Anything else gets \"I don't know\".")

if "store" not in st.session_state:
    st.session_state.store = rag.load_index()

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Build index from uploads", disabled=not uploads):
        tmp = Path(tempfile.mkdtemp())
        paths = []
        for u in uploads:
            p = tmp / u.name
            p.write_bytes(u.getvalue())
            paths.append(p)
        with st.spinner("Chunking and embedding..."):
            st.session_state.store = rag.build_index(paths)[0], rag.load_index()[1]
        st.success(f"Indexed {len(rag.load_index()[1])} chunks.")
    if st.button("Index bundled sample docs (./docs)"):
        with st.spinner("Indexing..."):
            rag.build_index(sorted(Path("docs").glob("*")))
            st.session_state.store = rag.load_index()
        st.success("Sample docs indexed.")
    k = st.slider("Chunks to retrieve (k)", 1, 8, 4)

index, chunks = st.session_state.store
if index is None:
    st.info("Build an index from the sidebar to get started.")
    st.stop()

q = st.chat_input("Ask a question about the documents...")
if q:
    st.chat_message("user").write(q)
    with st.spinner("Thinking..."):
        ans, hits = rag.answer(q, index, chunks, k)
    with st.chat_message("assistant"):
        st.write(ans)
        with st.expander("Sources"):
            for i, h in enumerate(hits, 1):
                st.markdown(f"**[{i}] {h['source']} {h['page']}** · similarity {h['score']:.2f}")
                st.caption(h["text"])
