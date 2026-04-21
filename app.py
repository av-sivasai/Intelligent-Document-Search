import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import datetime
from io import BytesIO

from utils.text_extractor import extract_text
from utils.preprocessor import clean_text, chunk_text, get_text_stats
from utils.embedder import embed_chunks, embed_query
from utils.vector_store import VectorStore

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Document Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom CSS for Premium UI
# -----------------------------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

    /* Global Typography & Spacing */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px;
    }
    
    /* Glassmorphism Cards */
    .result-card, .stat-box {
        background: rgba(150, 150, 150, 0.08);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(150, 150, 150, 0.2);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .result-card:hover, .stat-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(150, 150, 150, 0.3);
    }
    
    /* Typography inside Cards */
    .doc-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 12px;
        background: linear-gradient(90deg, #2196F3, #00BCD4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    
    .chunk-meta {
        font-size: 0.85rem;
        color: #78909C;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge {
        background: rgba(33, 150, 243, 0.15);
        color: #2196F3;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    .chunk-text {
        font-size: 1.05rem;
        line-height: 1.7;
        color: inherit;
        opacity: 0.9;
        font-weight: 400;
    }
    
    /* Highlight */
    .highlight {
        background: linear-gradient(120deg, rgba(255, 213, 79, 0.4) 0%, rgba(255, 193, 7, 0.4) 100%);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        color: inherit;
        box-shadow: 0 2px 10px rgba(255, 193, 7, 0.1);
    }
    
    /* Animated Progress Bar */
    .score-container {
        width: 100%;
        background-color: rgba(150, 150, 150, 0.15);
        border-radius: 8px;
        margin-top: 12px;
        margin-bottom: 20px;
        overflow: hidden;
        height: 10px;
    }
    
    .score-bar {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        position: relative;
        overflow: hidden;
    }
    
    /* Shimmer Effect for Loading / Score Bar */
    .score-bar::after {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Stat Box Polish */
    .stat-box {
        text-align: center;
        border-top: 4px solid #00BCD4;
        border-left: none;
    }
    .stat-box h3 {
        font-size: 2.5rem;
        margin: 0 0 8px 0;
        background: linear-gradient(90deg, #2196F3, #00BCD4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-box p {
        font-size: 1rem;
        margin: 0;
        opacity: 0.8;
        font-weight: 500;
    }

    /* Mobile Responsive */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 1.5rem !important;
        }
        .result-card, .stat-box {
            padding: 16px;
        }
        .doc-title {
            font-size: 1.2rem;
        }
        .chunk-text {
            font-size: 0.95rem;
        }
        .stat-box h3 {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session_state():
    if "store" not in st.session_state:
        store = VectorStore()
        store.load()
        st.session_state.store = store
        
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
        
    if "analytics" not in st.session_state:
        st.session_state.analytics = {
            "searches_performed": 0,
            "total_documents_processed": 0
        }
        
init_session_state()
store = st.session_state.store

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def highlight_text(text: str, query: str) -> str:
    """Highlights query terms in the text using basic regex."""
    if not query:
        return text
    terms = [re.escape(term) for term in query.split() if len(term) > 2]
    if not terms:
        return text
    pattern = re.compile(rf"({'|'.join(terms)})", re.IGNORECASE)
    return pattern.sub(r'<span class="highlight">\1</span>', text)

def export_results_csv(results: list) -> bytes:
    """Converts search results to a CSV byte object."""
    df = pd.DataFrame(results)
    return df.to_csv(index=False).encode('utf-8')

def process_uploaded_files(uploaded_files):
    """Processes newly uploaded files and adds them to the vector store."""
    if not uploaded_files:
        return
    
    with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
        progress_bar = st.progress(0)
        new_docs = 0
        
        for i, file in enumerate(uploaded_files):
            doc_name = file.name
            file_type = os.path.splitext(doc_name)[1].lower()
            
            # Duplicate check
            if doc_name in store.unique_documents:
                st.warning(f"⚠️ '{doc_name}' is already in the database. Skipped.")
                continue
                
            file_bytes = file.read()
            text = extract_text(file_bytes, doc_name)
            
            if not text:
                st.error(f"❌ Failed to extract text from {doc_name}.")
                continue
                
            cleaned_text = clean_text(text)
            chunks = chunk_text(cleaned_text)
            
            if not chunks:
                st.warning(f"⚠️ '{doc_name}' contained no usable text.")
                continue
                
            embeddings = embed_chunks(chunks)
            store.add_document(doc_name, file_type, chunks, embeddings)
            new_docs += 1
            st.session_state.analytics["total_documents_processed"] += 1
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        if new_docs > 0:
            store.save()
            st.success(f"✅ Successfully processed and indexed {new_docs} document(s)!")

# -----------------------------------------------------------------------------
# Sidebar: Navigation & Actions
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 AI Search Engine")
    st.markdown("Navigate through the app features.")
    
    app_mode = st.radio("Navigation", ["🔍 Semantic Search", "📊 Analytics Dashboard"])
    
    st.divider()
    
    st.subheader("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDFs, DOCXs, or TXTs",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    if st.button("Process Documents", use_container_width=True, type="primary"):
        process_uploaded_files(uploaded_files)
        
    st.divider()
    
    st.subheader("⚙️ Database Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Rebuild Index", help="Reload the FAISS index from disk", use_container_width=True):
            if store.load():
                st.success("Index reloaded!")
            else:
                st.error("No index found on disk.")
    with col2:
        if st.button("Clear DB", help="Wipe all data", use_container_width=True):
            store.clear()
            st.success("Database cleared.")
            st.rerun()

# -----------------------------------------------------------------------------
# Main Area: Semantic Search
# -----------------------------------------------------------------------------
if app_mode == "🔍 Semantic Search":
    st.title("🔍 Semantic Document Search")
    st.markdown("Find relevant information based on **meaning**, not just exact keyword matches.")
    
    # --- Search Bar & Suggestions ---
    col_search, col_sugg = st.columns([3, 1])
    with col_search:
        query = st.text_input("Enter your search query...", placeholder="e.g., 'machine learning algorithms in healthcare'")
    with col_sugg:
        suggestions = ["What is the methodology?", "Summary of results", "Financial projections", "Security protocols"]
        sugg = st.selectbox("Or choose a suggestion:", [""] + suggestions)
        if sugg and not query:
            query = sugg
            
    # --- Filters ---
    with st.expander("🛠️ Filters & Settings"):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            all_types = store.file_types
            filter_type = st.selectbox("File Type", ["All"] + all_types)
        with f_col2:
            all_docs = store.unique_documents
            filter_doc = st.selectbox("Specific Document", ["All"] + all_docs)
        with f_col3:
            top_k = st.slider("Top Results", min_value=1, max_value=20, value=5)
            
    # --- Execute Search ---
    if st.button("Search", type="primary") or query:
        if not query.strip():
            st.warning("Please enter a query to search.")
        elif store.total_chunks == 0:
            st.warning("Database is empty. Please upload some documents first.")
        else:
            with st.spinner("Searching for semantic matches..."):
                query_embedding = embed_query(query)
                
                kwargs = {"top_k": top_k}
                if filter_type != "All": kwargs["filter_type"] = filter_type
                if filter_doc != "All": kwargs["filter_doc"] = filter_doc
                
                results = store.search(query_embedding, **kwargs)
                
                st.session_state.analytics["searches_performed"] += 1
                st.session_state.search_history.insert(0, {
                    "query": query, 
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "results_count": len(results)
                })
                # Keep history short
                st.session_state.search_history = st.session_state.search_history[:50]
                
            if not results:
                st.info("No matching results found for your query.")
            else:
                st.success(f"Found {len(results)} relevant results.")
                
                # --- Export Button ---
                csv_data = export_results_csv(results)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name="search_results.csv",
                    mime="text/csv",
                )
                
                # --- Display Results ---
                for idx, res in enumerate(results):
                    score = res["score"]
                    # Map cosine similarity (approx 0 to 1) to percentage safely
                    score_pct = max(0, min(100, int(score * 100)))
                    
                    highlighted_chunk = highlight_text(res["text"], query)
                    
                    # Construct result card HTML
                    card_html = f"""
                    <div class="result-card">
                        <div class="doc-title">{res['doc_name']}</div>
                        <div class="chunk-meta">
                            <span class="badge">{res['file_type'].upper()}</span>
                            <span>CHUNK {res['chunk_idx']}</span>
                            <span>•</span>
                            <span>SCORE: {score:.3f}</span>
                        </div>
                        <div class="score-container">
                            <div class="score-bar" style="width: {score_pct}%;"></div>
                        </div>
                        <div class="chunk-text">
                            {highlighted_chunk}
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Area: Analytics Dashboard
# -----------------------------------------------------------------------------
elif app_mode == "📊 Analytics Dashboard":
    st.title("📊 System Analytics & Insights")
    st.markdown("Overview of the Intelligent Document Search Engine's performance and data.")
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <h3>{len(store.unique_documents)}</h3>
            <p>Total Documents Indexed</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <h3>{store.total_chunks}</h3>
            <p>Total Semantic Chunks</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <h3>{st.session_state.analytics['searches_performed']}</h3>
            <p>Searches Performed</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    
    # Document Inventory
    st.subheader("📑 Indexed Documents")
    if store.unique_documents:
        inventory_data = [{"Document Name": d, "File Type": os.path.splitext(d)[1]} for d in store.unique_documents]
        df_inv = pd.DataFrame(inventory_data)
        st.dataframe(df_inv, use_container_width=True)
    else:
        st.info("No documents have been uploaded yet.")
        
    # Search History
    st.subheader("🕒 Search History")
    if st.session_state.search_history:
        history_df = pd.DataFrame(st.session_state.search_history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No searches performed yet.")
        
    # Export Analytics
    if st.button("📥 Export Analytics Report"):
        report = f"Analytics Report - {datetime.datetime.now()}\n"
        report += f"Total Documents: {len(store.unique_documents)}\n"
        report += f"Total Chunks: {store.total_chunks}\n"
        report += f"Total Searches: {st.session_state.analytics['searches_performed']}\n\n"
        report += "Search History:\n"
        for item in st.session_state.search_history:
            report += f"- [{item['timestamp']}] '{item['query']}' ({item['results_count']} results)\n"
            
        st.download_button(
            label="Download TXT Report",
            data=report.encode("utf-8"),
            file_name="analytics_report.txt",
            mime="text/plain"
        )
