# 🧠 Intelligent Document Search Engine

A production-ready semantic search engine built with Python, Streamlit, Sentence Transformers, and FAISS. This application allows you to upload documents (PDF, DOCX, TXT) and search through them based on **meaning** rather than exact keywords.

## ✨ Features
- **Multi-Format Support:** Ingests PDF, DOCX, and TXT files.
- **Semantic Search:** Uses `sentence-transformers/all-MiniLM-L6-v2` to understand query meaning.
- **Lightning Fast Retrieval:** Uses FAISS (Facebook AI Similarity Search) for optimized vector search.
- **Modern UI:** Premium, responsive Streamlit dashboard with custom CSS, visual progress bars, and document metadata tags.
- **Keyword Highlighting:** Visually highlights keywords matched within the semantic chunks in the UI.
- **Analytics Dashboard:** Tracks total documents, chunk counts, and search history.
- **Export Results:** Download your search hits as a clean CSV or your analytics as a TXT report.

---

## 🚀 How to Run Locally

### 1. Prerequisites
- Python 3.10+
- Virtual Environment (recommended)

### 2. Setup
Clone the repository and install the dependencies:
```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
# Activate it (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## ☁️ How to Deploy on Streamlit Cloud

1. **Push to GitHub:** Commit your code (`app.py`, `utils/`, `requirements.txt`) and push to a public or private GitHub repository. (Note: Do not commit the `data/` folder if you don't want to pre-load documents).
2. **Log into Streamlit Community Cloud:** Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. **Deploy New App:** Click "New app", select your repository, branch, and specify `app.py` as the main file path.
4. **Launch:** Click "Deploy". Streamlit Cloud will automatically read `requirements.txt`, install dependencies, and host your search engine for free.

---

## 📸 Screenshots Description

If you are including this in a project report, here is how you can describe your application UI:

1. **Main Search Interface:** A sleek, dual-column design. The left sidebar handles document ingestion with drag-and-drop file uploading and database controls. The main area contains a large search bar, query suggestions, and filter dropdowns (File Type, Specific Document).
2. **Result Cards:** Displays hits in a card-based layout featuring the Document Title in bold blue, a green gradient "Similarity Score" progress bar, and the exact extracted text with keywords highlighted in yellow for quick reading.
3. **Analytics Dashboard:** A clean, metrics-driven view featuring three main KPI boxes (Total Documents, Total Chunks, Searches Performed). Below are interactive pandas dataframes showing the inventory of uploaded files and chronological search history.

---

## 🎓 Viva Explanation (For College Final Year Presentation)

**Q: What is the difference between keyword search and semantic search?**
*Answer:* Keyword search relies on exact string matching (e.g., searching for "car" won't match "automobile"). Semantic search uses machine learning models (Sentence Transformers) to convert text into mathematical vectors (embeddings) representing the *meaning* of the text. This means searching for "car" will accurately retrieve documents mentioning "automobile" or "vehicle" because their vectors are close together in the vector space.

**Q: Why did you choose FAISS?**
*Answer:* FAISS (Facebook AI Similarity Search) is an industry-standard library highly optimized for clustering and searching dense vectors. When we process large documents, we generate thousands of chunks. Comparing a query against all of them using a standard loop would be very slow (O(N) complexity). FAISS indexes the vectors and performs ultra-fast nearest-neighbor search, ensuring our UI responds in milliseconds.

**Q: How do you process large PDFs?**
*Answer:* We extract the raw text using `pdfplumber`/`PyPDF2`. Since transformers have a context limit (e.g., 512 tokens), we pass the raw text through a "text chunker" (`utils/preprocessor.py`). It splits the document into overlapping chunks of ~400 words. The overlap (e.g., 50 words) prevents important sentences from being cut off at the boundary. Each chunk is then individually embedded and indexed.

**Q: How is the state managed in this Streamlit app?**
*Answer:* We use `st.session_state` to persist the FAISS `VectorStore`, the search history, and analytics variables across user interactions. Without this, Streamlit would reset the database and variables every time a button is clicked.

## 📂 Folder Structure Explanation
- `app.py`: The main frontend application file containing UI layouts, styling, and application flow.
- `utils/text_extractor.py`: Utility to read bytes from uploaded PDFs/DOCXs/TXTs and convert them to plain text.
- `utils/preprocessor.py`: Cleans raw text (removes weird spacing) and handles the overlapping window chunking logic.
- `utils/embedder.py`: Handles loading the `all-MiniLM-L6-v2` transformer and converting text strings into float32 numpy arrays.
- `utils/vector_store.py`: Wraps the FAISS library to handle adding new vectors, saving to disk (`data/faiss.index`), and querying the nearest neighbors.
- `data/`: Local storage directory where the vector index and metadata are saved to persist between app restarts.
