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


## 📂 Folder Structure Explanation
- `app.py`: The main frontend application file containing UI layouts, styling, and application flow.
- `utils/text_extractor.py`: Utility to read bytes from uploaded PDFs/DOCXs/TXTs and convert them to plain text.
- `utils/preprocessor.py`: Cleans raw text (removes weird spacing) and handles the overlapping window chunking logic.
- `utils/embedder.py`: Handles loading the `all-MiniLM-L6-v2` transformer and converting text strings into float32 numpy arrays.
- `utils/vector_store.py`: Wraps the FAISS library to handle adding new vectors, saving to disk (`data/faiss.index`), and querying the nearest neighbors.
- `data/`: Local storage directory where the vector index and metadata are saved to persist between app restarts.
