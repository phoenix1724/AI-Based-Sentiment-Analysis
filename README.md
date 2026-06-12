# AI-Based Sentiment Analysis Dashboard (Mini Project) — 2026

An interactive, high-fidelity customer feedback analytics dashboard. This application leverages a hybrid Natural Language Processing (NLP) pipeline—combining lexicon-based sentiment analysis (**VADER**) and a tuned classical machine learning model (**TF-IDF + Logistic Regression**)—to classify textual feedback (Positive, Neutral, Negative) and identify actionable user concerns.

Served via **FastAPI** with a modern, dark-themed **glassmorphism** web interface.

---

## 🚀 Key Features

*   **Real-time Sentiment Sandbox:** Type or paste review text to see instant sentiment classifications. Includes a **Preprocessing Stage Visualizer** showing how text is lowercase-converted, tokenized, punctuation-cleaned, stopword-filtered, and lemmatized.
*   **Interactive Analytics Dashboard:** Real-time visual metrics including total feedback processed, positive/neutral/negative ratios, a satisfaction score gauge (-1.00 to +1.00), and top keyword drivers for both positive and negative sentiments.
*   **Bulk CSV Upload & Export:** Upload custom feedback spreadsheets (CSV). Features smart column auto-detection (filters out metadata and scores text columns) with pagination previews of up to 100 rows, and exports a fully labeled CSV file with VADER scores and ML sentiment tags.
*   **Model Details Panel:** Transparently shows active Machine Learning classifier statistics (validation accuracy: **86.2%**) and word coefficient weights (the mathematical impact of words on positive, neutral, and negative classifications).

---

## 🛠️ Tech Stack

### Backend (Python 3.10+)
*   **FastAPI:** High-performance, modern web framework.
*   **Uvicorn:** Ultra-fast ASGI web server implementation.
*   **NLTK (Natural Language Toolkit):** Preprocessing engines (tokenization, English stopwords, WordNet lemmatizers, and VADER lexicon).
*   **Scikit-Learn (v1.9.0):** Machine learning pipeline (`TfidfVectorizer` + `LogisticRegression`).
*   **Pandas:** High-speed bulk CSV data manipulation.

### Frontend (Modern Vanilla Web Stack)
*   **HTML5 & CSS3:** Semantic structures and custom styling (glassmorphism layouts, glowing accents, keyframe animations, and custom scrollbars).
*   **Vanilla JS (ES6):** Asynchronous API request client, debounced typing event handlers, and drag-and-drop CSV uploads.
*   **Chart.js (via CDN):** Dynamic distribution doughnut charts, word count horizontal bars, and satisfaction gauges.

---

## 📂 Project Structure

```text
├── app.py                # FastAPI backend API and static file serving
├── nlp_engine.py         # NLP text-cleaning pipelines, VADER, and ML model training
├── requirements.txt      # Python package dependencies
├── PRD.md                # Product Requirements Document (detailed specs)
├── README.md             # Project documentation (Setup & Usage)
├── data/
│   └── sample_feedback.csv  # Mock reviews dataset for testing CSV uploads
└── static/
    ├── index.html        # Web page structure and sidebar tabs
    ├── style.css         # Modern glassmorphism stylesheet
    └── script.js         # Frontend fetch client and Chart.js integrations
```

---

## ⚡ Setup & Installation

Follow these steps to run the application locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/phoenix1724/AI-Based-Sentiment-Analysis.git
cd AI-Based-Sentiment-Analysis
```

### 2. Install Dependencies
Make sure you have Python 3.10+ installed. Install the required packages:
```bash
pip install -r requirements.txt
```
*(Note: At startup, the server automatically checks for and downloads necessary NLTK corpora like `punkt`, `stopwords`, `wordnet`, and `vader_lexicon`—approx. 25MB total)*

### 3. Run the Development Server
Start the Uvicorn ASGI server:
```bash
uvicorn app:app --reload
```

### 4. Open the Web App
Open your web browser and navigate to:
```text
http://127.0.0.1:8000
```

---

## 🧠 NLP Preprocessing & ML Models

### Preprocessing Stages
For every feedback input, text is cleaned step-by-step:
1.  **Lowercasing:** Converts text to lowercase and strips out any HTML tags.
2.  **Tokenization:** Splits text blocks into single word tokens.
3.  **Punctuation Stripping:** Removes punctuation characters (`!`, `?`, `.`, `,`, etc.).
4.  **Stopwords Filtering:** Drops common filler words (e.g. *i, me, is, are, the, did*).
5.  **Lemmatization:** Matches words to their base lexical form (e.g., *crashing* $\rightarrow$ *crash*, *fastest* $\rightarrow$ *fast*).

### Sentiment Scoring
*   **VADER Sentiment:** Uses VADER's rule-based sentiment dictionary to evaluate positive/negative intensities. Returns a compound score from `-1.0` (negative) to `+1.0` (positive).
*   **Logistic Regression:** Trained on an inline corpus of 116 curated reviews (42 positive, 42 negative, 32 neutral), optimized with unigram features and `C=10.0` regularization to achieve **86.2%** validation accuracy.
