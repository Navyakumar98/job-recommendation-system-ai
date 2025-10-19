# 💼 JobGenie: Adaptive NLP-Powered Job Recommendation Platform

**JobGenie** is an AI-driven job recommendation engine that analyzes resumes and suggests the most relevant job opportunities based on your **skills, experience, and career background**.  
It leverages **Natural Language Processing (NLP)** and **semantic similarity search** to deliver accurate, explainable, and personalized job matches.

---

## 🚀 Key Features

- 📄 **Resume Upload** — Upload your PDF resume directly through the web interface.
- 🤖 **AI-Based Analysis** — Uses `SentenceTransformer` embeddings to understand the true context of your skills and experience.
- ⚡ **Fast & Scalable Search** — Powered by `FAISS` (Facebook AI Similarity Search) for efficient large-scale similarity lookup.
- 🧠 **Hybrid Filtering** — Combines `TF-IDF` keyword filtering and semantic similarity for more precise job matching.
- 🪄 **Streamlit Web App** — A clean, interactive UI that makes exploring recommendations easy and enjoyable.

---

## 🧰 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| **Programming Language** | Python 3.10+ |
| **Web Framework** | Streamlit |
| **AI & NLP** | SentenceTransformers (MiniLM-L6-v2) |
| **Vector Search** | FAISS |
| **Text Extraction** | PyMuPDF (fitz) |
| **Feature Filtering** | scikit-learn (TF-IDF) |
| **Data Handling** | pandas, numpy |

---

## ⚙️ Setup & Usage

###

1️⃣ Clone the Repository
git clone https://github.com/Navyakumar98/job-recommendation-system-ai.git

  cd job-recommendation-system-ai

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit Application
streamlit run app.py

4️⃣ Open the App

Once the server starts, open the URL shown in your terminal (usually http://localhost:8501) to start exploring your personalized job recommendations.


