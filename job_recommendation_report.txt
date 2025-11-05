**Job Recommendation System Report**

**Abstract**

This report details the functionality and architecture of the AI-powered Job Recommendation System. The system leverages Natural Language Processing (NLP) to match candidate resumes with job descriptions, providing a personalized and adaptive job search experience. The core of the system is a content-based filtering approach, where job and resume texts are converted into high-dimensional vectors (embeddings) to calculate semantic similarity. The system is designed to learn from user feedback, continuously improving the quality of its recommendations. This document covers the system's introduction, methodology, results, and a discussion of its components, including a recent bug fix related to data parsing.

**Introduction**

The Job Recommendation System is a Streamlit web application designed to bridge the gap between job seekers and relevant opportunities. Unlike traditional keyword-based search engines, this system understands the underlying meaning and context of a user's resume to provide more accurate and relevant job recommendations. The application allows users to upload their resumes, receive a list of ranked job openings, and provide feedback to refine future recommendations. This report provides a comprehensive overview of the system's inner workings, from data processing to the user interface.

**Method/Procedure**

The system follows a multi-stage process to generate job recommendations:

1.  **Data Loading and Preprocessing:** The system starts by loading a dataset of job listings from a CSV file. It combines several job attributes (e.g., title, duties, skills) into a single text field for each job. This unified text is then cleaned to remove noise (e.g., punctuation, lowercase conversion).

2.  **Text Embedding:** The core of the recommendation engine is the use of sentence-transformer models (specifically, `paraphrase-MiniLM-L6-v2`) to convert both the job descriptions and the user's resume into dense vector representations called embeddings. These embeddings capture the semantic meaning of the text.

3.  **Similarity Search with FAISS:** To efficiently find the most similar jobs for a given resume, the system uses Facebook AI Similarity Search (FAISS). A FAISS index is built from the job embeddings, allowing for fast and scalable similarity searches.

4.  **TF-IDF Prefiltering:** To optimize performance, the system first uses a TF-IDF vectorizer to perform a quick pre-filtering of jobs. This narrows down the search space to a smaller, more relevant subset of jobs before the more computationally intensive embedding-based search is performed.

5.  **Feedback-Driven Re-ranking:** The system incorporates user feedback to personalize recommendations. When a user rates a recommended job, this rating is used to create a "user profile vector." This vector represents the user's preferences, and it is used to adjust the ranking of future recommendations. The system adaptively blends the initial resume-based similarity with the feedback-based user profile similarity.

6.  **Heuristic-Based Scoring:** In addition to semantic similarity, the system also calculates scores based on location, salary, and experience. These scores are combined with the similarity score to produce a final "adjusted score," which is used to rank the jobs.

7.  **Bug Fix Explanation:** The `KeyError: 'timestamp'` occurred because the `data/metrics.csv` file, which logs the performance of the recommendation model over time, was being saved without a header row. When the `get_metrics_history` function in `model.py` tried to read this file using `pd.read_csv()`, pandas incorrectly inferred the first row of data as the header. Consequently, the column names were incorrect, and the code was unable to find the 'timestamp' column.

    The fix involved modifying the `get_metrics_history` function to read the CSV without a header (`header=None`) and then explicitly assign the correct column names from a predefined schema. This ensures that the DataFrame is always correctly structured, regardless of whether the CSV file has a header.

**Results**

The system produces a ranked list of job recommendations based on the user's resume. The "Enhance with AI Feedback" feature allows the system to retrain its model based on user ratings and display metrics that quantify the improvement in recommendation quality. These metrics include:

*   **NDCG@20 (Normalized Discounted Cumulative Gain):** A measure of ranking quality.
*   **Spearman-R:** A measure of the correlation between the ranks of the old and new recommendations.
*   **Reordered %:** The percentage of jobs that changed their position in the rankings after feedback was incorporated.

**Discussions**

The hybrid approach of combining semantic search with user feedback and heuristics (location, salary, experience) makes the system robust and adaptable. The use of FAISS ensures that the system can scale to handle a large number of job listings without sacrificing performance. The feedback loop is a critical component, as it allows the system to learn from user interactions and improve its recommendations over time.

The recent bug fix highlights the importance of robust data handling. In a real-world system, data schemas can change, and files can become corrupted. The implemented fix makes the system more resilient to such issues.

**Conclusions**

The AI-powered Job Recommendation System is an effective tool for connecting job seekers with relevant opportunities. Its architecture is designed to be both accurate and efficient, and its ability to learn from user feedback makes it a powerful and personalized job search assistant. Future improvements could include incorporating more sophisticated NLP models, expanding the range of heuristics used for scoring, and adding more features to the user interface.

**Appendices**

*   **`app.py`:** The main Streamlit application file that handles the user interface and user interactions.
*   **`model.py`:** The core of the recommendation system, containing the `JobRecommendationSystem` class, which encapsulates the logic for data loading, embedding, searching, and re-ranking.
*   **`data/metrics.csv`:** A log file that stores the performance metrics of the recommendation model over time.
*   **`JobsFE.csv`:** The dataset of job listings used by the system.
