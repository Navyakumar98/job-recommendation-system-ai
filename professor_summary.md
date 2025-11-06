# AI-Powered Job Recommendation System: Project Summary

## Overview

This project is an AI-powered job recommendation system designed to provide a personalized and adaptive job search experience. The system is built as a Streamlit web application that leverages Natural Language Processing (NLP) to match candidate resumes with job descriptions. The core of the system is a content-based filtering approach that uses semantic similarity to provide highly relevant recommendations.

## Methodology

The system's methodology can be broken down into the following key stages:

1.  **Data Preprocessing:** The system begins by loading a dataset of job listings from a CSV file sourced from Kaggle. It then creates a unified text field for each job by combining key attributes like the job title, duties, and skills. This text, along with the user's uploaded resume, is then cleaned by converting it to lowercase and removing punctuation.

2.  **Text Embedding:** We use a pre-trained `sentence-transformer` model (`paraphrase-MiniLM-L6-v2`) to convert both the job descriptions and the user's resume into high-dimensional vector representations, or "embeddings." These embeddings are crucial as they capture the semantic meaning of the text, allowing for a more nuanced comparison than simple keyword matching.

3.  **Similarity Search:** To efficiently find the most similar jobs for a given resume, we use Facebook AI Similarity Search (FAISS). A FAISS index is built from the job embeddings, which enables fast and scalable similarity searches.

4.  **Ranking and Personalization:** The system employs a multi-faceted approach to ranking and personalization:
    *   **TF-IDF Prefiltering:** A quick pre-filtering step using TF-IDF is performed to narrow down the search space to a smaller, more relevant subset of jobs.
    *   **Feedback-Driven Re-ranking:** The system incorporates user feedback (ratings on recommended jobs) to create a "user profile vector." This vector represents the user's preferences and is used to adjust the ranking of future recommendations.
    *   **Heuristic-Based Scoring:** In addition to semantic similarity, the system calculates scores based on location, salary, and experience to provide a more holistic and practical ranking.

## How It Works and Why It's Effective

The final output of the system is a ranked list of job recommendations that are tailored to the user's resume. The system is effective for several reasons:

*   **Semantic Understanding:** By using text embeddings, the system goes beyond simple keyword matching to understand the underlying meaning and context of the resume and job descriptions.
*   **Efficiency:** The use of FAISS ensures that the system can handle a large number of job listings without sacrificing performance.
*   **Personalization:** The feedback loop allows the system to learn from user interactions and continuously improve its recommendations over time.
*   **Holistic Approach:** The combination of semantic similarity, user feedback, and practical heuristics provides a robust and realistic solution that addresses the real-world needs of job seekers.

## Conclusion

The AI-Powered Job Recommendation System is a powerful and effective tool for connecting job seekers with relevant opportunities. Its architecture is designed to be accurate, efficient, and adaptable, making it a valuable asset in the modern job market.
