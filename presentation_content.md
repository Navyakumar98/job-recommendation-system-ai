# AI-Powered Job Recommendation System

## Slide 1: Title Slide

**AI-Powered Job Recommendation System**

*A Deep Dive into the Technology and a Kaggle Dataset*

**Presenter:** [Your Name]

---

## Slide 2: Project Overview

**Introduction to the Job Recommendation System**

*   **Problem:** Bridging the gap between job seekers and relevant opportunities.
*   **Solution:** A Streamlit web application that provides personalized job recommendations based on resume analysis.
*   **Core Technology:** Natural Language Processing (NLP) and semantic search.

---

## Slide 3: Dataset Overview

**Dataset Source: Kaggle**

*   **Dataset Name:** "Global Job Postings Dataset"
*   **Description:** A comprehensive collection of job postings with details on roles, responsibilities, skills, and compensation.
*   **Key Features:**
    *   `Job Id`: Unique identifier for each job.
    *   `workplace`: Location of the job.
    *   `working_mode`: Full-time, part-time, contract, etc.
    *   `salary`: Compensation for the role.
    *   `position`: Job title.
    *   `job_role_and_duties`: Detailed description of the job.
    *   `requisite_skill`: Required skills for the job.
    *   `offer_details`: Additional benefits and perks.

---

## Slide 4: Methodology - Feature Extraction & Preprocessing

**1. Unified Text Field**

*   Combined `workplace`, `working_mode`, `position`, `job_role_and_duties`, and `requisite_skill` into a single "job\_text" field for a holistic representation of each job.

**2. Text Cleaning**

*   **Lowercasing:** To ensure case-insensitive matching.
*   **Punctuation Removal:** To reduce noise in the text.

**3. Resume Processing**

*   Applied the same cleaning steps to the user's uploaded resume to ensure consistency.

---

## Slide 5: Methodology - Core Algorithms

**1. Text Embedding**

*   **Algorithm:** `sentence-transformers` (specifically `paraphrase-MiniLM-L6-v2`).
*   **Purpose:** To convert text (both job descriptions and resumes) into high-dimensional vectors (embeddings) that capture the semantic meaning of the text.

**2. Similarity Search**

*   **Algorithm:** Facebook AI Similarity Search (FAISS).
*   **Purpose:** To efficiently find the most similar job embeddings to a given resume embedding, enabling fast and scalable recommendations.

---

## Slide 6: Methodology - Ranking and Personalization

**1. TF-IDF Prefiltering**

*   **Purpose:** To quickly narrow down a subset of the most relevant jobs before performing the more computationally intensive embedding search.

**2. Feedback-Driven Re-ranking**

*   **Mechanism:** Incorporates user feedback (ratings) to create a "user profile vector."
*   **Benefit:** Personalizes recommendations over time by learning the user's unique preferences.

**3. Heuristic-Based Scoring**

*   **Factors:** Location, Salary, and Experience.
*   **Purpose:** To provide a more holistic and practical ranking of jobs by considering real-world constraints.

---

## Slide 7: Training and Testing (Ground Truth)

**No Explicit Training/Testing Split**

*   This is a content-based filtering system, not a traditional supervised learning model that requires a training/testing split.

**Ground Truth**

*   The "ground truth" is implicitly defined by the **semantic similarity** between a user's resume and the job descriptions. A "good" recommendation is one that is semantically close to the user's resume.
*   The system's performance is evaluated through **user feedback**, which serves as a form of online evaluation and continuous improvement.

---

## Slide 8: Results and How It Works

**Final Output**

*   A ranked list of job recommendations tailored to the user's resume.

**Why It Works**

*   **Semantic Understanding:** Goes beyond simple keyword matching to understand the *meaning* of the resume and job descriptions.
*   **Efficiency:** FAISS enables fast searching, even with a large number of job listings.
*   **Personalization:** The feedback loop continuously improves the recommendations based on user interactions.
*   **Holistic Approach:** The combination of semantic similarity, user feedback, and practical heuristics provides a robust and realistic solution.

---

## Slide 9: Conclusion

**Summary**

*   The AI-Powered Job Recommendation System is an effective tool for connecting job seekers with relevant opportunities.
*   Its architecture is designed to be accurate, efficient, and adaptable.

**Future Work**

*   Incorporate more sophisticated NLP models (e.g., larger transformers).
*   Expand the range of heuristics used for scoring.
*   Enhance the user interface for a more interactive experience.

---

## Slide 10: Q&A

**Questions?**
