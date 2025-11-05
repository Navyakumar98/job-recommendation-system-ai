# Understanding "Ground Truth" and Verifying Recommendations

In machine learning, the "ground truth" is the objective, correct answer that you want your model to predict. For a spam filter, the ground truth is whether an email is *actually* spam. For a weather forecast model, it's the weather that *actually* happened.

In this job recommendation system, the ground truth is the **ideal ranking of jobs for a specific user's resume**.

## How do we define the "ideal ranking"?

In a perfect scenario, a team of expert human recruiters would manually review every job in our dataset against a user's resume. They would then create a perfectly ordered list from most to least relevant. This manually created, expert-verified list would be our ground truth.

However, creating such a list is not practical. Instead, our system uses a combination of semantic similarity and user feedback to create a ranking that *approximates* this ideal ground truth.

## Verifying the Recommendations

You can verify the system's effectiveness by observing the two types of recommendations it produces.

### 1. Initial Recommendations (Old Recommendations)

*   **How it Works:** The initial recommendations are based purely on **semantic similarity**. The system uses a powerful language model (`paraphrase-MiniLM-L6-v2`) to understand the meaning and context of the text in your resume and in the job descriptions. It then calculates a `similarity` score and shows you the jobs that are most textually similar to your resume.

*   **How to Verify:**
    1.  Upload your resume.
    2.  Click "Recommend Jobs".
    3.  Examine the top 5-10 results.
    4.  Manually compare the job duties and required skills with the content of your resume. You should see a clear overlap in keywords, technologies, and responsibilities. This confirms that the model is correctly identifying jobs that are a good match "on paper."

### 2. Enhanced Recommendations (Feedback-Aware)

*   **How it Works:** This is where the system truly learns. When you rate the initial recommendations, you are providing a **proxy for the ground truth**. You are telling the model what *you* consider to be a relevant or irrelevant job. The system takes these ratings and creates a "user profile vector" that captures your preferences. The `adjusted_score` for the enhanced recommendations is a blend of the initial resume similarity and this new user preference score.

*   **How to Verify:**
    1.  After getting the initial recommendations, use the sliders to rate several jobs. Give high ratings (e.g., 5 stars) to jobs that are very relevant and low ratings (e.g., 1 star) to jobs that are not.
    2.  Click "Enhance Model with Feedback".
    3.  Observe the "Old vs Enhanced" comparison. You should see that the jobs you rated highly have moved up in the "Enhanced" list, while the jobs you rated poorly have moved down or disappeared.
    4.  This change demonstrates that the model is successfully learning from your feedback and adapting its ranking to better match your personal preferences.

## Understanding the Metrics

The dashboard displays two key metrics to help you understand the change between the old and new rankings:

*   **NDCG (Normalized Discounted Cumulative Gain):** This metric evaluates the quality of the new *ranking*. It checks if the most relevant jobs (based on the original `similarity` score) are placed higher up in the new list. A score closer to 1.0 means the new ranking is still respecting the original relevance scores, which is generally good.

*   **Spearman-R (Spearman's Rank Correlation):** This metric measures how similar the *order* of the two lists is. A score of 1.0 would mean the order is identical. A lower score indicates that your feedback has caused the list to be re-ordered. After providing feedback, you should expect this score to be less than 1.0, as the goal was to change the order to better suit your preferences.

By following these verification steps, you can confidently demonstrate that the recommendation system is working as intended, both in its initial analysis and its ability to adapt and learn from user feedback.
