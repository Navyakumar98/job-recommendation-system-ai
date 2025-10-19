import os
import string
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score

# -------------------- EMBEDDING MODEL (loaded once) --------------------
MODEL = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cpu")
# Optional lightweight speedup on CPU
MODEL = torch.quantization.quantize_dynamic(MODEL, {torch.nn.Linear}, dtype=torch.qint8)


def _spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Spearman rank correlation without external deps.
    Returns in [-1, 1]. Handles constant arrays.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    a_ranks = pd.Series(a).rank(method="average").to_numpy()
    b_ranks = pd.Series(b).rank(method="average").to_numpy()
    a_center = a_ranks - a_ranks.mean()
    b_center = b_ranks - b_ranks.mean()
    denom = (np.linalg.norm(a_center) * np.linalg.norm(b_center)) + 1e-9
    return float(np.dot(a_center, b_center) / denom)


def _ensure_data_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


class JobRecommendationSystem:
    def __init__(self, jobs_csv: str):
        """
        Load job data and precompute embeddings + FAISS index.
        Required columns in CSV:
          - 'Job Id', 'workplace', 'working_mode', 'position',
            'job_role_and_duties', 'requisite_skill'
        Optional: 'salary', 'offer_details'
        """
        if not os.path.exists(jobs_csv):
            raise FileNotFoundError(f"Jobs CSV not found: {jobs_csv}")

        self.jobs_df = pd.read_csv(jobs_csv)
        # Unified text field for embedding
        self.jobs_df["job_text"] = (
            self.jobs_df["workplace"].astype(str) + " " +
            self.jobs_df["working_mode"].astype(str) + " " +
            self.jobs_df["position"].astype(str) + " " +
            self.jobs_df["job_role_and_duties"].astype(str) + " " +
            self.jobs_df["requisite_skill"].astype(str)
        )

        self.job_info = self.jobs_df.copy()
        self.jobs_texts = self.jobs_df["job_text"].tolist()

        # Precompute job embeddings
        self.job_embeddings = MODEL.encode(self.jobs_texts, convert_to_numpy=True).astype(np.float16)

        # FAISS index (inner product)
        self.dim = int(self.job_embeddings.shape[1])
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.job_embeddings.astype(np.float16))

        # Cache: user profile vector learned from feedback
        self.user_profile_vector = None

        # Metrics log path
        self.metrics_path = os.path.join("data", "metrics.csv")

    # -------------------- Helpers --------------------
    def clean_text(self, text: str) -> str:
        return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

    def filter_top_jobs(self, resume_text: str, top_n: int = 100):
        """
        Fast prefilter using TF-IDF to keep the top-N candidates before FAISS search.
        Returns (filtered_texts, filtered_df, filtered_embeddings)
        """
        vectorizer = TfidfVectorizer()
        job_vectors = vectorizer.fit_transform(self.jobs_texts)
        resume_vector = vectorizer.transform([resume_text])
        similarity_scores = (job_vectors @ resume_vector.T).toarray().flatten()
        top_indices = np.argsort(similarity_scores)[-top_n:]
        return (
            [self.jobs_texts[i] for i in top_indices],
            self.job_info.iloc[top_indices].reset_index(drop=True),
            self.job_embeddings[top_indices],
        )

    def load_feedback_embeddings(self, feedback_file: str = "data/ratings.csv"):
        """
        Load ratings and return (embeddings, ratings, merged_df).
        - ratings must be numeric (1..5)
        """
        if not os.path.exists(feedback_file):
            return None, None, None

        df = pd.read_csv(feedback_file)
        if df.empty or "rating" not in df.columns:
            return None, None, None

        # Ensure numeric ratings and string ids for join safety
        df = df[df["rating"].astype(str).str.isnumeric()].copy()
        if df.empty:
            return None, None, None
        df["rating"] = df["rating"].astype(float)

        # Join with job info to build embeddings for rated jobs
        merged = pd.merge(df, self.job_info, left_on="job_id", right_on="Job Id", how="inner")
        if merged.empty:
            return None, None, None

        merged["job_text"] = (
            merged["workplace"].astype(str) + " " +
            merged["working_mode"].astype(str) + " " +
            merged["position"].astype(str) + " " +
            merged["job_role_and_duties"].astype(str) + " " +
            merged["requisite_skill"].astype(str)
        )
        job_embeds = MODEL.encode(merged["job_text"].tolist(), convert_to_numpy=True).astype(np.float16)
        ratings = merged["rating"].to_numpy(dtype=np.float16)
        return job_embeds, ratings, merged

    @staticmethod
    def _alpha_from_num_ratings(n_ratings: int) -> float:
        """
        Adaptive blend between resume similarity (alpha) and feedback (1-alpha).
        - Start at alpha≈0.9 with few ratings; decay to 0.5 as ratings grow.
        """
        alpha = 0.9 - 0.04 * n_ratings  # each rating reduces resume weight by 0.04
        return float(np.clip(alpha, 0.5, 0.9))

    # -------------------- Core recommend --------------------
    def recommend_jobs(self, resume_text: str, top_n: int = 20, use_feedback: bool = True):
        """
        Recommend jobs for a given resume.
        If use_feedback=True and ratings exist, apply feedback-driven re-ranking with:
          - adaptive resume vs feedback blending
          - small skill-overlap boost
        Returns dict with 'recommended_jobs' (list of dict rows).
        """
        resume_text = self.clean_text(resume_text)

        # TF-IDF prefilter
        filtered_texts, filtered_df, filtered_embeds = self.filter_top_jobs(
            resume_text, top_n=max(100, top_n * 3)
        )

        # Embed resume
        resume_embedding = MODEL.encode([resume_text], convert_to_numpy=True).astype(np.float16)

        # FAISS over filtered set
        index = faiss.IndexFlatIP(self.dim)
        index.add(filtered_embeds.astype(np.float16))
        distances, indices = index.search(resume_embedding.astype(np.float16), top_n)

        # Normalize base similarities to [0,1]
        base_sims = distances[0]
        sims_norm = (base_sims - base_sims.min()) / (base_sims.max() - base_sims.min() + 1e-9)

        recs = filtered_df.iloc[indices[0]].copy()
        recs["similarity"] = sims_norm

        # Skill overlap (simple token intersection)
        resume_words = set(resume_text.split())
        recs["matched_skills"] = recs["requisite_skill"].apply(
            lambda x: ", ".join(list(resume_words.intersection(set(str(x).split())))[:8])
        )
        recs["skill_overlap"] = recs["matched_skills"].apply(
            lambda s: 0 if pd.isna(s) or str(s).strip() == "" else len([t for t in str(s).split(",") if t.strip()])
        )

        if not use_feedback:
            # Add minimal adjusted_score for consistent schema
            # (still sorted by similarity already)
            recs["adjusted_score"] = recs["similarity"]
            return {"recommended_jobs": recs.to_dict(orient="records")}

        # Feedback-driven personalization
        rated_embeds, ratings, _ = self.load_feedback_embeddings()
        if rated_embeds is None:
            recs["adjusted_score"] = recs["similarity"]
            return {"recommended_jobs": recs.to_dict(orient="records")}

        # Weighted user preferences vector
        norm_r = (ratings - ratings.min()) / (ratings.max() - ratings.min() + 1e-9)
        user_vec = np.average(rated_embeds, axis=0, weights=norm_r).astype(np.float16)
        self.user_profile_vector = user_vec  # cache for later

        # Cosine similarity to user preference for the selected items
        filtered_sel = filtered_embeds[indices[0]]
        up_sim = np.dot(filtered_sel, user_vec) / (
            np.linalg.norm(filtered_sel, axis=1) * np.linalg.norm(user_vec) + 1e-9
        )

        # Adaptive weighting between resume similarity and user preference
        alpha = self._alpha_from_num_ratings(len(ratings))          # resume weight
        beta = 1.0 - alpha                                          # feedback weight

        # Small skill-overlap boost (cap count at 5 so boost <= 0.1)
        # Rationale: reward explicit skill matches without overpowering semantics
        skill_boost = 0.02 * np.minimum(recs["skill_overlap"].to_numpy(dtype=float), 5.0)

        recs["adjusted_score"] = alpha * recs["similarity"] + beta * up_sim + skill_boost
        recs = recs.sort_values(by="adjusted_score", ascending=False)
        return {"recommended_jobs": recs.to_dict(orient="records")}

    # -------------------- Retrain / Evaluate --------------------
    def retrain_with_feedback(self, resume_text: str, top_n: int = 20):
        """
        Compare old vs enhanced results and compute metrics:
          - NDCG@K (k=top_n) between old (similarity) and new (adjusted_score)
          - Spearman-R between old & new scores for matched rows
          - Reordered % (how many items changed position among common ids)
        Returns dict with lists and metrics, and logs metrics to data/metrics.csv.
        """
        # Old: no feedback
        old = pd.DataFrame(self.recommend_jobs(resume_text, top_n=top_n, use_feedback=False)["recommended_jobs"])
        # New: with feedback
        new = pd.DataFrame(self.recommend_jobs(resume_text, top_n=top_n, use_feedback=True)["recommended_jobs"])

        # Align on Job Id to compare scores directly
        comp = old[["Job Id", "position", "similarity"]].merge(
            new[["Job Id", "adjusted_score"]], on="Job Id", how="outer"
        )

        # Metrics (fill NAs with 0 for fair comparison)
        y_true = comp["similarity"].fillna(0).to_numpy()
        y_score = comp["adjusted_score"].fillna(0).to_numpy()

        # NDCG@top_n
        ndcg = float(ndcg_score([y_true], [y_score]))

        # Spearman rank correlation
        spear = _spearman_r(y_true, y_score)

        # Reordered percentage (by job id order change)
        old_order = {jid: i for i, jid in enumerate(old["Job Id"].tolist())}
        new_order = {jid: i for i, jid in enumerate(new["Job Id"].tolist())}
        common_ids = [jid for jid in old_order if jid in new_order]
        moved = sum(1 for jid in common_ids if old_order[jid] != new_order[jid])
        reordered_pct = (moved / max(1, len(common_ids))) * 100.0

        # Log metrics over time
        _ensure_data_dir(self.metrics_path)
        log_row = pd.DataFrame([{
            "timestamp": pd.Timestamp.now().isoformat(),
            "ndcg_at_k": round(ndcg, 4),
            "spearman_r": round(float(spear), 4),
            "reordered_pct": round(reordered_pct, 2),
            "k": top_n
        }])
        if os.path.exists(self.metrics_path):
            log_row.to_csv(self.metrics_path, mode="a", header=False, index=False)
        else:
            log_row.to_csv(self.metrics_path, mode="w", header=True, index=False)

        return {
            "old_jobs": old.to_dict(orient="records"),
            "new_jobs": new.to_dict(orient="records"),
            "comparison": comp,  # DataFrame
            "metrics": {
                "ndcg_at_k": round(ndcg, 3),
                "spearman_r": round(float(spear), 3),
                "reordered_pct": round(reordered_pct, 1),
            },
        }

    def get_metrics_history(self):
        """Return metrics history DataFrame if available."""
        if not os.path.exists(self.metrics_path):
            return pd.DataFrame(columns=["timestamp", "ndcg_at_k", "spearman_r", "reordered_pct", "k"])
        return pd.read_csv(self.metrics_path)
