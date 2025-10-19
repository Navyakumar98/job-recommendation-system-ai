import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # allow model downloads without SSL errors

import os
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from model import JobRecommendationSystem

# -------------------- CONSTANTS --------------------
RATINGS_DIR = os.path.join(os.getcwd(), "data")
RATINGS_PATH = os.path.join(RATINGS_DIR, "ratings.csv")

# -------------------- INITIAL SETUP --------------------
@st.cache_resource
def load_model():
    return JobRecommendationSystem("JobsFE.csv")

recommender = load_model()

# Initialize session state
for key, default in {
    "feedback": [],
    "resume_text": "",
    "resume_id": None,
    "job_results": [],
    "results_ready": False,
    "last_uploaded_file": None,
    "enhanced_results": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -------------------- HELPERS --------------------
def extract_text_from_pdf(pdf_file):
    """Extract plain text from uploaded PDF resume."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text.strip().lower()

def ensure_ratings_dir():
    """Make sure the data directory exists."""
    os.makedirs(RATINGS_DIR, exist_ok=True)

def save_rating(resume_id, job_id, rating):
    """Save or update numeric rating for (resume_id, job_id)."""
    ensure_ratings_dir()
    feedback_file = RATINGS_PATH
    rating_value = int(rating)

    # Load or create DataFrame
    if os.path.exists(feedback_file):
        df = pd.read_csv(feedback_file)
    else:
        df = pd.DataFrame(columns=["resume_id", "job_id", "rating"])

    # Normalize dtypes
    if not df.empty:
        df["resume_id"] = df["resume_id"].astype(str)
        df["job_id"] = df["job_id"].astype(str)

    new_entry = {"resume_id": str(resume_id), "job_id": str(job_id), "rating": rating_value}

    mask = (df["resume_id"] == str(resume_id)) & (df["job_id"] == str(job_id))
    if mask.any():
        df.loc[mask, "rating"] = rating_value
        st.toast(f"🔄 Updated rating for job {job_id}")
    else:
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        st.toast(f"💾 Added new rating for job {job_id}")

    df["rating"] = df["rating"].astype(int)
    df.to_csv(feedback_file, index=False)

def run_recommend():
    """Generate personalized job recommendations (uses feedback if available)."""
    with st.spinner("🔍 Analyzing your resume and finding best job matches..."):
        results = recommender.recommend_jobs(st.session_state.resume_text, top_n=20, use_feedback=True)
    st.session_state.job_results = results["recommended_jobs"]
    st.session_state.results_ready = True
    st.session_state.enhanced_results = None  # reset enhanced panel
    st.success(f"✅ Found {len(st.session_state.job_results)} matching jobs!")


# -------------------- UI THEME --------------------
st.markdown(
    """
    <style>
        .main { background-color: #f9fafc; }
        h1 { color: #004aad; text-align: center; }
        .stButton>button {
            background-color: #004aad; color: white;
            border-radius: 8px; font-weight: bold; padding: 8px 16px;
        }
        .stButton>button:hover { background-color: #0078ff; }
        .stSlider label { color: #004aad !important; }
        .job-card {
            background-color: #ffffff; border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            padding: 15px; margin-bottom: 20px;
        }
        .metric-pill {
            display:inline-block;padding:6px 10px;border-radius:999px;
            background:#eef4ff;border:1px solid #cfe0ff;margin-right:8px;font-size:12px;
        }
        .tag {
            display:inline-block;margin:2px 6px 0 0;padding:3px 8px;border-radius:999px;
            background:#f1f5f9;border:1px solid #e2e8f0;font-size:12px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- APP HEADER --------------------
st.title("💼 JobGenie: Adaptive NLP-Powered Job Recommendation Platform")
st.markdown("Upload your resume and get **personalized, explainable job matches** instantly 🚀")

uploaded_file = st.file_uploader("📄 Upload your resume (PDF only)", type=["pdf"])

# -------------------- RESET ON NEW RESUME --------------------
if uploaded_file is not None:
    # If a new file is uploaded, reset the UI and state
    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.resume_text = extract_text_from_pdf(uploaded_file)
        st.session_state.resume_id = hash(st.session_state.resume_text) % (10**8)

        # Clear previous results and sliders
        st.session_state.job_results = []
        st.session_state.results_ready = False
        st.session_state.enhanced_results = None
        for key in list(st.session_state.keys()):
            if key.startswith("slider_") or key.startswith("rate_btn_"):
                del st.session_state[key]
        st.toast("🧹 Cleared previous ratings and job results — ready for new recommendations!")

# -------------------- ACTION BUTTONS --------------------
st.divider()
cols = st.columns([1, 1, 2])
with cols[0]:
    if st.button("🚀 Recommend Jobs", disabled=not bool(st.session_state.resume_text)):
        run_recommend()
with cols[1]:
    enhance_disabled = not bool(st.session_state.resume_text)
    if st.button("✨ Enhance Model with Feedback", disabled=enhance_disabled):
        with st.spinner("🧠 Retraining model using your feedback..."):
            st.session_state.enhanced_results = recommender.retrain_with_feedback(
                st.session_state.resume_text, top_n=20
            )

# -------------------- SHOW JOB RESULTS --------------------
if st.session_state.results_ready and st.session_state.job_results:
    st.divider()
    st.markdown("### 🔎 Recommended Jobs for You")

    for i, job in enumerate(st.session_state.job_results[:20], start=1):
        job_id = job.get("Job Id", f"unknown_{i}")
        with st.container():
            st.markdown("<div class='job-card'>", unsafe_allow_html=True)

            st.markdown(f"#### {job['position'].title()} — {job['workplace'].title()}")
            st.write(f"**Mode:** {job['working_mode'].capitalize()}")

            # Progress indicators for explainability
            base_sim = float(job.get("similarity", 0))
            adj_sim = float(job.get("adjusted_score", base_sim))
            st.write(f"**Base Similarity:** {base_sim:.3f}")
            st.progress(min(max(base_sim, 0.0), 1.0))
            st.write(f"**Adjusted (Feedback + Skills):** {adj_sim:.3f}")
            st.progress(min(max(adj_sim, 0.0), 1.0))

            # Skills / duties
            st.write(f"**Duties:** {job['job_role_and_duties'][:250]}...")
            st.write(f"**Skills Required:** {job['requisite_skill']}")
            if job.get("matched_skills"):
                chips = "".join([f"<span class='tag'>{s.strip()}</span>" for s in str(job['matched_skills']).split(",") if s.strip()][:8])
                if chips:
                    st.markdown(f"**Matched Skills:** {chips}", unsafe_allow_html=True)

            # Rating control
            rating = st.slider(f"⭐ Rate Job {i} (1–5)", 1, 5, 3, key=f"slider_{i}")
            if st.button("💾 Submit Rating", key=f"rate_btn_{i}"):
                save_rating(st.session_state.resume_id, job_id, rating)
                st.toast(f"⭐ You rated Job {i} as {rating}/5")

            st.markdown("</div>", unsafe_allow_html=True)

# -------------------- ENHANCED MODEL (OLD vs NEW) --------------------
if st.session_state.enhanced_results:
    st.divider()
    er = st.session_state.enhanced_results

    st.markdown("### 📊 Old vs Enhanced Recommendations (Top 10)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Old (No Feedback)**")
        for i, job in enumerate(er["old_jobs"][:10], start=1):
            st.write(f"{i}. {job['position'].title()} — {job['workplace'].title()} "
                     f"(sim={job.get('similarity', 0):.3f})")
    with c2:
        st.markdown("**Enhanced (Feedback-Aware)**")
        for i, job in enumerate(er["new_jobs"][:10], start=1):
            st.write(f"{i}. {job['position'].title()} — {job['workplace'].title()} "
                     f"(adj={job.get('adjusted_score', 0):.3f})")

    # Metrics
    st.markdown("### 🧮 Metrics")
    m = er["metrics"]
    st.markdown(
        f"<span class='metric-pill'>NDCG@20: <b>{m['ndcg_at_k']}</b></span>"
        f"<span class='metric-pill'>Spearman-R: <b>{m['spearman_r']}</b></span>"
        f"<span class='metric-pill'>Reordered: <b>{m['reordered_pct']}%</b></span>",
        unsafe_allow_html=True
    )

    # Show comparison table (head)
    st.markdown("#### Detailed comparison (top 20 rows)")
    comp_df = er["comparison"].copy()
    st.dataframe(comp_df.head(20))

    # Metrics history chart (if available)
    st.markdown("#### 📈 Metrics trend")
    history = recommender.get_metrics_history()
    if not history.empty:
        history_display = history.copy()
        history_display["timestamp"] = pd.to_datetime(history_display["timestamp"])
        history_display = history_display.sort_values("timestamp")
        st.line_chart(history_display.set_index("timestamp")[["ndcg_at_k", "spearman_r"]])
    else:
        st.caption("Run enhancement a few times to accumulate metrics history.")

# -------------------- DEBUG PANEL --------------------
with st.sidebar.expander("🐞 Debug Info"):
    st.write("Ratings file path:", RATINGS_PATH)
    if os.path.exists(RATINGS_PATH):
        st.dataframe(pd.read_csv(RATINGS_PATH).tail(5))
    st.write("Session keys:", list(st.session_state.keys()))
