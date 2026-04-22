import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os

load_dotenv()


from src.fetcher import fetch_all
from src.resume_parser import parse_resume
from src.compressor import build_compressor, compress_all
from src.evaluator import evaluate_one
from src.ranking import build_df, rank, top_k
from src.evaluator_factory import create_evaluator
from src.question_generator import build_question_generator, generate_questions


NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
GITHUB_API_KEY = os.getenv("GITHUB_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")


st.set_page_config(page_title="Candilyzer", page_icon="🚀", layout="wide")
st.title("🚀 Candilyzer — AI Candidate Analyzer")




job = st.text_area("📋 Job Description", height=200, placeholder="Paste the full job description here...")

col1, col2 = st.columns(2)
with col1:
    githubs = st.text_input("🐙 GitHub usernames (comma separated)", placeholder="e.g. ssamadjon3106,..")
with col2:
    linkedins = st.text_input("💼 LinkedIn URLs (comma separated)", placeholder="e.g. https://www.linkedin.com/in/samadjon-sayfullayev/")

files = st.file_uploader(
    "📄 Upload Resumes (PDF or DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)



def render_badge(decision: str):
    if decision in ["Strong Hire", "Hire"]:
        st.success(f"🎯 {decision}")
    elif decision == "Interview Recommended":
        st.info(f"🟡 {decision}")
    else:
        st.error(f"❌ {decision}")


def display_candidate(r: dict, summary: str, job: str, qg):
    st.markdown(f"## 👤 {r['username']}")

    if not r.get("parsed"):
        st.warning("⚠️ No valid evaluation result.")
        with st.expander("Raw output"):
            st.code(r.get("raw", "No output"))
        st.divider()
        return

    p = r["parsed"]

    
    reqs = p.get("requirements_analysis", [])
    if reqs:
        st.markdown("### 📊 Requirement Match")
        for req in reqs:
            if isinstance(req, dict):
                st.markdown(f"- **{req.get('requirement', '?')}** → {req.get('match', '?')}")

    
    strengths = p.get("strengths", [])
    if strengths:
        st.markdown("### ✅ Strengths")
        for s in strengths:
            st.markdown(f"- {s}")

    
    gaps = p.get("gaps", [])
    if gaps:
        st.markdown("### ⚠️ Gaps")
        for g in gaps:
            st.markdown(f"- {g}")

    
    red_flags = p.get("red_flags", [])
    if red_flags:
        st.markdown("### 🚩 Red Flags")
        for rf in red_flags:
            st.markdown(f"- {rf}")

    
    st.markdown("### 🎯 Decision")
    decision = p.get("decision", "Unknown")
    render_badge(decision)
    st.markdown(f"**Confidence:** {p.get('confidence', 'Low')} — {p.get('confidence_reason', '')}")

    
    if p.get("overall_summary"):
        st.markdown(f"> {p['overall_summary']}")

    
    if decision in ["Strong Hire", "Hire", "Interview Recommended"]:
        with st.expander("🧠 Interview Questions"):

            key_state = f"questions_{r['username']}" 

            if st.button(f"Generate Questions for {r['username']}", key=f"q_{r['username']}"):
                with st.spinner("Generating questions..."):
                    questions = generate_questions(
                        qg,
                        job,
                        summary or "No summary available",
                        p
                    )
                   
                    st.session_state[key_state] = questions

            
            if key_state in st.session_state:
                st.markdown(st.session_state[key_state])

    st.divider()


def get_top_candidates(results: list, top_n: int = 3):
    df = build_df(results)

    if df.empty:
        return [], results

    df = rank(df)
    top_names = df.head(top_n)["Candidate"].tolist()

    top = [r for r in results if r["username"] in top_names]
    rest = [r for r in results if r["username"] not in top_names]

    return top, rest



if st.button("🔍 Analyze Candidates", type="primary"):

    
    if not NEBIUS_API_KEY:
        st.error("❌ Missing NEBIUS_API_KEY in .env file.")
        st.stop()

    if not job.strip():
        st.warning("⚠️ Please enter a job description.")
        st.stop()

    gh = [g.strip() for g in githubs.split(",") if g.strip()]
    li = [l.strip() for l in linkedins.split(",") if l.strip()]
    resumes = [parse_resume(f) for f in files] if files else []

    if not gh and not li and not resumes:
        st.warning("⚠️ Please provide at least one GitHub username, LinkedIn URL, or resume.")
        st.stop()

    
    max_len = max(len(gh), len(li), len(resumes), 1)
    candidates = []
    for i in range(max_len):
        candidates.append({
            "username": gh[i] if i < len(gh) else (f"candidate_{i+1}"),
            "linkedin": li[i] if i < len(li) else None,
            "resume": resumes[i] if i < len(resumes) else None
        })

    
    with st.spinner("🔍 Fetching candidate data..."):
        raw = fetch_all(candidates, GITHUB_API_KEY, EXA_API_KEY)

    with st.spinner("🧠 Summarizing candidate profiles..."):
        compressor = build_compressor(NEBIUS_API_KEY)
        compressed = compress_all(raw, compressor)

    summary_map = {
        c["username"]: c.get("summary")
        for c in compressed
    }

    
    failed = [c["username"] for c in compressed if not c.get("summary")]
    if failed:
        st.warning(f"⚠️ Could not summarize: {', '.join(failed)}")

    
    evaluator = create_evaluator(NEBIUS_API_KEY, compressed)
    qg = build_question_generator(NEBIUS_API_KEY)
    results = []

    st.divider()
    st.subheader("👤 Candidate Evaluations")

    with st.spinner("⚙️ Evaluating candidates..."):
        with ThreadPoolExecutor(5) as executor:
            future_map = {
                executor.submit(evaluate_one, c, evaluator, job): c
                for c in compressed
            }
            for f in as_completed(future_map):
                r = f.result()
                results.append(r)

    
    if not results:
        st.error("No results to display.")
        st.stop()

    top_results, other_results = get_top_candidates(results)

    if top_results:
        st.subheader("⭐ Top Candidates")
        for r in top_results:
            display_candidate(r, summary_map.get(r["username"]), job, qg)

    if other_results:
        st.subheader("📄 Other Candidates")
        for r in other_results:
            display_candidate(r, summary_map.get(r["username"]), job, qg)


    
    if len(results) > 1:
        df = build_df(results)

        if not df.empty:
            df = rank(df)

            st.subheader("🏆 Full Ranking")
            st.dataframe(df, use_container_width=True)

            st.subheader("⭐ Top 3")
            st.dataframe(top_k(df), use_container_width=True)
    else:
        st.info("Single candidate mode — ranking not applicable.")