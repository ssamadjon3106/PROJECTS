import copy
import logging
import os

import streamlit as st
from dotenv import load_dotenv

from src.candidates import CandidateInputError, build_candidates
from src.compressor import compress_all
from src.demo_data import SAMPLE_JOB, SAMPLE_RESULTS, SAMPLE_SUMMARIES
from src.fetcher import fetch_all
from src.pipeline import evaluate_all
from src.resume_parser import ResumeParseError, parse_resume
from src.ui import (
    apply_theme,
    render_empty_state,
    render_results,
    render_section_intro,
    render_sidebar,
    render_title,
)


load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GITHUB_API_KEY = os.getenv("GITHUB_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")

st.set_page_config(
    page_title="Candilyzer | Evidence-first candidate evaluation",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="auto",
)
apply_theme()
render_sidebar(bool(GEMINI_API_KEY), bool(GITHUB_API_KEY), bool(EXA_API_KEY))
render_title()

intro, sample_action = st.columns([3, 1])
with intro:
    st.caption(
        "Human review required · Job-related evidence only · No automated hiring decisions"
    )
with sample_action:
    if st.button("View sample analysis", width="stretch"):
        st.session_state.analysis_state = {
            "job": SAMPLE_JOB,
            "results": copy.deepcopy(SAMPLE_RESULTS),
            "summaries": SAMPLE_SUMMARIES.copy(),
            "sample": True,
        }
        st.rerun()

form_host = (
    st.expander("Run another analysis", expanded=False)
    if "analysis_state" in st.session_state
    else st.container()
)
with form_host:
    render_section_intro(
        "Role profile",
        "Paste the responsibilities and explicit must-have requirements used for this review.",
    )
    with st.form("candidate-analysis"):
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            job = st.text_area(
                "Job description",
                height=270,
                placeholder="Paste responsibilities, must-have requirements, and preferred experience...",
            )
        with right:
            st.markdown("**Candidate evidence**")
            st.caption(
                "Sources are paired by line and upload order. Keep each candidate on the same row."
            )
            githubs = st.text_area(
                "GitHub usernames",
                height=88,
                placeholder="One candidate per line",
                help="Public profiles work without a token; a token increases the GitHub rate limit.",
            )
            linkedins = st.text_area(
                "LinkedIn profile URLs",
                height=88,
                placeholder="One HTTPS profile URL per line",
                help="LinkedIn discovery requires EXA_API_KEY.",
            )
            files = st.file_uploader(
                "Resumes",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                help="PDF or DOCX, up to 10 MB and 40 PDF pages per file.",
            )
        st.markdown(
            '<div class="review-note">Only submit candidate data you are authorized to process. '
            "Verify every result against the original sources before taking action.</div>",
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button(
            "Analyze submitted evidence", type="primary", width="stretch"
        )

if submitted:
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is required for live analysis. Add it to the app environment.")
    elif not job.strip():
        st.error("Add a job description before running the evaluation.")
    else:
        try:
            resumes = [(file.name, parse_resume(file)) for file in files]
            candidates = build_candidates(githubs, linkedins, resumes)

            if linkedins.strip() and not EXA_API_KEY:
                st.warning(
                    "EXA_API_KEY is not configured, so LinkedIn sources will be marked unavailable."
                )

            with st.status("Analyzing submitted evidence", expanded=True) as status:
                st.write("Collecting candidate evidence...")
                raw = fetch_all(candidates, GITHUB_API_KEY, EXA_API_KEY)
                st.write("Compressing source facts with injection-resistant instructions...")
                compressed = compress_all(raw, GEMINI_API_KEY)
                st.write("Evaluating role requirements with isolated model sessions...")
                results = evaluate_all(compressed, GEMINI_API_KEY, job.strip())
                status.update(label="Analysis complete", state="complete", expanded=False)

            st.session_state.analysis_state = {
                "job": job.strip(),
                "results": results,
                "summaries": {
                    candidate["username"]: candidate.get("summary") or ""
                    for candidate in compressed
                },
                "sample": False,
            }
            st.rerun()
        except (CandidateInputError, ResumeParseError) as exc:
            st.error(str(exc))
        except Exception:
            logger.exception("Unexpected analysis failure")
            st.error(
                "The analysis could not be completed. No results were saved; try again or check the service status."
            )

if "analysis_state" in st.session_state:
    render_results(st.session_state.analysis_state, GEMINI_API_KEY)
else:
    render_empty_state()
