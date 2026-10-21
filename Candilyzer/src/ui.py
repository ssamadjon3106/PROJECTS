import html
import json
from typing import Iterable

import streamlit as st

from src.question_generator import build_question_generator, generate_questions
from src.ranking import build_df, rank, score_evaluation


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
        :root {
          --ground: #f4f8f8; --surface: #ffffff; --raised: #f4f6f8;
          --active: #e8eeee; --border: #e3e9ea; --rule: #edf1f1;
          --control: #bdc8ca; --text: #131416; --muted: #565f61;
          --accent: #176c78; --positive: #1e6f4a; --negative: #b4232a;
        }
        html, body, [class*="css"] { font-family: "IBM Plex Sans", system-ui, sans-serif; }
        .stApp { background: var(--ground); color: var(--text); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background: var(--ground); border-right: 0; }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
        [data-testid="stMainBlockContainer"] {
          max-width: 1280px; background: var(--surface); border: 1px solid var(--border);
          border-radius: 16px; margin: .5rem .5rem .5rem 0; padding: 2rem 2.25rem 3rem;
        }
        h1 { font-size: 2.15rem !important; line-height: 1.12 !important; letter-spacing: -.035em; }
        h2 { font-size: 1.2rem !important; line-height: 1.3 !important; }
        h3, h4 { font-size: 1rem !important; line-height: 1.35 !important; }
        p, li, label, [data-testid="stMarkdownContainer"] { color: var(--text); }
        .brand { display:flex; align-items:center; gap:.7rem; margin:.1rem 0 1.7rem; }
        .brand-mark { width:27px; height:27px; border:6px solid var(--accent); border-right-color:transparent; border-radius:50%; transform:rotate(-18deg); box-sizing:border-box; }
        .brand-copy strong { display:block; font-size:1.02rem; line-height:1.05; }
        .brand-copy span { color:var(--muted); font-size:.77rem; }
        .title-row { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin-bottom:1.4rem; }
        .title-row h1 { margin:0 0 .35rem; }
        .title-row p { color:var(--muted); margin:0; max-width:720px; }
        .pipeline { color:var(--muted); font-size:.83rem; line-height:1.75; }
        .pipeline strong { color:var(--text); font-weight:500; }
        .service-row { display:flex; align-items:center; justify-content:space-between; font-size:.82rem; padding:.45rem 0; border-bottom:1px solid var(--rule); }
        .service-state { display:inline-flex; align-items:center; gap:.35rem; color:var(--muted); }
        .service-dot { width:7px; height:7px; border-radius:50%; background:#8a9496; }
        .service-dot.ready { background:var(--positive); }
        div[data-testid="stForm"] { border:0; padding:0; }
        textarea, input { border-color:var(--control) !important; border-radius:6px !important; }
        textarea:focus, input:focus { border-color:var(--accent) !important; box-shadow:0 0 0 1px var(--accent) !important; }
        .stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
          min-height:2.35rem; border-radius:6px; border:1px solid var(--control); font-weight:500;
        }
        [data-testid="stBaseButton-primaryFormSubmit"] {
          background:#182022 !important; border-color:#182022 !important; color:white !important;
        }
        [data-testid="stBaseButton-primaryFormSubmit"] p {
          color:white !important;
        }
        .metric-strip { display:grid; grid-template-columns:repeat(4,1fr); border:1px solid var(--border); border-radius:8px; margin:1rem 0 1.5rem; overflow:hidden; }
        .metric-cell { padding:1rem 1.1rem; border-right:1px solid var(--border); }
        .metric-cell:last-child { border-right:0; }
        .metric-label { color:var(--muted); font-size:.77rem; margin-bottom:.35rem; }
        .metric-value { font-size:1.65rem; font-weight:400; font-variant-numeric:tabular-nums; letter-spacing:-.03em; }
        .finding { background:var(--raised); border-radius:8px; padding:1rem 1.1rem; margin:.25rem 0 1.5rem; }
        .finding strong { font-weight:600; }
        .ranking { border:1px solid var(--border); border-radius:8px; overflow:hidden; margin:.75rem 0 1.5rem; }
        .rank-row { display:grid; grid-template-columns:36px minmax(150px,1.4fr) minmax(180px,1fr) 92px 92px; align-items:center; gap:1rem; padding:.8rem 1rem; border-bottom:1px solid var(--rule); }
        .rank-row:last-child { border-bottom:0; }
        .rank-number { color:var(--muted); font-variant-numeric:tabular-nums; }
        .rank-name { font-weight:500; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .rank-score { display:grid; grid-template-columns:1fr 42px; align-items:center; gap:.65rem; }
        .rank-meta { color:var(--muted); font-size:.78rem; }
        .coverage { border:1px solid var(--border); border-radius:8px; overflow:hidden; margin:.75rem 0 1.5rem; }
        .coverage-row { display:grid; grid-template-columns:minmax(150px,1.2fr) 3fr 70px; align-items:center; gap:1rem; padding:.85rem 1rem; border-bottom:1px solid var(--rule); }
        .coverage-row:last-child { border-bottom:0; }
        .coverage-name { font-weight:500; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .coverage-track { height:8px; border-radius:99px; background:#e6ecec; overflow:hidden; }
        .coverage-fill { display:block; height:100%; background:var(--accent); border-radius:99px; }
        .coverage-value { text-align:right; font-variant-numeric:tabular-nums; }
        .decision { display:inline-flex; align-items:center; min-height:24px; padding:0 .55rem; border-radius:99px; font-size:.75rem; font-weight:500; }
        .decision.positive { background:#dcfae5; color:#12452b; }
        .decision.neutral { background:#e8eeee; color:#303839; }
        .decision.warning { background:#fff4d6; color:#71410b; }
        .decision.negative { background:#ffedea; color:#831c1e; }
        .source-list { display:flex; flex-wrap:wrap; gap:.4rem; margin:.65rem 0 1.1rem; }
        .source { border:1px solid var(--border); border-radius:99px; padding:.22rem .55rem; color:var(--muted); font-size:.74rem; }
        .source.available { color:#174d36; border-color:#a7d7bd; background:#f0faf4; }
        .requirement { border-top:1px solid var(--rule); padding:.9rem 0; }
        .requirement:first-child { border-top:0; }
        .requirement-head { display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; }
        .requirement-title { font-weight:500; }
        .requirement-meta { display:flex; gap:.45rem; flex:none; }
        .tag { border-radius:4px; padding:.18rem .42rem; font-size:.72rem; background:#e8eeee; color:#303839; white-space:nowrap; }
        .tag.met { background:#dcfae5; color:#12452b; }
        .tag.partial { background:#fff4d6; color:#71410b; }
        .tag.not-met { background:#ffedea; color:#831c1e; }
        .evidence { color:var(--muted); margin:.35rem 0 0; font-size:.84rem; }
        .empty-frame { border:1px dashed var(--control); border-radius:8px; padding:1.25rem; margin-top:1.25rem; }
        .empty-frame strong { display:block; margin-bottom:.35rem; }
        [data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:8px; overflow:hidden; }
        [data-baseweb="tab-list"] { gap:1.2rem; border-bottom:1px solid var(--border); }
        [data-baseweb="tab"] { background:transparent; padding-left:0; padding-right:0; }
        [data-baseweb="tab-highlight"] { background-color:var(--accent); }
        [data-testid="stAlert"] { border-radius:8px; }
        @media (max-width: 900px) {
          [data-testid="stMainBlockContainer"] { margin:0; border-radius:0; padding:1.25rem; }
          .metric-strip { grid-template-columns:repeat(2,1fr); }
          .metric-cell:nth-child(2) { border-right:0; }
          .metric-cell:nth-child(-n+2) { border-bottom:1px solid var(--border); }
          .rank-row { grid-template-columns:30px minmax(120px,1fr) minmax(140px,1fr) 72px; }
          .rank-row .rank-confidence { display:none; }
        }
        @media (max-width: 520px) {
          .title-row { display:block; }
          h1 { font-size:1.8rem !important; }
          .coverage-row { grid-template-columns:1fr 56px; }
          .coverage-track { grid-column:1 / -1; grid-row:2; }
          .metric-value { font-size:1.4rem; }
          .rank-row { grid-template-columns:28px 1fr 62px; gap:.65rem; padding:.85rem; }
          .rank-row .rank-score { grid-column:2 / -1; grid-row:2; }
          .rank-row .rank-fit { grid-column:3; grid-row:1; text-align:right; }
          .rank-row .rank-confidence { display:none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand() -> None:
    st.markdown(
        """
        <div class="brand">
          <span class="brand-mark" aria-hidden="true"></span>
          <span class="brand-copy"><strong>Candilyzer</strong><span>Evidence-first hiring intelligence</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(gemini_ready: bool, github_ready: bool, exa_ready: bool) -> None:
    with st.sidebar:
        render_brand()
        st.markdown("#### Evaluation pipeline")
        st.markdown(
            '<p class="pipeline"><strong>Collect evidence</strong> from approved sources<br>'
            '<strong>Compress facts</strong> without adding claims<br>'
            '<strong>Evaluate requirements</strong> with structured output<br>'
            '<strong>Rank candidates</strong> from auditable coverage</p>',
            unsafe_allow_html=True,
        )
        st.markdown("#### Services")
        services = [
            ("Gemini evaluation", gemini_ready),
            ("GitHub expanded quota", github_ready),
            ("LinkedIn search", exa_ready),
        ]
        for label, ready in services:
            state = "Ready" if ready else "Optional" if label != "Gemini evaluation" else "Missing"
            dot_class = "ready" if ready else ""
            st.markdown(
                f'<div class="service-row"><span>{html.escape(label)}</span><span class="service-state">'
                f'<span class="service-dot {dot_class}"></span>{state}</span></div>',
                unsafe_allow_html=True,
            )
        st.caption(
            "Candilyzer supports hiring decisions; it does not make them. Review source evidence before acting."
        )


def render_title() -> None:
    st.markdown(
        """
        <div class="title-row">
          <div><h1>Candidate evaluation</h1>
          <p>Compare technical candidates against one role using traceable evidence, calibrated coverage, and interview-ready gaps.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_strip(items: Iterable[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="metric-cell"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="metric-strip">{cells}</div>', unsafe_allow_html=True)


def _decision_class(decision: str) -> str:
    return {
        "Strong Hire": "positive",
        "Hire": "positive",
        "Interview Recommended": "warning",
        "Reject": "negative",
    }.get(decision, "neutral")


def _render_sources(status: dict) -> None:
    chips = []
    for key, label in (("github", "GitHub"), ("linkedin", "LinkedIn"), ("resume", "Resume")):
        available = bool(status.get(key))
        state_class = "available" if available else ""
        state_label = "available" if available else "not available"
        chips.append(
            f'<span class="source {state_class}">{label}: {state_label}</span>'
        )
    st.markdown(f'<div class="source-list">{"".join(chips)}</div>', unsafe_allow_html=True)


def _render_requirements(requirements: list[dict]) -> None:
    rows = []
    for requirement in requirements:
        match = requirement.get("match", "Not Met")
        match_class = match.lower().replace(" ", "-")
        rows.append(
            '<div class="requirement"><div class="requirement-head">'
            f'<span class="requirement-title">{html.escape(requirement.get("requirement", ""))}</span>'
            '<span class="requirement-meta">'
            f'<span class="tag">{html.escape(requirement.get("importance", ""))}</span>'
            f'<span class="tag {match_class}">{html.escape(match)}</span></span></div>'
            f'<p class="evidence">{html.escape(requirement.get("evidence", ""))}</p></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def _render_coverage(results: list[dict]) -> None:
    rows = []
    for result in results:
        evaluation = result.get("parsed")
        if not evaluation:
            continue
        coverage = score_evaluation(evaluation)["coverage"]
        rows.append(
            '<div class="coverage-row">'
            f'<span class="coverage-name">{html.escape(result["username"])}</span>'
            f'<span class="coverage-track"><span class="coverage-fill" style="width:{coverage:.1f}%"></span></span>'
            f'<span class="coverage-value">{coverage:.1f}%</span></div>'
        )
    st.markdown(f'<div class="coverage">{"".join(rows)}</div>', unsafe_allow_html=True)


def _render_ranking(frame) -> None:
    rows = []
    for index, item in frame.iterrows():
        score = float(item["Score"])
        rows.append(
            '<div class="rank-row">'
            f'<span class="rank-number">{index + 1}</span>'
            f'<span class="rank-name">{html.escape(str(item["Candidate"]))}</span>'
            '<span class="rank-score">'
            f'<span class="coverage-track"><span class="coverage-fill" style="width:{score:.1f}%"></span></span>'
            f'<span>{score:.1f}</span></span>'
            f'<span class="rank-meta rank-fit">Must have {html.escape(str(item["Must-have fit"]))}</span>'
            f'<span class="rank-meta rank-confidence">{html.escape(str(item["Confidence"]))} confidence</span>'
            '</div>'
        )
    st.markdown(f'<div class="ranking">{"".join(rows)}</div>', unsafe_allow_html=True)


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-frame">
          <strong>No analysis in this session</strong>
          Add a role and at least one candidate source, or open the sample analysis to review the complete workflow without API calls.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(state: dict, api_key: str) -> None:
    results = state["results"]
    successful = [result for result in results if result.get("parsed")]
    failed = [result for result in results if not result.get("parsed")]
    frame = rank(build_df(successful))

    st.markdown("## Analysis results")
    if state.get("sample"):
        st.info("This is labeled sample data for product review. Run an analysis to use real candidate evidence.")

    if frame.empty:
        st.error("No candidate evaluation completed successfully.")
        for result in failed:
            st.write(f"{result['username']}: {result.get('error', 'Unknown error')}")
        return

    top = frame.iloc[0]
    total_requirements = sum(
        len(result["parsed"].get("requirements_analysis", [])) for result in successful
    )
    total_risks = sum(len(result["parsed"].get("red_flags", [])) for result in successful)
    _metric_strip(
        [
            ("Candidates evaluated", str(len(successful))),
            ("Top evidence score", f"{top['Score']:.1f}"),
            ("Requirements reviewed", str(total_requirements)),
            ("Flagged risks", str(total_risks)),
        ]
    )
    st.markdown(
        f'<div class="finding"><strong>{html.escape(str(top["Candidate"]))}</strong> leads with '
        f'{top["Evidence coverage"]:.1f}% weighted requirement coverage and {html.escape(str(top["Must-have fit"]))} '
        'must-have requirements fully evidenced.</div>',
        unsafe_allow_html=True,
    )

    actions, _ = st.columns([1, 4])
    with actions:
        payload = json.dumps(
            {"job": state["job"], "results": successful}, indent=2, ensure_ascii=False
        )
        st.download_button(
            "Download evaluation JSON",
            data=payload,
            file_name="candilyzer-evaluation.json",
            mime="application/json",
            width="stretch",
        )

    st.markdown("### Ranked fit")
    _render_ranking(frame)

    st.markdown("### Requirement coverage")
    st.caption("Weighted by must-have importance before confidence and risk adjustments.")
    ordered_results = sorted(
        successful,
        key=lambda result: score_evaluation(result["parsed"])["score"],
        reverse=True,
    )
    _render_coverage(ordered_results)

    st.markdown("### Candidate evidence")
    tabs = st.tabs([result["username"] for result in ordered_results])
    summaries = state.get("summaries", {})
    for tab, result in zip(tabs, ordered_results):
        with tab:
            evaluation = result["parsed"]
            breakdown = score_evaluation(evaluation)
            decision = evaluation.get("decision", "Unknown")
            st.markdown(
                f'<span class="decision {_decision_class(decision)}">{html.escape(decision)}</span>',
                unsafe_allow_html=True,
            )
            _render_sources(result.get("source_status", {}))
            _metric_strip(
                [
                    ("Evidence score", f"{breakdown['score']:.1f}"),
                    ("Coverage", f"{breakdown['coverage']:.1f}%"),
                    ("Must-have met", f"{breakdown['must_have_met']}/{breakdown['must_have_total']}"),
                    ("Confidence", evaluation.get("confidence", "Low")),
                ]
            )
            st.write(evaluation.get("overall_summary", ""))
            st.caption(evaluation.get("confidence_reason", ""))

            st.markdown("#### Requirement evidence")
            _render_requirements(evaluation.get("requirements_analysis", []))

            strengths_col, gaps_col = st.columns(2, gap="large")
            with strengths_col:
                st.markdown("#### Strengths")
                strengths = evaluation.get("strengths") or ["No evidenced strengths returned."]
                for item in strengths:
                    st.markdown(f"- {item}")
            with gaps_col:
                st.markdown("#### Gaps and risks")
                items = (evaluation.get("gaps") or []) + (evaluation.get("red_flags") or [])
                for item in items or ["No specific gaps or risks returned."]:
                    st.markdown(f"- {item}")

            if state.get("sample"):
                st.caption("Interview question generation is disabled for sample data.")
            elif api_key:
                key = f"questions-{result['username']}"
                if st.button("Generate interview questions", key=f"generate-{result['username']}"):
                    with st.spinner("Generating evidence-focused questions..."):
                        st.session_state[key] = generate_questions(
                            build_question_generator(api_key),
                            state["job"],
                            summaries.get(result["username"], "No summary available"),
                            evaluation,
                        )
                if key in st.session_state:
                    st.markdown(st.session_state[key])

    if failed:
        with st.expander(f"Incomplete evaluations ({len(failed)})"):
            for result in failed:
                st.write(f"{result['username']}: {result.get('error', 'Unknown error')}")
