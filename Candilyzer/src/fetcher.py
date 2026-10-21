from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st

from src.config import MAX_WORKERS



def _github_api(username: str, token: str) -> dict | None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    profile_res = requests.get(
        f"https://api.github.com/users/{username}",
        headers=headers,
        timeout=10
    )
    if profile_res.status_code != 200:
        return None

    profile = profile_res.json()

    repos_res = requests.get(
        f"https://api.github.com/users/{username}/repos?sort=pushed&per_page=12",
        headers=headers,
        timeout=10
    )
    repos = repos_res.json() if repos_res.status_code == 200 else []

    repo_summaries = [
        f"- {r.get('name', 'Unnamed repository')} ({r.get('language') or 'N/A'}): "
        f"{r.get('description') or 'No description'}; "
        f"stars={r.get('stargazers_count', 0)}, forks={r.get('forks_count', 0)}, "
        f"archived={r.get('archived', False)}, pushed_at={r.get('pushed_at') or 'unknown'}"
        for r in repos if isinstance(r, dict)
    ]

    return {
        "name": profile.get("name", username),
        "bio": profile.get("bio", ""),
        "company": profile.get("company", ""),
        "location": profile.get("location", ""),
        "public_repos": profile.get("public_repos", 0),
        "followers": profile.get("followers", 0),
        "repos": "\n".join(repo_summaries)
    }


def _format_github(data: dict) -> str:
    return f"""
Name: {data['name']}
Bio: {data['bio']}
Company: {data['company']}
Location: {data['location']}
Public Repos: {data['public_repos']} | Followers: {data['followers']}

Recent Repositories:
{data['repos']}
""".strip()


@st.cache_data(ttl=3600)
def cached_github(username: str, token: str) -> str | None:
    if not username:
        return None
    try:
        data = _github_api(username, token)
        if not data:
            return None
        return _format_github(data)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def cached_linkedin(query: str, key: str) -> str | None:
    if not query or not key:
        return None
    try:
        from agno.tools.exa import ExaTools
        tool = ExaTools(api_key=key)
        return str(tool.search(query))
    except Exception:
        return None


def fetch_candidate(c: dict, github_key: str, exa_key: str) -> dict:
    username = c.get("username") or "Unknown"

    try:
        github_data = cached_github(c.get("github_username") or "", github_key)
        linkedin_data = cached_linkedin(c.get("linkedin", ""), exa_key)
        resume = c.get("resume")
        source_status = {
            "github": bool(github_data),
            "linkedin": bool(linkedin_data),
            "resume": bool(resume),
        }

        return {
            "candidate_id": c.get("candidate_id"),
            "username": username,
            "github": github_data,
            "linkedin": linkedin_data,
            "resume": resume,
            "source_status": source_status,
            "error": None if any(source_status.values()) else "No candidate evidence could be retrieved.",
        }
    except Exception as exc:
        return {
            "candidate_id": c.get("candidate_id"),
            "username": username,
            "github": None,
            "linkedin": None,
            "resume": None,
            "source_status": {"github": False, "linkedin": False, "resume": False},
            "error": str(exc),
        }


def fetch_all(candidates: list, github_key: str, exa_key: str) -> list:
    with ThreadPoolExecutor(MAX_WORKERS) as executor:
        futures = [
            executor.submit(fetch_candidate, c, github_key, exa_key)
            for c in candidates
        ]
        return [f.result() for f in futures]
