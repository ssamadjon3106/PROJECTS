from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any

import requests
import streamlit as st

from src.config import MAX_WORKERS

logger = logging.getLogger(__name__)

GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "Candilyzer/1.0",
}
REQUEST_TIMEOUT = (3.05, 10)
SOURCE_LABELS = {"github": "GitHub", "linkedin": "LinkedIn"}


class SourceFetchError(RuntimeError):
    """A safe, user-facing source collection failure."""


def _github_get(url: str, token: str) -> requests.Response:
    headers = GITHUB_HEADERS | ({"Authorization": f"Bearer {token}"} if token else {})
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    if token and response.status_code in {401, 403}:
        return requests.get(url, headers=GITHUB_HEADERS, timeout=REQUEST_TIMEOUT)
    return response


def _github_api(username: str, token: str) -> dict | None:
    profile_res = _github_get(f"https://api.github.com/users/{username}", token)
    if profile_res.status_code == 404:
        raise SourceFetchError(f"GitHub profile @{username} was not found.")
    if profile_res.status_code == 403:
        raise SourceFetchError("GitHub rate limit reached. Try again later or configure a valid token.")
    if profile_res.status_code != 200:
        raise SourceFetchError("GitHub evidence is temporarily unavailable.")

    profile = profile_res.json()

    repos_res = _github_get(
        f"https://api.github.com/users/{username}/repos?sort=pushed&per_page=12",
        token,
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


@st.cache_data(ttl=3600, show_spinner=False)
def cached_github(username: str, token: str) -> str | None:
    if not username:
        return None
    data = _github_api(username, token)
    return _format_github(data) if data else None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_linkedin(query: str, key: str) -> str | None:
    if not query or not key:
        return None
    from agno.tools.exa import ExaTools

    result = str(ExaTools(api_key=key).search(query)).strip()
    if not result:
        raise SourceFetchError("LinkedIn search returned no public evidence.")
    return result


def _collect_source(name: str, loader: Any, errors: dict[str, str]) -> str | None:
    try:
        return loader()
    except SourceFetchError as exc:
        errors[name] = str(exc)
    except Exception:
        logger.exception("Unexpected %s source failure", name)
        errors[name] = f"{SOURCE_LABELS.get(name, name.title())} evidence is temporarily unavailable."
    return None


def fetch_candidate(c: dict, github_key: str, exa_key: str) -> dict:
    username = c.get("username") or "Unknown"
    errors: dict[str, str] = {}
    github_username = c.get("github_username") or ""
    linkedin_url = c.get("linkedin") or ""

    github_data = _collect_source(
        "github",
        lambda: cached_github(github_username, github_key),
        errors,
    ) if github_username else None
    linkedin_data = _collect_source(
        "linkedin",
        lambda: cached_linkedin(linkedin_url, exa_key),
        errors,
    ) if linkedin_url and exa_key else None
    if linkedin_url and not exa_key:
        errors["linkedin"] = "LinkedIn search is not configured for this deployment."

    resume = c.get("resume")
    source_status = {
        "github": bool(github_data),
        "linkedin": bool(linkedin_data),
        "resume": bool(resume),
    }
    error = None
    if not any(source_status.values()):
        error = " ".join(errors.values()) or "No candidate evidence could be retrieved."

    return {
        "candidate_id": c.get("candidate_id"),
        "username": username,
        "github": github_data,
        "linkedin": linkedin_data,
        "resume": resume,
        "source_status": source_status,
        "source_errors": errors,
        "error": error,
    }


def fetch_all(candidates: list[dict], github_key: str, exa_key: str) -> list[dict]:
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(candidates) or 1)) as executor:
        futures = [
            executor.submit(fetch_candidate, c, github_key, exa_key)
            for c in candidates
        ]
        return [f.result() for f in futures]
