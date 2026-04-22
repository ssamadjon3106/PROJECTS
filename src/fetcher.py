import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor
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

    # fetch repos
    repos_res = requests.get(
        f"https://api.github.com/users/{username}/repos?sort=updated&per_page=10",
        headers=headers,
        timeout=10
    )
    repos = repos_res.json() if repos_res.status_code == 200 else []

    repo_summaries = [
        f"- {r['name']} ({r.get('language', 'N/A')}): {r.get('description', 'No description')} "
        f"[⭐{r.get('stargazers_count', 0)}]"
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
            print(f"[fetcher] GitHub returned no data for: {username}")
            return None
        return _format_github(data)
    except Exception as e:
        print(f"[fetcher] GitHub fetch failed for {username}: {e}")
        return None


@st.cache_data(ttl=3600)
def cached_linkedin(query: str, key: str) -> str | None:
    if not query or not key:
        return None
    try:
        from agno.tools.exa import ExaTools
        tool = ExaTools(api_key=key)
        return tool.search(query)
    except Exception as e:
        print(f"[fetcher] LinkedIn fetch failed for {query}: {e}")
        return None

def fetch_candidate(c: dict, github_key: str, exa_key: str) -> dict:
    username = c.get("username") or c.get("linkedin") or "Unknown"

    try:
        github_data = cached_github(c.get("username", ""), github_key)
        linkedin_data = cached_linkedin(c.get("linkedin", ""), exa_key)

        return {
            "username": username,
            "github": github_data,
            "linkedin": linkedin_data,
            "resume": c.get("resume"),
            "error": None
        }
    except Exception as e:
        return {
            "username": username,
            "github": None,
            "linkedin": None,
            "resume": None,
            "error": str(e)
        }


def fetch_all(candidates: list, github_key: str, exa_key: str) -> list:
    with ThreadPoolExecutor(MAX_WORKERS) as executor:
        futures = [
            executor.submit(fetch_candidate, c, github_key, exa_key)
            for c in candidates
        ]
        return [f.result() for f in futures]