import re
from pathlib import Path
from urllib.parse import urlparse

from src.config import MAX_CANDIDATES


GITHUB_USERNAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")


class CandidateInputError(ValueError):
    pass


def split_values(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[\n,]+", value) if part.strip()]


def _validate_github(username: str) -> None:
    if not GITHUB_USERNAME.fullmatch(username):
        raise CandidateInputError(f"Invalid GitHub username: {username}")


def _validate_linkedin(url: str) -> None:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        hostname == "linkedin.com" or hostname.endswith(".linkedin.com")
    ):
        raise CandidateInputError(f"Invalid LinkedIn URL: {url}")


def _linkedin_label(url: str) -> str:
    path_parts = [part for part in urlparse(url).path.split("/") if part]
    return path_parts[-1] if path_parts else "LinkedIn candidate"


def build_candidates(
    github_text: str,
    linkedin_text: str,
    resumes: list[tuple[str, str]],
) -> list[dict]:
    githubs = split_values(github_text)
    linkedins = split_values(linkedin_text)

    for username in githubs:
        _validate_github(username)
    for url in linkedins:
        _validate_linkedin(url)

    count = max(len(githubs), len(linkedins), len(resumes), 0)
    if count == 0:
        raise CandidateInputError(
            "Add at least one GitHub username, LinkedIn URL, or resume."
        )
    if count > MAX_CANDIDATES:
        raise CandidateInputError(
            f"Analyze at most {MAX_CANDIDATES} candidates per run."
        )

    candidates: list[dict] = []
    for index in range(count):
        github = githubs[index] if index < len(githubs) else ""
        linkedin = linkedins[index] if index < len(linkedins) else ""
        resume_name, resume_text = resumes[index] if index < len(resumes) else ("", "")
        label = github or (_linkedin_label(linkedin) if linkedin else Path(resume_name).stem)
        candidates.append(
            {
                "candidate_id": f"candidate-{index + 1}",
                "username": label or f"Candidate {index + 1}",
                "github_username": github or None,
                "linkedin": linkedin or None,
                "resume": resume_text or None,
                "resume_name": resume_name or None,
            }
        )
    return candidates
