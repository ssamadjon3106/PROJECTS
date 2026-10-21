from pathlib import Path
from typing import BinaryIO

import docx
import pdfplumber

from src.config import (
    MAX_RESUME_BYTES,
    MAX_RESUME_CHARS,
    MAX_RESUME_PAGES,
)


class ResumeParseError(ValueError):
    pass


def _file_size(file: BinaryIO) -> int:
    size = getattr(file, "size", None)
    if isinstance(size, int):
        return size
    current = file.tell()
    file.seek(0, 2)
    size = file.tell()
    file.seek(current)
    return size


def parse_resume(file: BinaryIO) -> str:
    if not file:
        raise ResumeParseError("No resume file was provided.")

    name = getattr(file, "name", "resume")
    suffix = Path(name).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise ResumeParseError(f"Unsupported resume format: {suffix or 'unknown'}")
    if _file_size(file) > MAX_RESUME_BYTES:
        raise ResumeParseError(
            f"{name} exceeds the {MAX_RESUME_BYTES // (1024 * 1024)} MB limit."
        )

    try:
        file.seek(0)
        if suffix == ".pdf":
            with pdfplumber.open(file) as pdf:
                if len(pdf.pages) > MAX_RESUME_PAGES:
                    raise ResumeParseError(
                        f"{name} exceeds the {MAX_RESUME_PAGES}-page limit."
                    )
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        else:
            document = docx.Document(file)
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)

        normalized = text.strip()
        if not normalized:
            raise ResumeParseError(f"No readable text was found in {name}.")
        return normalized[:MAX_RESUME_CHARS]

    except ResumeParseError:
        raise
    except Exception as exc:
        raise ResumeParseError(f"Could not parse {name}: {exc}") from exc
