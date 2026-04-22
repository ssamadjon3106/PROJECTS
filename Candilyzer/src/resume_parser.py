import pdfplumber
import docx


def parse_resume(file):
    if not file:
        return None

    try:
        if file.name.endswith(".pdf"):
            text = ""
            with pdfplumber.open(file) as pdf:
                for p in pdf.pages:
                    text += p.extract_text() or ""
            return text.strip() or None

        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
            return text.strip() or None

    except Exception as e:
        print(f"[resume_parser] Failed to parse {file.name}: {e}")
        return None

    return None