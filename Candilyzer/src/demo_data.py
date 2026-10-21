SAMPLE_JOB = """Senior AI Product Engineer

Build and operate production LLM applications. The role requires strong Python and FastAPI,
hands-on RAG and evaluation experience, reliable API design, Docker-based deployment, and clear
communication with product teams. Experience with observability and human-in-the-loop workflows
is preferred."""


def _requirements(matches: list[str], evidence: list[str]) -> list[dict]:
    requirements = [
        ("Production Python and FastAPI experience", "Must Have"),
        ("RAG system design and evaluation", "Must Have"),
        ("Reliable API and failure-handling practices", "Must Have"),
        ("Docker-based deployment", "Must Have"),
        ("Observability and human-in-the-loop workflows", "Nice to Have"),
    ]
    return [
        {
            "requirement": requirement,
            "importance": importance,
            "match": match,
            "evidence": proof,
        }
        for (requirement, importance), match, proof in zip(requirements, matches, evidence)
    ]


SAMPLE_RESULTS = [
    {
        "username": "Amina Karimova",
        "parsed": {
            "requirements_analysis": _requirements(
                ["Met", "Met", "Met", "Met", "Partial"],
                [
                    "Resume: built typed FastAPI services in Python for two shipped AI products.",
                    "GitHub and resume: implemented document chunking, retrieval tests, and citation checks.",
                    "GitHub: bounded retries, request validation, and structured provider errors are present.",
                    "Resume: deployed containerized services through a documented CI pipeline.",
                    "Resume: added model-call telemetry; approval workflow ownership is not fully demonstrated.",
                ],
            ),
            "strengths": [
                "Evidence spans implementation, testing, and deployment rather than framework mentions.",
                "RAG work includes retrieval evaluation and citation validation.",
            ],
            "gaps": ["Needs deeper evidence of designing human review operations at scale."],
            "red_flags": [],
            "overall_summary": "Strong evidence across the role's four must-have areas, with the clearest depth in RAG evaluation and production Python services.",
            "decision": "Strong Hire",
            "confidence": "High",
            "confidence_reason": "The resume and public repositories provide consistent, implementation-level evidence.",
        },
        "source_status": {"github": True, "linkedin": True, "resume": True},
        "raw": "",
        "error": None,
    },
    {
        "username": "Daniel Ortiz",
        "parsed": {
            "requirements_analysis": _requirements(
                ["Met", "Partial", "Met", "Met", "Not Met"],
                [
                    "Resume and GitHub: maintains Python APIs with FastAPI and typed request models.",
                    "Resume: integrated a vector database, but no retrieval evaluation evidence is provided.",
                    "GitHub: includes timeouts, retry policies, and API contract tests.",
                    "GitHub: repository includes Dockerfiles and a CI build workflow.",
                    "No evidence of model observability or a human approval workflow was found.",
                ],
            ),
            "strengths": ["Strong backend reliability and API engineering evidence."],
            "gaps": ["RAG experience appears integration-focused rather than evaluation-focused."],
            "red_flags": [],
            "overall_summary": "A credible backend engineer with useful LLM integration experience, but the RAG evaluation and operating-model evidence is incomplete.",
            "decision": "Interview Recommended",
            "confidence": "High",
            "confidence_reason": "The available sources agree, but they do not cover every requirement.",
        },
        "source_status": {"github": True, "linkedin": False, "resume": True},
        "raw": "",
        "error": None,
    },
    {
        "username": "Noah Williams",
        "parsed": {
            "requirements_analysis": _requirements(
                ["Partial", "Not Met", "Partial", "Not Met", "Partial"],
                [
                    "Resume: lists Python services; FastAPI ownership and production scope are unclear.",
                    "No evidence of retrieval design, grounding, or RAG evaluation was found.",
                    "GitHub: basic exception handling is present, without retry or contract-test evidence.",
                    "No container or deployment evidence was found in the supplied sources.",
                    "Resume: participated in manual model review, with no observability implementation evidence.",
                ],
            ),
            "strengths": ["Some relevant Python and model-review exposure."],
            "gaps": ["Missing evidence for RAG, deployment, and production API ownership."],
            "red_flags": ["Several resume claims could not be connected to public implementation evidence."],
            "overall_summary": "The profile shows adjacent experience but does not yet demonstrate the production ownership required for this role.",
            "decision": "Reject",
            "confidence": "Medium",
            "confidence_reason": "Only two limited sources were available and several claims lack implementation evidence.",
        },
        "source_status": {"github": True, "linkedin": False, "resume": True},
        "raw": "",
        "error": None,
    },
]


SAMPLE_SUMMARIES = {
    "Amina Karimova": "Production AI engineer with evidence-backed FastAPI, RAG evaluation, and Docker deployment experience.",
    "Daniel Ortiz": "Backend engineer with strong API reliability evidence and partial RAG implementation experience.",
    "Noah Williams": "Python engineer with adjacent ML experience and limited evidence for production RAG ownership.",
}
