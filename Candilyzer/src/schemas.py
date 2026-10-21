from typing import Literal, NotRequired, TypedDict


Decision = Literal["Strong Hire", "Hire", "Interview Recommended", "Reject"]
Confidence = Literal["High", "Medium", "Low"]
Importance = Literal["Must Have", "Nice to Have"]
RequirementMatch = Literal["Met", "Partial", "Not Met"]


class RequirementAnalysis(TypedDict):
    requirement: str
    importance: Importance
    match: RequirementMatch
    evidence: str


class Evaluation(TypedDict):
    requirements_analysis: list[RequirementAnalysis]
    strengths: list[str]
    gaps: list[str]
    red_flags: list[str]
    overall_summary: str
    decision: Decision
    confidence: Confidence
    confidence_reason: str


class SourceStatus(TypedDict):
    github: bool
    linkedin: bool
    resume: bool


class EvaluationResult(TypedDict):
    username: str
    parsed: Evaluation | None
    raw: str
    error: NotRequired[str | None]
    source_status: NotRequired[SourceStatus]
