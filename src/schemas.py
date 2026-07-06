"""
Data schemas for CA Compass topic candidates.

Using Pydantic v2 for validation and easy JSON serialisation.
If you prefer stdlib dataclasses, a plain-dict fallback is also supported
in the agent layer — but Pydantic is cleaner for future API exposure.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ── Allowed values ─────────────────────────────────────────────────────────────

SubjectTag = Literal[
    "Polity",
    "Economy",
    "International Relations",
    "Environment",
    "Science & Tech",
    "Social Issues",
    "Governance",
    "History/Culture",
    "Geography",
    "Miscellaneous",
]

ExamTag = Literal["Prelims", "Mains"]

GSPaperTag = Literal["GS1", "GS2", "GS3", "GS4", "Essay"]


# ── Main schema ────────────────────────────────────────────────────────────────

class TopicCandidate(BaseModel):
    """Represents a single UPSC-relevant topic identified from a PDF chunk."""

    topic_title: str = Field(..., min_length=3, description="Short, descriptive topic name")
    relevance_score: int = Field(..., ge=0, le=100, description="0–100 UPSC relevance score")
    subject_tag: SubjectTag = Field(..., description="Broad UPSC subject area")
    exam_tags: list[ExamTag] = Field(..., min_length=1, description="Prelims and/or Mains")
    gs_paper_tags: list[GSPaperTag] = Field(..., min_length=1, description="Relevant GS papers")
    why_relevant: str = Field(..., min_length=10, description="1–3 sentence explanation")
    source_chunk_preview: str = Field(
        ..., description="Short snippet from the source chunk (≤300 chars)"
    )

    @field_validator("source_chunk_preview")
    @classmethod
    def truncate_preview(cls, v: str) -> str:
        """Ensure the preview doesn't balloon in the UI."""
        return v[:300].strip() + ("…" if len(v) > 300 else "")

    @field_validator("exam_tags", "gs_paper_tags", mode="before")
    @classmethod
    def deduplicate_lists(cls, v: list) -> list:
        """Remove duplicates while preserving order."""
        seen: set = set()
        return [x for x in v if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]

    def to_dict(self) -> dict:
        """Convenience method for Streamlit rendering."""
        return self.model_dump()


# ── Deep-dive analysis schema ──────────────────────────────────────────────────

class TopicAnalysis(BaseModel):
    """
    Full UPSC-oriented deep-dive for a single selected topic.
    Produced by AnalysisAgent.
    """

    topic_title: str = Field(..., description="Topic being analysed (mirrors TopicCandidate)")
    concise_summary: str = Field(..., description="2–3 sentence plain-English summary")
    background_context: str = Field(..., description="Historical / policy background (3–5 sentences)")
    why_it_matters_for_upsc: str = Field(..., description="Why UPSC tests this; exam pattern link")
    prelims_angle: str = Field(..., description="Key factual / static points for MCQs")
    mains_angle: str = Field(..., description="Analytical dimensions for essay / GS answers")
    revision_bullets: list[str] = Field(..., min_length=3, description="5 crisp revision bullets")
    keywords_to_remember: list[str] = Field(..., min_length=3, description="3–7 keywords/terms")

    @field_validator("revision_bullets", mode="before")
    @classmethod
    def cap_bullets(cls, v: list) -> list:
        return v[:7]   # never return more than 7 even if LLM overshoots

    @field_validator("keywords_to_remember", mode="before")
    @classmethod
    def cap_keywords(cls, v: list) -> list:
        return v[:7]

    def to_dict(self) -> dict:
        return self.model_dump()


# ── Exam practice schemas ──────────────────────────────────────────────────────

class PYQRecord(BaseModel):
    """A single Previous Year Question retrieved from the CSV dataset."""

    year: int = Field(..., description="Year the question appeared")
    exam_stage: Literal["Prelims", "Mains"] = Field(..., description="Prelims or Mains")
    subject: str = Field(..., description="Subject category in the CSV")
    topic: str = Field(..., description="Specific topic tag in the CSV")
    question: str = Field(..., description="Full question text")
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(default="Medium")

    def to_dict(self) -> dict:
        return self.model_dump()


class PrelimsQuestion(BaseModel):
    """A generated Prelims-style MCQ."""

    question: str = Field(..., description="Question stem")
    options: list[str] = Field(..., min_length=4, max_length=4, description="Exactly 4 options")
    correct_answer: str = Field(..., description="Correct option label, e.g. 'A'")
    explanation: str = Field(..., description="Brief explanation of the correct answer")

    def to_dict(self) -> dict:
        return self.model_dump()


class ExamPractice(BaseModel):
    """
    Full exam practice output produced by ExamCoachAgent for a selected topic.
    """

    topic_title: str = Field(..., description="Topic this practice set is for")
    related_pyqs: list[dict] = Field(default_factory=list, description="Retrieved PYQs from CSV")
    generated_prelims: list[dict] = Field(..., min_length=1, description="3 generated MCQs")
    generated_mains: str = Field(..., description="One UPSC-style 10/15-mark Mains question")
    revision_topics: list[str] = Field(..., min_length=3, description="~5 concepts to revise")
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(..., description="Overall difficulty level")
    similar_themes: list[str] = Field(..., min_length=2, description="3–5 related UPSC themes")

    @field_validator("generated_prelims", mode="before")
    @classmethod
    def cap_prelims(cls, v: list) -> list:
        return v[:3]

    @field_validator("similar_themes", mode="before")
    @classmethod
    def cap_themes(cls, v: list) -> list:
        return v[:5]

    @field_validator("revision_topics", mode="before")
    @classmethod
    def cap_revision(cls, v: list) -> list:
        return v[:5]

    def to_dict(self) -> dict:
        return self.model_dump()


# ── Study notebook schema ──────────────────────────────────────────────────────

class NotebookRecord(BaseModel):
    """
    A single persisted study session entry.
    Mirrors the dict written by StudyNotebook.save_record() — used for
    validation when re-loading records from disk.
    """

    record_id: str = Field(..., description="Short UUID4 identifier")
    saved_at: str = Field(..., description="ISO-8601 datetime string")
    topic_title: str = Field(..., description="Topic studied")
    subject: str = Field(..., description="UPSC subject tag")
    summary: str = Field(default="", description="Concise summary from AnalysisAgent")
    revision_bullets: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    related_pyqs: list[dict] = Field(default_factory=list)
    generated_mains: str = Field(default="", description="Generated Mains question")
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(default="Medium")
    similar_themes: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump()
