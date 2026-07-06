"""
RelevanceAgent — identifies UPSC-relevant topics from text chunks.

Modes:
  • LLM mode  : Gemini (google-genai SDK) when GEMINI_API_KEY is set.
  • Heuristic : keyword-based scoring, no API key required.
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from difflib import SequenceMatcher

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.schemas import TopicCandidate
from src.utils.prompts import (
    HEURISTIC_UPSC_KEYWORDS,
    RELEVANCE_AGENT_SYSTEM,
    RELEVANCE_AGENT_USER_TEMPLATE,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ── Model config ───────────────────────────────────────────────────────────────
_GEMINI_MODEL   = "gemini-2.0-flash"
_TEMPERATURE    = 0.2     # low temp for factual / structured output
_MAX_TOKENS     = 2048

# ── GS-paper heuristic mapping ─────────────────────────────────────────────────
_GS_MAP: dict[str, list[str]] = {
    "Polity":                  ["GS2"],
    "Economy":                 ["GS3"],
    "International Relations": ["GS2"],
    "Environment":             ["GS3"],
    "Science & Tech":          ["GS3"],
    "Governance":              ["GS2"],
    "Social Issues":           ["GS1", "GS2"],
    "History/Culture":         ["GS1"],
    "Geography":               ["GS1"],
    "Miscellaneous":           ["GS3"],
}

_HIGH_SCORE_PRELIMS_THRESHOLD = 50
_TOP_N = 5


class RelevanceAgent:
    """
    Identifies and ranks UPSC-relevant topics across a list of text chunks.

    Usage:
        agent = RelevanceAgent()
        topics = agent.identify_topics(chunks)   # list[dict]
    """

    def __init__(self) -> None:
        self.mode = _detect_mode()
        self._client: genai.Client | None = None
        if self.mode == "llm":
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # ── Public API ─────────────────────────────────────────────────────────────

    def identify_topics(self, chunks: list[str]) -> list[dict]:
        """
        Process all chunks and return the top-N UPSC-relevant topics as dicts.
        Returns an empty list if nothing relevant is found.
        """
        candidates: list[dict] = []

        for chunk in chunks:
            try:
                if self.mode == "llm":
                    results = self._call_gemini(chunk)
                else:
                    results = self._heuristic_analyse(chunk)
                candidates.extend(results)
            except Exception:
                # Never let a single chunk failure break the whole run
                continue

        candidates = _deduplicate(candidates)
        candidates.sort(key=lambda t: t.get("relevance_score", 0), reverse=True)
        return candidates[:_TOP_N]

    # ── LLM mode ───────────────────────────────────────────────────────────────

    def _call_gemini(self, chunk: str) -> list[dict]:
        """
        Call Gemini with JSON-mode enabled and parse the response into
        a list of validated TopicCandidate dicts.
        Falls back to heuristic on any API or parse error.
        """
        try:
            prompt = RELEVANCE_AGENT_USER_TEMPLATE.format(chunk=chunk)
            response = self._client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=RELEVANCE_AGENT_SYSTEM,
                    temperature=_TEMPERATURE,
                    max_output_tokens=_MAX_TOKENS,
                    response_mime_type="application/json",
                ),
            )
            return _parse_llm_response(response.text, chunk)
        except Exception as exc:
            logger.warning("RelevanceAgent Gemini call failed: %s — using heuristic", exc)
            return self._heuristic_analyse(chunk)

    # ── Heuristic mode ─────────────────────────────────────────────────────────

    def _heuristic_analyse(self, chunk: str) -> list[dict]:
        """
        Keyword-based UPSC relevance scoring. No API key required.
        Returns up to 2 candidates per chunk.
        """
        lower = chunk.lower()
        subject_scores: dict[str, int] = {}

        for subject, keywords in HEURISTIC_UPSC_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in lower)
            if hits > 0:
                subject_scores[subject] = hits

        if not subject_scores:
            return []

        ranked = sorted(subject_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        results = []

        for subject, hits in ranked:
            raw_score = min(hits, 10)
            relevance_score = int(15 + (raw_score / 10) * 70)

            if relevance_score < 30:
                continue

            topic_title = _derive_topic_title(chunk, subject)
            exam_tags = ["Mains"]
            if relevance_score >= _HIGH_SCORE_PRELIMS_THRESHOLD:
                exam_tags = ["Prelims", "Mains"]

            results.append({
                "topic_title":          topic_title,
                "relevance_score":      relevance_score,
                "subject_tag":          subject,
                "exam_tags":            exam_tags,
                "gs_paper_tags":        _GS_MAP.get(subject, ["GS3"]),
                "why_relevant":         _derive_why_relevant(chunk, subject, hits),
                "source_chunk_preview": chunk[:250].strip(),
            })

        return results


# ── Helpers ────────────────────────────────────────────────────────────────────

def _detect_mode() -> str:
    return "llm" if os.getenv("GEMINI_API_KEY") else "heuristic"


def _parse_llm_response(raw: str, chunk: str) -> list[dict]:
    """
    Validate raw Gemini JSON into TopicCandidate dicts.
    Falls back to heuristic for the chunk on any error.
    """
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(cleaned)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array")

        validated = []
        for item in data:
            item.setdefault("source_chunk_preview", chunk[:200])
            try:
                candidate = TopicCandidate(**item)
                validated.append(candidate.to_dict())
            except Exception:
                continue
        return validated

    except Exception as exc:
        logger.warning("RelevanceAgent parse failed: %s — using heuristic for chunk", exc)
        agent = RelevanceAgent.__new__(RelevanceAgent)
        agent.mode = "heuristic"
        return agent._heuristic_analyse(chunk)


def _derive_topic_title(chunk: str, subject: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", chunk)
    for sent in sentences:
        sent = sent.strip()
        if 20 <= len(sent) <= 120:
            return textwrap.shorten(sent, width=90, placeholder="…")
    title = chunk[:60].strip().split("\n")[0]
    return textwrap.shorten(title, width=90, placeholder="…") or f"{subject} — Current Affairs"


def _derive_why_relevant(chunk: str, subject: str, hits: int) -> str:
    quality = "highly" if hits >= 5 else "moderately"
    return (
        f"This passage is {quality} relevant to the {subject} domain in UPSC CSE, "
        f"containing {hits} key term(s) commonly tested in the exam. "
        "Review for potential Prelims MCQs and Mains answer-writing material."
    )


def _deduplicate(candidates: list[dict], similarity_threshold: float = 0.75) -> list[dict]:
    unique: list[dict] = []
    for candidate in candidates:
        title = candidate.get("topic_title", "").lower()
        is_duplicate = False
        for seen in unique:
            seen_title = seen.get("topic_title", "").lower()
            if SequenceMatcher(None, title, seen_title).ratio() >= similarity_threshold:
                if candidate.get("relevance_score", 0) > seen.get("relevance_score", 0):
                    unique.remove(seen)
                    unique.append(candidate)
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(candidate)
    return unique
