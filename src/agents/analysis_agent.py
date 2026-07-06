"""
AnalysisAgent — produces a UPSC-oriented deep-dive for a single selected topic.

Modes:
  • LLM mode  : Gemini (google-genai SDK) when GEMINI_API_KEY is set.
  • Heuristic : template-based fallback, no API key required.
"""

from __future__ import annotations

import json
import logging
import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.schemas import TopicAnalysis
from src.utils.prompts import (
    ANALYSIS_AGENT_SYSTEM,
    ANALYSIS_AGENT_USER_TEMPLATE,
    HEURISTIC_UPSC_KEYWORDS,
)

load_dotenv()

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-2.0-flash"
_TEMPERATURE  = 0.3
_MAX_TOKENS   = 2048

# ── Static heuristic content bank ─────────────────────────────────────────────

_HEURISTIC_PRELIMS = {
    "Polity":                  "Focus on relevant constitutional articles, amendment history, landmark Supreme Court judgements, and the bodies / authorities involved.",
    "Economy":                 "Note key indicators (growth rate, inflation target, fiscal deficit limit), the regulatory authority, and any recent policy changes or budget allocations.",
    "International Relations": "Identify the parties involved, treaty / agreement name, year, and India's strategic interest. Note any multilateral forum linkage (UN, G20, SCO, etc.).",
    "Environment":             "Remember the legislation involved (e.g. Environment Protection Act, Wildlife Protection Act), nodal ministry, and any international convention linkage (CBD, UNFCCC, Ramsar).",
    "Science & Tech":          "Note the agency / organisation (ISRO, DRDO, DST), the technology or mission name, and India's global ranking or milestone.",
    "Governance":              "Identify the scheme / policy name, the ministry, year of launch, target beneficiaries, and any convergence with other flagship schemes.",
    "Social Issues":           "Focus on the latest census / NFHS / NSSO data point, the constitutional provision (Article 15, 21, 46, etc.), and the ministry responsible.",
    "History/Culture":         "Note the period (ancient / medieval / modern), key figures, UNESCO recognition status if applicable, and any recently-in-news angle.",
    "Geography":               "Identify the location (state, river basin, mountain range), any disaster / climate linkage, and relevant government body (NDMA, CWC, etc.).",
    "Miscellaneous":           "Extract key named entities, dates, and any government or international body mentioned for static recall.",
}

_HEURISTIC_MAINS = {
    "Polity":                  "Examine the constitutional validity, separation of powers implications, and judicial vs executive balance. Suggest reforms and way forward drawing from Law Commission / Sarkaria / Punchhi recommendations.",
    "Economy":                 "Analyse causes (structural / cyclical), short-term vs long-term impact on growth, employment, and fiscal space. Way forward: reforms, international coordination, inclusive growth lens.",
    "International Relations": "Discuss India's strategic calculus, neighbourhood-first / Act East / multilateral dimensions. Analyse impact on regional stability and India's foreign policy doctrine. Way forward: diplomatic engagement.",
    "Environment":             "Cover the ecology–development tension, impact on vulnerable communities, India's international commitments. Way forward: green growth, technology transfer, international climate finance.",
    "Science & Tech":          "Link to India's innovation ecosystem, Atmanirbhar Bharat, and global competitiveness. Discuss ethical / security dimensions where relevant. Way forward: R&D investment, public-private partnerships.",
    "Governance":              "Examine implementation gaps, federalism angles, and accountability mechanisms. Suggest strengthening monitoring frameworks and community participation. Way forward: technology-driven governance.",
    "Social Issues":           "Apply a rights-based framework. Discuss intersectionality (gender, caste, income), data gaps, and ground-level implementation challenges. Way forward: convergence of schemes, community ownership.",
    "History/Culture":         "Connect to India's civilisational continuity, soft power, and national identity. Examine colonial legacy where relevant. Way forward: preservation, documentation, and digital heritage.",
    "Geography":               "Link physical geography to human / economic geography outcomes. Discuss disaster risk reduction, climate adaptation. Way forward: resilient infrastructure, early warning systems.",
    "Miscellaneous":           "Contextualise within India's development goals (SDGs, Vision 2047). Identify stakeholders, trade-offs, and a balanced way forward.",
}


class AnalysisAgent:
    """
    Generates a structured UPSC deep-dive for a single topic.

    Usage:
        agent = AnalysisAgent()
        analysis = agent.analyse(topic_dict)   # returns dict
    """

    def __init__(self) -> None:
        self.mode = _detect_mode()
        self._client: genai.Client | None = None
        if self.mode == "llm":
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyse(self, topic: dict) -> dict:
        """
        Produce a TopicAnalysis for the given TopicCandidate dict.
        Always returns a dict; never raises (falls back to heuristic on error).
        """
        try:
            if self.mode == "llm":
                return self._call_gemini(topic)
            return self._heuristic_analyse(topic)
        except Exception:
            return self._heuristic_analyse(topic)

    # ── LLM mode ───────────────────────────────────────────────────────────────

    def _build_prompt(self, topic: dict) -> str:
        gs_papers = ", ".join(topic.get("gs_paper_tags", ["GS3"]))
        source_context = topic.get("source_chunk_preview", topic.get("why_relevant", ""))
        return ANALYSIS_AGENT_USER_TEMPLATE.format(
            topic_title=topic.get("topic_title", "Unknown Topic"),
            subject_tag=topic.get("subject_tag", "Miscellaneous"),
            gs_papers=gs_papers,
            source_context=source_context,
        )

    def _call_gemini(self, topic: dict) -> dict:
        """
        Call Gemini with JSON-mode and parse into a validated TopicAnalysis dict.
        Falls back to heuristic on any API or parse error.
        """
        try:
            response = self._client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=self._build_prompt(topic),
                config=types.GenerateContentConfig(
                    system_instruction=ANALYSIS_AGENT_SYSTEM,
                    temperature=_TEMPERATURE,
                    max_output_tokens=_MAX_TOKENS,
                    response_mime_type="application/json",
                ),
            )
            return _parse_llm_response(response.text, topic)
        except Exception as exc:
            logger.warning("AnalysisAgent Gemini call failed: %s — using heuristic", exc)
            return self._heuristic_analyse(topic)

    # ── Heuristic mode ─────────────────────────────────────────────────────────

    def _heuristic_analyse(self, topic: dict) -> dict:
        """Template-driven fallback — works without any API key."""
        title   = topic.get("topic_title", "This Topic")
        subject = topic.get("subject_tag", "Miscellaneous")
        preview = topic.get("source_chunk_preview", "")
        why     = topic.get("why_relevant", "")
        gs_str  = " / ".join(topic.get("gs_paper_tags", ["GS3"]))

        subject_keywords = HEURISTIC_UPSC_KEYWORDS.get(subject, [])
        lower_preview = (preview + " " + why).lower()
        matched_kws = [kw for kw in subject_keywords if kw in lower_preview][:7]
        if len(matched_kws) < 3:
            matched_kws = subject_keywords[:5]

        data = {
            "topic_title": title,
            "concise_summary": (
                f"{title} is a current-affairs development relevant to the {subject} domain. "
                f"{why} It is mapped to {gs_str} of UPSC CSE Mains."
            ),
            "background_context": (
                f"The {subject} domain has been a recurring theme in UPSC CSE, featuring in both "
                f"Prelims and Mains over the past several years. "
                f"{title} connects to broader policy, legislative, or geopolitical shifts that "
                f"aspirants must contextualise within India's constitutional and governance framework. "
                f"Understanding the historical evolution of this issue helps build a stronger Mains answer."
            ),
            "why_it_matters_for_upsc": (
                f"UPSC regularly tests {subject} topics because they reflect India's evolving policy "
                f"landscape and constitutional priorities. {title} is likely to appear as a Mains "
                f"{gs_str} question requiring analytical depth, or as a Prelims MCQ testing a "
                f"specific fact, act, or body."
            ),
            "prelims_angle": _HEURISTIC_PRELIMS.get(
                subject,
                "Focus on key named entities, relevant legislation, nodal ministry, and any numerical data mentioned.",
            ),
            "mains_angle": _HEURISTIC_MAINS.get(
                subject,
                "Analyse causes, stakeholders, short- and long-term implications, and a balanced way forward.",
            ),
            "revision_bullets": [
                f"{title} falls under the {subject} domain, relevant to {gs_str}.",
                f"Key exam angle: {topic.get('exam_tags', ['Mains'])[0]} — {why[:120]}",
                "Identify the nodal ministry / agency / constitutional body involved.",
                "Note any legislation, amendment, scheme, or treaty name for static recall.",
                "Practice one Mains answer: 'Critically examine [topic] in the context of [policy goal].'",
            ],
            "keywords_to_remember": matched_kws or [subject.lower(), "current affairs", "upsc cse"],
        }

        return TopicAnalysis(**data).to_dict()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _detect_mode() -> str:
    return "llm" if os.getenv("GEMINI_API_KEY") else "heuristic"


def _parse_llm_response(raw: str, topic: dict) -> dict:
    """
    Validate raw Gemini JSON into a TopicAnalysis dict.
    Falls back to heuristic on any error.
    """
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(cleaned)
        data.setdefault("topic_title", topic.get("topic_title", "Unknown"))
        return TopicAnalysis(**data).to_dict()
    except Exception as exc:
        logger.warning("AnalysisAgent parse failed: %s — using heuristic", exc)
        agent = AnalysisAgent.__new__(AnalysisAgent)
        agent.mode = "heuristic"
        return agent._heuristic_analyse(topic)
