"""
ExamCoachAgent — connects a current-affairs topic to UPSC exam practice.

Flow:
  1. retrieve_pyqs()  : lightweight CSV search -> top-3 related PYQs
  2. generate_practice() : LLM or heuristic -> MCQs, Mains Q, revision topics, themes

Architecture note:
  retrieve_pyqs() is intentionally isolated so it can be replaced with a
  semantic / vector search later without touching the rest of the agent.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from difflib import SequenceMatcher

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.schemas import ExamPractice, PYQRecord, PrelimsQuestion
from src.utils.prompts import (
    EXAM_COACH_SYSTEM,
    EXAM_COACH_USER_TEMPLATE,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_CSV_PATH = Path(__file__).parent.parent / "data" / "pyqs_sample.csv"
_TOP_PYQS = 3


# ══════════════════════════════════════════════════════════════════════════════
# PYQ Retrieval  (swap this function for semantic search in a future step)
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_pyqs(topic_title: str, subject_tag: str, keywords: list[str]) -> list[dict]:
    """
    Search the local CSV for the most relevant previous year questions.

    Scoring (additive):
      +4  exact subject match
      +2  per keyword found in question or topic column
      +1  partial title similarity (SequenceMatcher ≥ 0.3)

    Returns up to _TOP_PYQS dicts matching PYQRecord fields.
    This function is the only place that knows about the CSV — swap it for a
    vector-store call and the rest of the agent stays unchanged.
    """
    if not _CSV_PATH.exists():
        return []

    rows: list[dict] = []
    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    lower_title = topic_title.lower()
    lower_kws = [k.lower() for k in keywords]

    # Normalise subject_tag → CSV subject column (they're the same vocabulary)
    target_subject = subject_tag.lower()

    scored: list[tuple[int, dict]] = []
    for row in rows:
        score = 0
        row_subject = row.get("subject", "").lower()
        row_topic   = row.get("topic", "").lower()
        row_question = row.get("question", "").lower()

        # Subject match
        if row_subject == target_subject:
            score += 4
        elif target_subject in row_subject or row_subject in target_subject:
            score += 2

        # Keyword hits in question text and topic column
        haystack = row_question + " " + row_topic
        for kw in lower_kws:
            if kw in haystack:
                score += 2

        # Partial title similarity
        sim = SequenceMatcher(None, lower_title, row_topic).ratio()
        if sim >= 0.3:
            score += int(sim * 3)   # up to +3

        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [row for _, row in scored[:_TOP_PYQS]]

    # Convert to PYQRecord dicts
    result = []
    for row in top:
        try:
            rec = PYQRecord(
                year=int(row.get("year", 0)),
                exam_stage=row.get("exam_stage", ""),
                subject=row.get("subject", ""),
                topic=row.get("topic", ""),
                question=row.get("question", ""),
                difficulty=row.get("difficulty", "Medium"),
            )
            result.append(rec.to_dict())
        except Exception:
            continue

    return result

class ExamCoachAgent:
    """
    Generates exam practice material for a selected UPSC topic.

    Usage:
        agent = ExamCoachAgent()
        practice = agent.generate(topic_dict, analysis_dict)  # returns dict
    """

    def __init__(self) -> None:
        self.mode = _detect_mode()
        self._client: genai.Client | None = None
        if self.mode == "llm":
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(self, topic: dict, analysis: dict | None = None) -> dict:
        """
        Produce an ExamPractice dict for the given topic.

        analysis is optional but improves output quality if provided.
        Never raises -- falls back to heuristic on any error.
        """
        # Step 1: retrieve PYQs (always -- LLM mode also uses them)
        keywords = list(analysis.get("keywords_to_remember", []) if analysis else [])
        related_pyqs = retrieve_pyqs(
            topic_title=topic.get("topic_title", ""),
            subject_tag=topic.get("subject_tag", "Miscellaneous"),
            keywords=keywords,
        )

        # Step 2: generate practice content
        try:
            if self.mode == "llm":
                practice = self._call_llm(topic, analysis, related_pyqs)
            else:
                practice = self._heuristic_generate(topic, analysis, related_pyqs)
        except Exception:
            practice = self._heuristic_generate(topic, analysis, related_pyqs)

        return practice

    # ── LLM mode ───────────────────────────────────────────────────────────────

    def _call_llm(self, topic: dict, analysis: dict | None, related_pyqs: list[dict]) -> dict:
        if os.getenv("GEMINI_API_KEY"):
            return self._call_gemini(topic, analysis, related_pyqs)
        return self._heuristic_generate(topic, analysis, related_pyqs)

    def _build_prompt(self, topic: dict, analysis: dict | None, related_pyqs: list[dict]) -> str:
        pyq_text = "\n".join(
            f"- [{r.get('year')} {r.get('exam_stage')}] {r.get('question', '')}"
            for r in related_pyqs
        ) or "No closely matching PYQs found."

        context = ""
        if analysis:
            context = (
                f"Summary: {analysis.get('concise_summary', '')}\n"
                f"Prelims angle: {analysis.get('prelims_angle', '')}\n"
                f"Mains angle: {analysis.get('mains_angle', '')}"
            )

        return EXAM_COACH_USER_TEMPLATE.format(
            topic_title=topic.get("topic_title", "Unknown Topic"),
            subject_tag=topic.get("subject_tag", "Miscellaneous"),
            gs_papers=", ".join(topic.get("gs_paper_tags", ["GS3"])),
            context=context,
            related_pyqs=pyq_text,
        )

    def _call_gemini(self, topic: dict, analysis: dict | None, related_pyqs: list[dict]) -> dict:
        """
        Call Gemini with JSON-mode and parse into a validated ExamPractice dict.
        Falls back to heuristic on any API or parse error.
        """
        try:
            response = self._client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=self._build_prompt(topic, analysis, related_pyqs),
                config=types.GenerateContentConfig(
                    system_instruction=EXAM_COACH_SYSTEM,
                    temperature=_TEMPERATURE,
                    max_output_tokens=_MAX_TOKENS,
                    response_mime_type="application/json",
                ),
            )
            return _parse_llm_response(response.text, topic, related_pyqs)
        except Exception as exc:
            logger.warning("ExamCoachAgent Gemini call failed: %s -- using heuristic", exc)
            return self._heuristic_generate(topic, analysis, related_pyqs)

    # ── Heuristic mode ─────────────────────────────────────────────────────────

    def _heuristic_generate(
        self,
        topic: dict,
        analysis: dict | None,
        related_pyqs: list[dict],
    ) -> dict:
        """
        Template-driven exam practice — works without any API key.
        """
        title   = topic.get("topic_title", "This Topic")
        subject = topic.get("subject_tag", "Miscellaneous")
        gs_list = topic.get("gs_paper_tags", ["GS3"])
        gs_str  = " / ".join(gs_list)

        keywords = []
        if analysis:
            keywords = analysis.get("keywords_to_remember", [])

        # ── Prelims MCQs (3 templated questions) ──────────────────────────────
        generated_prelims = _heuristic_prelims(title, subject, keywords)

        # ── Mains question ────────────────────────────────────────────────────
        generated_mains = _heuristic_mains(title, subject, gs_str)

        # ── Revision topics ───────────────────────────────────────────────────
        revision_topics = _heuristic_revision_topics(title, subject, keywords)

        # ── Difficulty ────────────────────────────────────────────────────────
        score = topic.get("relevance_score", 50)
        difficulty = "Hard" if score >= 75 else "Medium" if score >= 45 else "Easy"

        # ── Similar themes ────────────────────────────────────────────────────
        similar_themes = _SIMILAR_THEMES.get(subject, _SIMILAR_THEMES["Miscellaneous"])

        data = {
            "topic_title": title,
            "related_pyqs": related_pyqs,
            "generated_prelims": [q.to_dict() for q in generated_prelims],
            "generated_mains": generated_mains,
            "revision_topics": revision_topics,
            "difficulty": difficulty,
            "similar_themes": similar_themes,
        }

        return ExamPractice(**data).to_dict()


# ── Heuristic content helpers ──────────────────────────────────────────────────

def _heuristic_prelims(title: str, subject: str, keywords: list[str]) -> list[PrelimsQuestion]:
    """Generate 3 templated Prelims MCQs relevant to the subject and topic."""
    templates = _PRELIMS_TEMPLATES.get(subject, _PRELIMS_TEMPLATES["Miscellaneous"])
    questions = []
    for i, tmpl in enumerate(templates[:3]):
        q = PrelimsQuestion(
            question=tmpl["question"].format(topic=title),
            options=tmpl["options"],
            correct_answer=tmpl["correct_answer"],
            explanation=tmpl["explanation"].format(topic=title, subject=subject),
        )
        questions.append(q)
    return questions


def _heuristic_mains(title: str, subject: str, gs_str: str) -> str:
    template = _MAINS_TEMPLATES.get(subject, _MAINS_TEMPLATES["Miscellaneous"])
    return template.format(topic=title, gs=gs_str)


def _heuristic_revision_topics(title: str, subject: str, keywords: list[str]) -> list[str]:
    base = _REVISION_TOPICS.get(subject, _REVISION_TOPICS["Miscellaneous"])
    # Inject extracted keywords as the first item if available
    if keywords:
        kw_item = f"Key terms: {', '.join(keywords[:4])}"
        return [kw_item] + base[:4]
    return base[:5]


# ── Static template banks ──────────────────────────────────────────────────────

_PRELIMS_TEMPLATES: dict[str, list[dict]] = {
    "Polity": [
        {
            "question": "With reference to '{topic}', which of the following statements is correct?",
            "options": [
                "A) It is governed by a Constitutional amendment",
                "B) It falls under the Concurrent List of the Seventh Schedule",
                "C) It is administered by a body constituted under an Act of Parliament",
                "D) It is mentioned explicitly in the Directive Principles",
            ],
            "correct_answer": "C",
            "explanation": "Topics in {subject} are often administered by statutory bodies. Review the relevant Act, Articles, and landmark Supreme Court judgements for {topic}.",
        },
        {
            "question": "Consider the following statements about '{topic}': 1. It involves both the Union and State governments. 2. It has been subject to judicial review. Which of the above is/are correct?",
            "options": ["A) 1 only", "B) 2 only", "C) Both 1 and 2", "D) Neither 1 nor 2"],
            "correct_answer": "C",
            "explanation": "Most {subject} issues involve federal dimensions and judicial oversight. Knowing the constitutional provisions is key for {topic}.",
        },
        {
            "question": "Which of the following best describes the significance of '{topic}' in the context of Indian democracy?",
            "options": [
                "A) It strengthens executive accountability",
                "B) It enhances legislative supremacy",
                "C) It promotes judicial independence",
                "D) It balances centre-state relations",
            ],
            "correct_answer": "A",
            "explanation": "In {subject}, the focus is usually on accountability and governance. For {topic}, connect it to relevant constitutional provisions.",
        },
    ],
    "Economy": [
        {
            "question": "Which of the following is a direct implication of '{topic}' on the Indian economy?",
            "options": [
                "A) Increase in fiscal deficit",
                "B) Impact on monetary policy transmission",
                "C) Effect on current account balance",
                "D) All of the above depending on the scale",
            ],
            "correct_answer": "D",
            "explanation": "Economic topics like {topic} often have multi-dimensional effects. Always analyse fiscal, monetary, and external sector implications.",
        },
        {
            "question": "Consider the following about '{topic}': 1. It is regulated by RBI or SEBI. 2. It directly affects the common citizen. Which is/are correct?",
            "options": ["A) 1 only", "B) 2 only", "C) Both 1 and 2", "D) Neither"],
            "correct_answer": "C",
            "explanation": "Most {subject} issues connect regulatory bodies to ground-level impact. For {topic}, identify the regulator and the beneficiary.",
        },
        {
            "question": "'{topic}' is most closely associated with which of the following policy objectives?",
            "options": [
                "A) Price stability",
                "B) Employment generation",
                "C) Financial inclusion",
                "D) Export promotion",
            ],
            "correct_answer": "C",
            "explanation": "In the {subject} domain, UPSC tests your ability to link news events to policy goals. For {topic}, identify the primary objective.",
        },
    ],
    "Environment": [
        {
            "question": "With reference to '{topic}', consider the following: 1. India is a signatory to the related international convention. 2. A nodal ministry oversees its implementation in India. Which is/are correct?",
            "options": ["A) 1 only", "B) 2 only", "C) Both 1 and 2", "D) Neither"],
            "correct_answer": "C",
            "explanation": "Environment topics like {topic} always have an international convention link and a domestic nodal ministry. Identify both for the exam.",
        },
        {
            "question": "Which of the following best describes the UPSC significance of '{topic}'?",
            "options": [
                "A) It is relevant only for Mains GS3",
                "B) It links to both biodiversity loss and climate change",
                "C) It is exclusively a state subject under the Constitution",
                "D) It was enacted before India's independence",
            ],
            "correct_answer": "B",
            "explanation": "Environment questions like {topic} are cross-cutting — they connect climate, biodiversity, and governance. For {subject}, always look for the multi-dimensional angle.",
        },
        {
            "question": "'{topic}' is governed under which legislative or regulatory framework in India?",
            "options": [
                "A) The Wildlife Protection Act, 1972",
                "B) The Environment Protection Act, 1986",
                "C) The Forest Conservation Act, 1980",
                "D) Requires identification from the source",
            ],
            "correct_answer": "D",
            "explanation": "The correct legislation depends on the specific issue in {topic}. Prelims MCQs often test whether you know the exact Act — verify from the source context.",
        },
    ],
    "International Relations": [
        {
            "question": "Which of the following is the primary multilateral forum associated with '{topic}'?",
            "options": [
                "A) United Nations General Assembly",
                "B) G20",
                "C) Shanghai Cooperation Organisation",
                "D) Depends on the specific issue — verify from context",
            ],
            "correct_answer": "D",
            "explanation": "IR topics like {topic} must be linked to a specific forum. Identify whether it is bilateral, regional, or global in scope.",
        },
        {
            "question": "India's policy on '{topic}' best reflects which foreign policy principle?",
            "options": [
                "A) Non-alignment",
                "B) Strategic autonomy",
                "C) Neighbourhood first",
                "D) Act East policy",
            ],
            "correct_answer": "B",
            "explanation": "India's foreign policy in {subject} is increasingly characterised by strategic autonomy. Connect {topic} to India's broader geopolitical interests.",
        },
        {
            "question": "Consider the following about '{topic}': 1. It has implications for India's maritime security. 2. It involves a treaty obligation for India. Which is/are correct?",
            "options": ["A) 1 only", "B) 2 only", "C) Both 1 and 2", "D) Neither"],
            "correct_answer": "A",
            "explanation": "IR questions test both factual recall and analytical understanding. For {topic}, check whether a treaty or convention is directly involved.",
        },
    ],
    "Science & Tech": [
        {
            "question": "Which Indian agency or ministry primarily oversees developments related to '{topic}'?",
            "options": [
                "A) ISRO under the Department of Space",
                "B) DRDO under the Ministry of Defence",
                "C) DST under the Ministry of Science and Technology",
                "D) Depends on the specific technology area",
            ],
            "correct_answer": "D",
            "explanation": "Science & Tech topics like {topic} are governed by different agencies depending on the domain. Always identify the nodal agency for Prelims.",
        },
        {
            "question": "'{topic}' is significant for India's goals under which of the following?",
            "options": [
                "A) Atmanirbhar Bharat",
                "B) National Innovation Mission",
                "C) Production Linked Incentive scheme",
                "D) All of the above may apply",
            ],
            "correct_answer": "D",
            "explanation": "In {subject}, UPSC tests linkages between technology and policy. {topic} may connect to one or more national initiatives — read the context carefully.",
        },
        {
            "question": "Consider the following statements about '{topic}': 1. India has indigenous capability in this technology. 2. It has both civilian and defence applications. Which is/are correct?",
            "options": ["A) 1 only", "B) 2 only", "C) Both 1 and 2", "D) Neither"],
            "correct_answer": "C",
            "explanation": "Most technology topics tested in UPSC, including {topic}, have dual-use potential. Connect to India's self-reliance goals for the best answer.",
        },
    ],
    "Governance": [
        {
            "question": "'{topic}' is primarily associated with which of the following governance objectives?",
            "options": [
                "A) Improving last-mile delivery of welfare schemes",
                "B) Strengthening anti-corruption mechanisms",
                "C) Digitising government services",
                "D) All depending on the scope",
            ],
            "correct_answer": "D",
            "explanation": "Governance topics like {topic} are multi-dimensional. Identify the specific objective — transparency, delivery, accountability — for a precise Prelims answer.",
        },
        {
            "question": "Which of the following best describes the role of technology in implementing '{topic}'?",
            "options": [
                "A) Technology is irrelevant to this topic",
                "B) Digital platforms are central to its delivery",
                "C) Technology only plays a monitoring role",
                "D) Technology usage is constitutionally restricted",
            ],
            "correct_answer": "B",
            "explanation": "Modern {subject} reforms, including {topic}, almost always involve digital infrastructure. Note the technology platform and the beneficiary group.",
        },
        {
            "question": "'{topic}' relates most closely to which tier of government?",
            "options": [
                "A) Central government only",
                "B) State governments primarily",
                "C) Local self-government bodies",
                "D) Concurrent responsibility across tiers",
            ],
            "correct_answer": "D",
            "explanation": "Governance in India is a federal responsibility. For {topic}, identify the level at which implementation happens — this is a common Prelims angle.",
        },
    ],
    "Social Issues": [
        {
            "question": "'{topic}' is most closely associated with which Sustainable Development Goal (SDG)?",
            "options": [
                "A) SDG 1 — No Poverty",
                "B) SDG 3 — Good Health and Well-being",
                "C) SDG 5 — Gender Equality",
                "D) Depends on the specific social issue",
            ],
            "correct_answer": "D",
            "explanation": "UPSC frequently links social issues to SDGs. For {topic}, identify the primary SDG and any secondary SDGs for a comprehensive Mains answer.",
        },
        {
            "question": "Which constitutional provision most directly addresses '{topic}'?",
            "options": [
                "A) Article 14 — Right to Equality",
                "B) Article 21 — Right to Life",
                "C) Article 46 — Promotion of educational interests of SCs and STs",
                "D) Requires identification from the specific issue",
            ],
            "correct_answer": "D",
            "explanation": "Social issues in {subject} connect to Fundamental Rights and Directive Principles. For {topic}, identify the relevant constitutional article.",
        },
        {
            "question": "Consider the following: 1. {topic} disproportionately affects women and children. 2. Government schemes address this through cash transfers. Which is/are likely correct?",
            "options": ["A) 1 only", "B) 2 only", "C) Both 1 and 2", "D) Neither"],
            "correct_answer": "A",
            "explanation": "Many social issues have a gender and child welfare dimension. For {topic}, check the data from NFHS or NSSO to support Mains answers.",
        },
    ],
    "History/Culture": [
        {
            "question": "'{topic}' is most closely associated with which period of Indian history?",
            "options": [
                "A) Ancient India (pre-600 CE)",
                "B) Medieval India (600–1750 CE)",
                "C) Modern India (post-1750 CE)",
                "D) Requires identification from context",
            ],
            "correct_answer": "D",
            "explanation": "History questions in UPSC test period mapping. For {topic}, identify the specific era and its key characteristics.",
        },
        {
            "question": "Which of the following is a correct association with '{topic}'?",
            "options": [
                "A) It has been awarded UNESCO recognition",
                "B) It is protected under the Ancient Monuments Act",
                "C) It is associated with a major freedom movement event",
                "D) Verify from source — all options are plausible",
            ],
            "correct_answer": "D",
            "explanation": "For {subject} topics like {topic}, UPSC tests associations with legislation, UNESCO status, or historical events. Always verify from the source.",
        },
        {
            "question": "The significance of '{topic}' for UPSC CSE lies primarily in its:",
            "options": [
                "A) Connection to India's civilisational heritage",
                "B) Relevance to colonial history",
                "C) Link to India's freedom struggle",
                "D) All may apply — identify from context",
            ],
            "correct_answer": "D",
            "explanation": "History/Culture questions for {topic} span ancient heritage to modern nationalism. Map the topic to a specific angle — heritage, colonialism, or independence — for precision.",
        },
    ],
    "Geography": [
        {
            "question": "'{topic}' is located in or primarily associated with which geographical region of India?",
            "options": [
                "A) The Indo-Gangetic Plain",
                "B) The Deccan Plateau",
                "C) The Western Ghats",
                "D) Requires identification from context",
            ],
            "correct_answer": "D",
            "explanation": "Geography questions test spatial awareness. For {topic}, map it to a specific region, river basin, or physiographic division.",
        },
        {
            "question": "Which of the following climate or disaster event is most closely associated with '{topic}'?",
            "options": [
                "A) Monsoon variability",
                "B) Seismic activity",
                "C) Cyclone patterns",
                "D) Context-dependent",
            ],
            "correct_answer": "D",
            "explanation": "For {subject} topics like {topic}, identify the specific climate hazard or physical feature. Prelims MCQs often test exact geographical associations.",
        },
        {
            "question": "'{topic}' is most relevant to which body of water or river system?",
            "options": [
                "A) The Ganga river system",
                "B) The Peninsular river system",
                "C) The Arabian Sea coast",
                "D) Identify from source context",
            ],
            "correct_answer": "D",
            "explanation": "River systems, coasts, and basins are common Geography anchors. For {topic}, confirm the specific water body from the source article.",
        },
    ],
    "Miscellaneous": [
        {
            "question": "'{topic}' is most likely to appear in UPSC Prelims as a question testing:",
            "options": [
                "A) Static factual recall",
                "B) Current affairs awareness",
                "C) Conceptual understanding",
                "D) All of the above",
            ],
            "correct_answer": "D",
            "explanation": "UPSC Prelims tests multiple dimensions for current affairs topics. For {topic}, prepare static facts, the news angle, and the conceptual framework.",
        },
        {
            "question": "Which of the following best captures the Mains relevance of '{topic}'?",
            "options": [
                "A) It requires a purely factual answer",
                "B) It demands analysis of causes and consequences",
                "C) It is relevant only for Essay paper",
                "D) It has no Mains relevance",
            ],
            "correct_answer": "B",
            "explanation": "UPSC Mains rewards analytical thinking. For {topic}, always structure answers with causes, implications, and a way forward.",
        },
        {
            "question": "'{topic}' is most accurately categorised under which UPSC subject domain?",
            "options": [
                "A) Polity and Governance",
                "B) Economy and Development",
                "C) Environment and Ecology",
                "D) Identify the correct subject from context",
            ],
            "correct_answer": "D",
            "explanation": "Correctly categorising {topic} within a UPSC domain helps you locate relevant static material and PYQs. Use the subject tag as your starting point.",
        },
    ],
}

_MAINS_TEMPLATES: dict[str, str] = {
    "Polity":                 "Critically examine the constitutional and institutional dimensions of '{topic}'. How does it reflect the balance between the fundamental rights of citizens and the directive principles of state policy? Suggest a way forward. (250 words / GS2)",
    "Economy":                "Analyse the economic implications of '{topic}' for India's growth trajectory. Discuss its impact on employment, fiscal stability, and inclusive development. What policy interventions are needed? (250 words / {gs})",
    "International Relations": "Evaluate the significance of '{topic}' for India's foreign policy objectives. How does it affect India's strategic partnerships and regional influence? Suggest a diplomatic way forward. (250 words / GS2)",
    "Environment":            "'{topic}' highlights the tension between development and environmental sustainability. Critically examine its ecological implications and discuss India's regulatory framework and international commitments in addressing this challenge. (250 words / GS3)",
    "Science & Tech":         "Discuss the strategic importance of '{topic}' for India's self-reliance in critical technologies. What are the opportunities and challenges? How can India leverage this for economic and security gains? (250 words / {gs})",
    "Governance":             "Effective governance requires both institutional capacity and citizen participation. Critically examine how '{topic}' reflects the strengths and weaknesses of India's governance architecture. Suggest reforms. (250 words / GS2)",
    "Social Issues":          "'{topic}' reveals the persistent inequalities in Indian society. Using a rights-based approach, critically analyse its causes, consequences, and the effectiveness of government interventions. Suggest a way forward. (250 words / {gs})",
    "History/Culture":        "Examine the historical significance of '{topic}' and its relevance for India's contemporary identity. How can India leverage its cultural heritage for soft power and national integration? (150 words / GS1)",
    "Geography":              "Discuss the geographical factors that make '{topic}' a recurring challenge for India. Examine the policy response and suggest measures for building resilience. (150 words / GS1)",
    "Miscellaneous":          "'{topic}' has emerged as a significant issue in recent times. Critically examine its causes, implications for India's development goals, and suggest a comprehensive way forward. (250 words / {gs})",
}

_REVISION_TOPICS: dict[str, list[str]] = {
    "Polity":                 ["Constitutional provisions and relevant Articles", "Landmark Supreme Court judgements on the topic", "Key parliamentary committees or commissions", "Centre-state relations and federalism angle", "Comparison with international practices"],
    "Economy":                ["Key economic indicators and data points", "Regulatory body and its mandate", "Government schemes and policy interventions", "India's global ranking or treaty obligations", "Impact on different income groups"],
    "International Relations":["Bilateral/multilateral forum involved", "India's strategic interest and foreign policy doctrine", "Key treaties, agreements, or declarations", "Historical background of the relationship", "Regional security implications"],
    "Environment":            ["International conventions and India's commitments", "Relevant legislation and nodal ministry", "Species, ecosystems, or regions involved", "Climate change linkage", "Recent government policies or action plans"],
    "Science & Tech":         ["Nodal agency — ISRO, DRDO, DST, or ministry", "India's milestones and global ranking", "Atmanirbhar Bharat and PLI scheme linkage", "Dual-use (civilian + defence) dimensions", "Ethical or security concerns"],
    "Governance":             ["Scheme name, launch year, and nodal ministry", "Target beneficiaries and coverage data", "Technology platform used", "Federalism — central vs state implementation", "Evaluation metrics and challenges"],
    "Social Issues":          ["Latest NFHS / Census / NSSO data", "Constitutional provisions (Articles 14, 15, 21, 46)", "Relevant government schemes", "SDG linkage", "Gender and intersectionality dimension"],
    "History/Culture":        ["Period classification — ancient / medieval / modern", "Key figures and events", "UNESCO or ASI recognition status", "Colonial legacy angle if applicable", "Connection to national movement"],
    "Geography":              ["Region, state, or river basin", "Climate type and disaster vulnerability", "Relevant government body (NDMA, CWC, IMD)", "India's mineral or resource significance", "Physiographic classification"],
    "Miscellaneous":          ["Key named entities (persons, organisations)", "Relevant legislation or international body", "India's policy position", "Cross-cutting GS paper linkage", "Recent news angle and static background"],
}

_SIMILAR_THEMES: dict[str, list[str]] = {
    "Polity":                 ["Parliamentary Privileges", "Judicial Activism vs Judicial Restraint", "Constitutional Morality", "Federal Disputes and Inter-State Councils", "Electoral Reforms"],
    "Economy":                ["Monetary Policy Transmission", "Fiscal Federalism", "Inclusive Finance and Jan Dhan", "India's External Debt and BoP", "Gig Economy and Labour Reforms"],
    "International Relations":["India's Neighbourhood First Policy", "Multilateral Institutions Reform", "India-US Strategic Partnership", "Indo-Pacific Strategy", "China's Belt and Road Initiative"],
    "Environment":            ["Paris Agreement and NDCs", "Biodiversity Convention — COP15", "Green Hydrogen Mission", "National Clean Air Programme", "Compensatory Afforestation Fund"],
    "Science & Tech":         ["National Deep Tech Startup Policy", "Semiconductor Mission", "India's Space Economy", "AI Ethics and Regulation", "Quantum Computing Mission"],
    "Governance":             ["Direct Benefit Transfer", "Aspirational Districts Programme", "e-Courts and Legal Tech", "Anti-Corruption Framework — Lokpal", "Public Procurement Reforms"],
    "Social Issues":          ["Feminisation of Agriculture", "Intergenerational Poverty", "Urban Homelessness", "Mental Health Policy", "Nutrition Mission — POSHAN Abhiyaan"],
    "History/Culture":        ["Syncretic Culture and Composite Heritage", "Intangible Cultural Heritage — UNESCO", "Tribal Art Forms", "Colonial Architecture Conservation", "National Archives Policy"],
    "Geography":              ["Western Disturbances and North India Weather", "Glacial Lake Outburst Floods", "Mangrove Conservation", "India's Exclusive Economic Zone", "Urban Heat Islands"],
    "Miscellaneous":          ["SDG Localisation in India", "Data Governance and Privacy", "Circular Economy", "One Health Approach", "Disaster Risk Reduction — Sendai Framework"],
}


# ── Helpers ────────────────────────────────────────────────────────────────────


# ── Helpers ────────────────────────────────────────────────────────────────────

_GEMINI_MODEL = "gemini-2.0-flash"
_TEMPERATURE  = 0.4
_MAX_TOKENS   = 3000   # exam practice output is larger than other agents


def _detect_mode() -> str:
    return "llm" if os.getenv("GEMINI_API_KEY") else "heuristic"


def _parse_llm_response(raw: str, topic: dict, related_pyqs: list[dict]) -> dict:
    """Validate raw Gemini JSON into an ExamPractice dict. Falls back to heuristic on error."""
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(cleaned)
        data.setdefault("topic_title", topic.get("topic_title", "Unknown"))
        data.setdefault("related_pyqs", related_pyqs)
        return ExamPractice(**data).to_dict()
    except Exception as exc:
        logger.warning("ExamCoachAgent parse failed: %s -- using heuristic", exc)
        agent = ExamCoachAgent.__new__(ExamCoachAgent)
        agent.mode = "heuristic"
        return agent._heuristic_generate(topic, None, related_pyqs)
