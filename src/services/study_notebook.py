"""
StudyNotebook — local study session persistence and dashboard analytics.

This is a plain service module, not an agent.
All logic is deterministic; no LLM calls are made here.

Storage: a single JSON file at src/data/study_notebook.json
         (path is resolved relative to this file so it works regardless of
          the working directory Streamlit is launched from).

Public API
----------
notebook = StudyNotebook()
notebook.save_record(topic, analysis, practice)  -> str  (record_id)
notebook.load_records()                           -> list[dict]
notebook.delete_record(record_id)                -> bool
notebook.generate_statistics()                    -> dict
notebook.generate_revision_queue()               -> list[dict]
notebook.generate_study_plan()                    -> dict
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

# ── Storage path ──────────────────────────────────────────────────────────────
_DATA_DIR  = Path(__file__).parent.parent / "data"
_NOTEBOOK_PATH = _DATA_DIR / "study_notebook.json"

# Topics that haven't been revisited in this many days go into the revision queue
_REVISION_DAYS = 7

# Subjects in canonical order (mirrors SubjectTag)
_ALL_SUBJECTS = [
    "Polity", "Economy", "International Relations", "Environment",
    "Science & Tech", "Social Issues", "Governance",
    "History/Culture", "Geography", "Miscellaneous",
]

# Estimated minutes per difficulty level for study plan
_STUDY_TIME: dict[str, int] = {"Easy": 20, "Medium": 35, "Hard": 50}


class StudyNotebook:
    """
    Manages the local study notebook stored as a JSON array.

    Each record is a flat dict:
    {
        "record_id":       str   (UUID4),
        "saved_at":        str   (ISO-8601 datetime),
        "topic_title":     str,
        "subject":         str,
        "summary":         str,
        "revision_bullets":list[str],
        "keywords":        list[str],
        "related_pyqs":    list[dict],
        "generated_mains": str,
        "difficulty":      str   ("Easy" | "Medium" | "Hard"),
        "similar_themes":  list[str],
    }
    """

    # ── Persistence ────────────────────────────────────────────────────────────

    def load_records(self) -> list[dict]:
        """Return all saved records, newest first."""
        if not _NOTEBOOK_PATH.exists():
            return []
        try:
            raw = _NOTEBOOK_PATH.read_text(encoding="utf-8").strip()
            records = json.loads(raw) if raw else []
            # Sort newest first by saved_at
            records.sort(key=lambda r: r.get("saved_at", ""), reverse=True)
            return records
        except (json.JSONDecodeError, OSError):
            return []

    def save_record(
        self,
        topic: dict,
        analysis: dict,
        practice: dict,
    ) -> str:
        """
        Assemble and persist a study record from the three agent outputs.

        Returns the new record_id.
        """
        records = self.load_records()

        record_id = str(uuid.uuid4())[:8]   # short 8-char ID — readable in the UI

        record = {
            "record_id":       record_id,
            "saved_at":        datetime.now().isoformat(timespec="seconds"),
            "topic_title":     topic.get("topic_title", "Unknown Topic"),
            "subject":         topic.get("subject_tag", "Miscellaneous"),
            "summary":         analysis.get("concise_summary", ""),
            "revision_bullets":analysis.get("revision_bullets", []),
            "keywords":        analysis.get("keywords_to_remember", []),
            "related_pyqs":    practice.get("related_pyqs", []),
            "generated_mains": practice.get("generated_mains", ""),
            "difficulty":      practice.get("difficulty", "Medium"),
            "similar_themes":  practice.get("similar_themes", []),
        }

        # Prepend so newest is first in file too
        records.insert(0, record)
        self._write(records)
        return record_id

    def delete_record(self, record_id: str) -> bool:
        """Delete a record by its short ID. Returns True if found and deleted."""
        records = self.load_records()
        new_records = [r for r in records if r.get("record_id") != record_id]
        if len(new_records) == len(records):
            return False
        self._write(new_records)
        return True

    def _write(self, records: list[dict]) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _NOTEBOOK_PATH.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Analytics ──────────────────────────────────────────────────────────────

    def generate_statistics(self) -> dict:
        """
        Compute dashboard statistics from saved records.

        Returns
        -------
        {
            "total_topics":        int,
            "subject_counts":      dict[str, int],   # all subjects, 0 for unstudied
            "difficulty_counts":   dict[str, int],
            "average_difficulty":  str,
            "recent_topics":       list[dict],        # up to 5 most recent
            "weak_subjects":       list[str],         # least studied (non-zero floor)
            "study_streak_days":   int,               # consecutive days with at least 1 save
        }
        """
        records = self.load_records()

        if not records:
            return {
                "total_topics":       0,
                "subject_counts":     {s: 0 for s in _ALL_SUBJECTS},
                "difficulty_counts":  {"Easy": 0, "Medium": 0, "Hard": 0},
                "average_difficulty": "—",
                "recent_topics":      [],
                "weak_subjects":      [],
                "study_streak_days":  0,
            }

        subject_counts: Counter = Counter()
        difficulty_counts: Counter = Counter({"Easy": 0, "Medium": 0, "Hard": 0})

        for r in records:
            subject_counts[r.get("subject", "Miscellaneous")] += 1
            diff = r.get("difficulty", "Medium")
            if diff in difficulty_counts:
                difficulty_counts[diff] += 1

        # Fill zeros for subjects never studied
        full_subject_counts = {s: subject_counts.get(s, 0) for s in _ALL_SUBJECTS}

        # Average difficulty (weighted)
        total = len(records)
        score_sum = (
            difficulty_counts["Easy"] * 1
            + difficulty_counts["Medium"] * 2
            + difficulty_counts["Hard"] * 3
        )
        avg_score = score_sum / total if total else 2.0
        average_difficulty = (
            "Easy" if avg_score < 1.5 else "Hard" if avg_score >= 2.5 else "Medium"
        )

        # Weak subjects: studied at least once, sort by count ascending
        studied = {s: c for s, c in full_subject_counts.items() if c > 0}
        weak_subjects = sorted(studied, key=lambda s: studied[s])[:3]

        # Recent topics (up to 5)
        recent_topics = [
            {
                "topic_title": r.get("topic_title", "—"),
                "subject":     r.get("subject", "—"),
                "saved_at":    r.get("saved_at", "—"),
                "difficulty":  r.get("difficulty", "—"),
                "record_id":   r.get("record_id", ""),
            }
            for r in records[:5]
        ]

        # Study streak: count consecutive calendar days (from today backwards)
        saved_dates = sorted(
            {r.get("saved_at", "")[:10] for r in records if r.get("saved_at")},
            reverse=True,
        )
        streak = 0
        check_date = date.today()
        for d_str in saved_dates:
            try:
                d = date.fromisoformat(d_str)
            except ValueError:
                continue
            if d == check_date:
                streak += 1
                check_date -= timedelta(days=1)
            elif d < check_date:
                break   # gap found

        return {
            "total_topics":       total,
            "subject_counts":     full_subject_counts,
            "difficulty_counts":  dict(difficulty_counts),
            "average_difficulty": average_difficulty,
            "recent_topics":      recent_topics,
            "weak_subjects":      weak_subjects,
            "study_streak_days":  streak,
        }

    def generate_revision_queue(self) -> list[dict]:
        """
        Return topics not revisited in the last _REVISION_DAYS days.

        Deduplicates by topic_title — only the oldest unseen save is returned
        for each title so the queue doesn't have repeats.
        """
        records = self.load_records()
        cutoff = datetime.now() - timedelta(days=_REVISION_DAYS)

        seen_titles: set[str] = set()
        queue: list[dict] = []

        # records are newest-first; iterate to find oldest occurrence per title
        # We want topics whose MOST RECENT save is older than the cutoff.
        # Build a dict: title → most recent saved_at datetime
        latest: dict[str, datetime] = {}
        for r in records:
            title = r.get("topic_title", "")
            try:
                dt = datetime.fromisoformat(r.get("saved_at", ""))
            except ValueError:
                continue
            if title not in latest or dt > latest[title]:
                latest[title] = dt

        for title, dt in latest.items():
            if dt < cutoff:
                # Find the record for this title to get metadata
                record = next((r for r in records if r.get("topic_title") == title), None)
                if record:
                    days_ago = (datetime.now() - dt).days
                    queue.append({
                        "topic_title": title,
                        "subject":     record.get("subject", "—"),
                        "difficulty":  record.get("difficulty", "—"),
                        "last_studied": dt.strftime("%d %b %Y"),
                        "days_ago":    days_ago,
                        "record_id":   record.get("record_id", ""),
                    })

        # Sort most overdue first
        queue.sort(key=lambda x: x["days_ago"], reverse=True)
        return queue

    def generate_study_plan(self) -> dict:
        """
        Build a deterministic daily study plan from notebook history.

        Logic:
          • Priority topics   : last 2 saves (freshest material to consolidate)
          • Revision due      : up to 3 from the revision queue
          • Weak subjects     : 1–2 subjects with lowest study count
          • Suggested practice: recommend one Prelims + one Mains session
          • Estimated time    : sum of per-difficulty estimates

        Returns a plain dict suitable for direct Streamlit rendering.
        """
        records = self.load_records()
        stats   = self.generate_statistics()
        queue   = self.generate_revision_queue()

        if not records:
            return {
                "has_data":          False,
                "priority_topics":   [],
                "revision_due":      [],
                "weak_subjects":     [],
                "suggested_practice":[],
                "estimated_minutes": 0,
                "message":           "Save your first study session to get a personalised study plan.",
            }

        # Priority topics: most recent 2 unique topics
        seen: set[str] = set()
        priority_topics: list[dict] = []
        for r in records:
            title = r.get("topic_title", "")
            if title not in seen:
                seen.add(title)
                priority_topics.append({
                    "topic_title": title,
                    "subject":     r.get("subject", "—"),
                    "difficulty":  r.get("difficulty", "Medium"),
                    "action":      "Review analysis + redo Mains question from memory",
                })
            if len(priority_topics) == 2:
                break

        # Revision due: top 3 from queue
        revision_due = queue[:3]

        # Weak subjects: up to 2
        weak_subjects = stats.get("weak_subjects", [])[:2]

        # Suggested practice
        suggested_practice: list[str] = []
        if priority_topics:
            suggested_practice.append(
                f"Attempt one timed Mains answer on '{priority_topics[0]['topic_title']}' (25 min)"
            )
        if revision_due:
            suggested_practice.append(
                f"Revise '{revision_due[0]['topic_title']}' — focus on keywords and Prelims MCQs"
            )
        if weak_subjects:
            suggested_practice.append(
                f"Read one current-affairs article on {weak_subjects[0]} to strengthen weak subject"
            )
        if not suggested_practice:
            suggested_practice.append("Upload today's newspaper PDF and run a fresh analysis")

        # Estimated time
        minutes = 0
        for t in priority_topics:
            minutes += _STUDY_TIME.get(t.get("difficulty", "Medium"), 35)
        for t in revision_due:
            minutes += _STUDY_TIME.get(t.get("difficulty", "Medium"), 35) // 2  # revision = half time
        minutes += len(suggested_practice) * 10   # ~10 min per suggested task

        return {
            "has_data":           True,
            "priority_topics":    priority_topics,
            "revision_due":       revision_due,
            "weak_subjects":      weak_subjects,
            "suggested_practice": suggested_practice,
            "estimated_minutes":  minutes,
            "message":            "",
        }
