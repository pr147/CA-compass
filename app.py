"""
CA Compass — UPSC Current Affairs Copilot
v5: ADK multi-agent pipeline + MCP server + security guardrails
"""

import os
import tempfile
import uuid
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from src.pdf_parser import extract_pdf
from src.chunker import chunk_text
from src.agents.adk_pipeline import (
    run_relevance_pipeline,
    run_analysis_pipeline,
    run_exam_pipeline,
    get_session_state,
)
from src.services.study_notebook import StudyNotebook
from src.security import (
    SecurityViolation,
    validate_upload,
    validate_page_count,
    validate_upsc_relevance,
    sanitise_chunks,
    security_report,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CA Compass",
    page_icon="🧭",
    layout="centered",
)

notebook = StudyNotebook()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧭 CA Compass")
    has_key = bool(os.getenv("GEMINI_API_KEY"))
    mode_label = "🤖 ADK + Gemini" if has_key else "📐 Heuristic"
    st.caption(f"Mode: {mode_label}")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📄 Study Session", "📓 Study Dashboard"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        "**Step 1 →** Upload a PDF\n\n"
        "**Step 2 →** Select a topic → Analysis\n\n"
        "**Step 3 →** Generate Exam Practice\n\n"
        "**Step 4 →** Save to Notebook\n\n"
        "**Step 5 →** Track progress in Dashboard"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — STUDY SESSION
# ══════════════════════════════════════════════════════════════════════════════
if page == "📄 Study Session":

    st.title("📄 Study Session")
    st.markdown(
        "Upload a current-affairs PDF → extract **UPSC-relevant topics** → "
        "deep-dive **analysis** → **exam practice** → save to notebook."
    )
    st.divider()

    # ── Session state ──────────────────────────────────────────────────────────
    for key, default in [
        ("topics", []),
        ("analysis", None),
        ("exam_practice", None),
        ("selected_topic_index", 0),
        ("last_saved_id", None),
        ("adk_session_id", str(uuid.uuid4())),   # ADK session — persists across steps
        ("security_report", None),
        ("redaction_count", 0),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Step 1: PDF upload & topic extraction ──────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload a PDF (newspaper, magazine, or current-affairs compilation)",
        type=["pdf"],
        help=f"Max {50} MB · Max 300 pages · Selectable text required.",
    )

    analyze_btn = st.button(
        "🔍 Analyze PDF",
        disabled=(uploaded_file is None),
        type="primary",
    )

    if analyze_btn and uploaded_file is not None:
        # Reset state for new analysis; keep the ADK session ID so it accumulates
        st.session_state.topics = []
        st.session_state.analysis = None
        st.session_state.exam_practice = None
        st.session_state.last_saved_id = None
        st.session_state.security_report = None
        st.session_state.redaction_count = 0

        tmp_path = None
        try:
            file_bytes = uploaded_file.read()

            # ── Security: validate upload before touching it ───────────────────
            with st.status("🔒 Running security checks…", expanded=False) as sec_status:
                try:
                    validate_upload(file_bytes, uploaded_file.name)
                    sec_status.update(label="🔒 Upload validated", state="complete")
                except SecurityViolation as e:
                    sec_status.update(label=f"🚫 Upload rejected: {e}", state="error")
                    st.error(str(e))
                    st.stop()

            # Write to temp file for PyMuPDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            # ── Extract text ───────────────────────────────────────────────────
            with st.status("Extracting text from PDF…", expanded=False) as s:
                pdf_data = extract_pdf(tmp_path)
                s.update(label=f"✅ Extracted {pdf_data['page_count']} pages", state="complete")

            # ── Security: page count limit ─────────────────────────────────────
            try:
                validate_page_count(pdf_data["page_count"])
            except SecurityViolation as e:
                st.error(str(e))
                st.stop()

            if not pdf_data["full_text"].strip():
                st.warning("No text extracted — PDF may be scanned without OCR text.")
                st.stop()

            # ── Security: UPSC relevance gate ──────────────────────────────────
            try:
                validate_upsc_relevance(pdf_data["full_text"])
            except SecurityViolation as e:
                st.error(str(e))
                st.stop()

            # ── Chunk ──────────────────────────────────────────────────────────
            with st.status("Chunking text…", expanded=False) as s:
                chunks = chunk_text(pdf_data["full_text"])
                s.update(label=f"✅ Created {len(chunks)} chunks", state="complete")

            # ── Security: sanitise chunks (prompt injection) ───────────────────
            with st.status("🔒 Scanning for prompt injection…", expanded=False) as s:
                chunks, redaction_count = sanitise_chunks(chunks)
                st.session_state.redaction_count = redaction_count
                label = (
                    f"🔒 Clean ({len(chunks)} chunks)" if redaction_count == 0
                    else f"⚠️ {redaction_count} chunk(s) sanitised"
                )
                s.update(label=label, state="complete")

            # ── Store security report in session ───────────────────────────────
            st.session_state.security_report = security_report(
                file_bytes_len=len(file_bytes),
                page_count=pdf_data["page_count"],
                chunk_count=len(chunks),
                redaction_count=redaction_count,
            )

            # ── ADK: run RelevanceAgent via pipeline ───────────────────────────
            with st.status("🤖 ADK RelevanceAgent identifying topics…", expanded=True) as s:
                st.session_state.topics = run_relevance_pipeline(
                    chunks, st.session_state.adk_session_id
                )
                s.update(
                    label=f"✅ Found {len(st.session_state.topics)} relevant topics",
                    state="complete",
                )

        except SecurityViolation as e:
            st.error(f"Security check failed: {e}")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.caption("Check the PDF is not password-protected and contains selectable text.")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── Security report expander (always visible after an analysis) ────────────
    if st.session_state.security_report:
        rep = st.session_state.security_report
        with st.expander("🔒 Security & Validation Report", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("File Size", f"{rep['file_size_mb']} MB")
            c2.metric("Pages", rep["page_count"])
            c3.metric("Chunks", rep["chunks_processed"])
            st.markdown(
                f"- **Size limit:** {rep['size_limit_mb']} MB ✅\n"
                f"- **Page limit:** {rep['page_limit']} ✅\n"
                f"- **UPSC relevance gate:** passed ✅\n"
                f"- **Prompt injection scan:** {rep['injection_check']} ✅"
            )

    # ── Step 1 results ─────────────────────────────────────────────────────────
    topics = st.session_state.topics

    if topics:
        st.divider()
        st.subheader(f"Top {len(topics)} UPSC-Relevant Topics")

        for i, topic in enumerate(topics, 1):
            score = topic.get("relevance_score", 0)
            icon = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
            with st.expander(
                f"{i}. {topic.get('topic_title', 'Unknown')}  {icon} {score}/100",
                expanded=(i <= 2),
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Subject:** `{topic.get('subject_tag', '—')}`")
                    tags = topic.get("exam_tags", [])
                    st.markdown(f"**Exam:** {' · '.join(f'`{t}`' for t in tags) or '—'}")
                with c2:
                    gs = topic.get("gs_paper_tags", [])
                    st.markdown(f"**GS Papers:** {' · '.join(f'`{t}`' for t in gs) or '—'}")
                    st.markdown(f"**Score:** `{score}/100`")
                st.markdown(f"> {topic.get('why_relevant', '—')}")
                preview = topic.get("source_chunk_preview", "")
                if preview:
                    st.caption(preview)

        # ── Step 2: analysis ───────────────────────────────────────────────────
        st.divider()
        st.subheader("🔬 Deep-Dive Analysis")

        topic_titles = [t.get("topic_title", f"Topic {i+1}") for i, t in enumerate(topics)]
        selected_index = st.selectbox(
            "Choose a topic to analyse",
            options=range(len(topic_titles)),
            format_func=lambda i: topic_titles[i],
            index=st.session_state.selected_topic_index,
            key="topic_selector",
        )
        if selected_index != st.session_state.selected_topic_index:
            st.session_state.analysis = None
            st.session_state.exam_practice = None
            st.session_state.last_saved_id = None
        st.session_state.selected_topic_index = selected_index

        if st.button("📖 Deep-Dive Analysis", type="primary"):
            st.session_state.analysis = None
            st.session_state.exam_practice = None
            st.session_state.last_saved_id = None
            with st.status(
                f"🤖 ADK AnalysisAgent processing '{topics[selected_index].get('topic_title', '')}'…",
                expanded=True,
            ) as s:
                try:
                    st.session_state.analysis = run_analysis_pipeline(
                        topics[selected_index],
                        st.session_state.adk_session_id,
                    )
                    s.update(label="✅ Analysis complete", state="complete")
                except Exception as e:
                    s.update(label=f"❌ {e}", state="error")

    # ── Step 2 results ─────────────────────────────────────────────────────────
    analysis = st.session_state.analysis

    if analysis:
        st.divider()
        st.subheader(f"📚 {analysis.get('topic_title', 'Analysis')}")

        st.markdown("#### 🗒 Concise Summary")
        st.info(analysis.get("concise_summary", "—"))

        ca, cb = st.columns(2)
        with ca:
            st.markdown("#### 🏛 Background Context")
            st.markdown(analysis.get("background_context", "—"))
        with cb:
            st.markdown("#### 🎯 Why It Matters for UPSC")
            st.markdown(analysis.get("why_it_matters_for_upsc", "—"))

        st.divider()
        cp, cm = st.columns(2)
        with cp:
            st.markdown("#### 📌 Prelims Angle")
            st.markdown(analysis.get("prelims_angle", "—"))
        with cm:
            st.markdown("#### ✍️ Mains Angle")
            st.markdown(analysis.get("mains_angle", "—"))

        st.divider()
        st.markdown("#### 🔁 Revision Bullets")
        for b in analysis.get("revision_bullets", []):
            st.markdown(f"- {b}")

        st.markdown("#### 🔑 Keywords to Remember")
        st.markdown("  ".join(f"`{k}`" for k in analysis.get("keywords_to_remember", [])))

        # ── ADK session state peek (observability) ─────────────────────────────
        with st.expander("🤖 ADK Session State", expanded=False):
            state = get_session_state(st.session_state.adk_session_id)
            st.caption(
                f"Session ID: `{st.session_state.adk_session_id}` · "
                f"Keys: {list(state.keys())}"
            )

        # ── Step 3: exam practice ──────────────────────────────────────────────
        st.divider()
        st.subheader("🎓 Exam Practice")

        if st.button("📝 Generate Exam Practice", type="primary"):
            st.session_state.exam_practice = None
            st.session_state.last_saved_id = None
            selected_topic = topics[st.session_state.selected_topic_index]
            with st.status("🤖 ADK ExamCoachAgent generating practice…", expanded=True) as s:
                try:
                    st.session_state.exam_practice = run_exam_pipeline(
                        selected_topic,
                        analysis,
                        st.session_state.adk_session_id,
                    )
                    s.update(label="✅ Exam practice ready", state="complete")
                except Exception as e:
                    s.update(label=f"❌ {e}", state="error")

    # ── Step 3 results ─────────────────────────────────────────────────────────
    practice = st.session_state.exam_practice

    if practice:
        st.divider()

        diff = practice.get("difficulty", "Medium")
        diff_icon = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(diff, "⚪")
        cd, ct = st.columns([1, 3])
        with cd:
            st.markdown(f"**Difficulty:** {diff_icon} `{diff}`")
        with ct:
            themes = practice.get("similar_themes", [])
            st.markdown("**Similar Themes:**  " + "  ".join(f"`{t}`" for t in themes))

        st.divider()
        st.markdown("#### 📜 Related Previous Year Questions")
        pyqs = practice.get("related_pyqs", [])
        if pyqs:
            for pyq in pyqs:
                with st.expander(
                    f"[{pyq.get('year')} · {pyq.get('exam_stage')} · {pyq.get('subject')}]"
                    f" — `{pyq.get('difficulty', 'Medium')}`"
                ):
                    st.markdown(pyq.get("question", "—"))
        else:
            st.caption("No closely matching PYQs found in the dataset.")

        st.divider()
        st.markdown("#### 📋 Generated Prelims MCQs")
        for idx, mcq in enumerate(practice.get("generated_prelims", []), 1):
            with st.expander(
                f"MCQ {idx}: {mcq.get('question', '')[:80]}…", expanded=(idx == 1)
            ):
                st.markdown(f"**Q{idx}.** {mcq.get('question', '—')}")
                st.markdown("")
                for opt in mcq.get("options", []):
                    st.markdown(f"&nbsp;&nbsp;{opt}")
                st.markdown("")
                st.success(f"✅ Correct Answer: **{mcq.get('correct_answer', '—')}**")
                if mcq.get("explanation"):
                    st.caption(f"💡 {mcq['explanation']}")

        st.divider()
        st.markdown("#### ✍️ Generated Mains Question")
        st.warning(practice.get("generated_mains", "—"))

        st.divider()
        st.markdown("#### 📖 Important Concepts to Revise")
        for rt in practice.get("revision_topics", []):
            st.markdown(f"- {rt}")

        # ── Step 4: Save to Notebook ───────────────────────────────────────────
        st.divider()
        st.subheader("💾 Save to Study Notebook")

        if st.session_state.last_saved_id:
            st.success(
                f"✅ Saved (ID: `{st.session_state.last_saved_id}`). "
                "Switch to **Study Dashboard** to view your progress."
            )
            if st.button("💾 Save Again"):
                st.session_state.last_saved_id = None
                st.rerun()
        else:
            if st.button("💾 Save to Study Notebook", type="primary"):
                try:
                    selected_topic = topics[st.session_state.selected_topic_index]
                    record_id = notebook.save_record(selected_topic, analysis, practice)
                    st.session_state.last_saved_id = record_id
                    st.success(f"✅ Saved! Record ID: `{record_id}`.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not save: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — STUDY DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("📓 Study Dashboard")
    st.markdown("Track your UPSC preparation progress across all saved study sessions.")
    st.divider()

    records = notebook.load_records()
    stats   = notebook.generate_statistics()
    queue   = notebook.generate_revision_queue()
    plan    = notebook.generate_study_plan()

    if not records:
        st.info(
            "Your notebook is empty. Complete a study session and click "
            "**Save to Study Notebook** to start tracking your progress."
        )
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Topics Studied", stats["total_topics"])
    m2.metric("Study Streak", f"{stats['study_streak_days']} day(s)")
    m3.metric("Avg Difficulty", stats["average_difficulty"])
    m4.metric("Due for Revision", len(queue))

    st.divider()
    st.subheader("📊 Subject Distribution")
    studied_subjects = {s: c for s, c in stats["subject_counts"].items() if c > 0}
    if studied_subjects:
        st.bar_chart(studied_subjects, use_container_width=True)
    else:
        st.caption("No subjects studied yet.")

    diff_counts = stats["difficulty_counts"]
    if any(diff_counts.values()):
        st.subheader("🎯 Difficulty Breakdown")
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("🟢 Easy",   diff_counts.get("Easy", 0))
        dc2.metric("🟡 Medium", diff_counts.get("Medium", 0))
        dc3.metric("🔴 Hard",   diff_counts.get("Hard", 0))

    st.divider()
    st.subheader("🕐 Recent Topics")
    for r in stats["recent_topics"]:
        diff_icon = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(r["difficulty"], "⚪")
        st.markdown(
            f"{diff_icon} **{r['topic_title']}** &nbsp;·&nbsp; "
            f"`{r['subject']}` &nbsp;·&nbsp; {r['saved_at'][:10]}"
        )

    st.divider()
    st.subheader("⚠️ Weak Subjects")
    weak = stats["weak_subjects"]
    if weak:
        st.markdown("These subjects have the fewest saved topics — prioritise them:")
        for s in weak:
            st.markdown(f"- **{s}** — {stats['subject_counts'].get(s, 0)} topic(s) saved")
    else:
        st.caption("Well-balanced! All subjects have at least one saved topic.")

    st.divider()
    st.subheader("🔁 Revision Queue")
    st.caption("Topics not revisited in the last 7 days.")
    if queue:
        for item in queue:
            diff_icon = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(item["difficulty"], "⚪")
            st.markdown(
                f"{diff_icon} **{item['topic_title']}** &nbsp;·&nbsp; "
                f"`{item['subject']}` &nbsp;·&nbsp; "
                f"Last studied: {item['last_studied']} ({item['days_ago']} days ago)"
            )
    else:
        st.success("✅ All topics are up to date — nothing due for revision.")

    st.divider()
    st.subheader("📅 Today's Study Plan")
    if not plan["has_data"]:
        st.info(plan["message"])
    else:
        st.markdown(f"**Estimated study time: ~{plan['estimated_minutes']} minutes**")
        if plan["priority_topics"]:
            st.markdown("**🔥 Priority Topics**")
            for t in plan["priority_topics"]:
                diff_icon = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(t["difficulty"], "⚪")
                st.markdown(f"- {diff_icon} **{t['topic_title']}** `{t['subject']}` — {t['action']}")
        if plan["revision_due"]:
            st.markdown("**🔁 Due for Revision**")
            for t in plan["revision_due"]:
                st.markdown(
                    f"- **{t['topic_title']}** — last studied {t['last_studied']} "
                    f"({t['days_ago']} days ago)"
                )
        if plan["weak_subjects"]:
            st.markdown("**📚 Weak Subjects to Focus On**")
            for s in plan["weak_subjects"]:
                st.markdown(f"- {s}")
        if plan["suggested_practice"]:
            st.markdown("**✏️ Suggested Practice**")
            for p in plan["suggested_practice"]:
                st.markdown(f"- {p}")

    st.divider()
    st.subheader("📒 All Saved Records")
    st.caption(f"{len(records)} record(s) saved locally.")

    for rec in records:
        diff_icon = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(rec.get("difficulty"), "⚪")
        with st.expander(
            f"{diff_icon} {rec.get('topic_title', '—')}  ·  "
            f"`{rec.get('subject', '—')}`  ·  {rec.get('saved_at', '')[:10]}"
        ):
            if rec.get("summary"):
                st.markdown(f"**Summary:** {rec['summary']}")
            bullets = rec.get("revision_bullets", [])
            if bullets:
                st.markdown("**Revision Bullets:**")
                for b in bullets:
                    st.markdown(f"- {b}")
            keywords = rec.get("keywords", [])
            if keywords:
                st.markdown("**Keywords:** " + "  ".join(f"`{k}`" for k in keywords))
            mains = rec.get("generated_mains", "")
            if mains:
                st.markdown("**Mains Question:**")
                st.warning(mains)
            themes = rec.get("similar_themes", [])
            if themes:
                st.markdown("**Similar Themes:** " + "  ".join(f"`{t}`" for t in themes))
            if st.button("🗑 Delete this record", key=f"del_{rec.get('record_id')}"):
                if notebook.delete_record(rec.get("record_id", "")):
                    st.success("Record deleted.")
                    st.rerun()
                else:
                    st.error("Could not delete record.")
