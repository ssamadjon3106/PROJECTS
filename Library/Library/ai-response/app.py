from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st
from pypdf import PdfReader

from agent import (
    ROUTING_AUTO_REPLY,
    ROUTING_BANK_REVIEW,
    ROUTING_HUMAN_REVIEW_REQUIRED,
    process_legal_request,
)
from knowledge_base import LEGAL_DOCS_DIR, ensure_knowledge_base
from storage import (
    approve_interaction,
    create_interaction,
    fetch_recent_interactions,
    initialize_database,
    mark_auto_approved,
    reject_interaction,
    update_generated_response,
)


APP_TITLE = "AI Legal Agent Uzbekistan"
APP_SUBTITLE = "RAG + huquqiy tahlil + risk/compliance + SQLite saqlash"
DEFAULT_MODEL_NAME = "gpt-4o-mini"


@st.cache_resource(show_spinner=False)
def bootstrap_application() -> tuple[list[Path], str]:
    ensure_knowledge_base(LEGAL_DOCS_DIR)
    initialize_database()
    return list(Path(LEGAL_DOCS_DIR).glob("*.txt")), str(LEGAL_DOCS_DIR)


def extract_pdf_text(uploaded_file: Any) -> str:
    if uploaded_file is None:
        return ""

    try:
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text.strip())
        return "\n\n".join(pages).strip()
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        st.error(f"PDF o‘qishda xatolik: {exc}")
        return ""


def compose_input_text(user_text: str, pdf_text: str) -> str:
    user_text = (user_text or "").strip()
    pdf_text = (pdf_text or "").strip()

    if user_text and pdf_text:
        return f"{user_text}\n\n[PDF MATNI]\n{pdf_text}"
    if user_text:
        return user_text
    if pdf_text:
        return pdf_text
    return ""


def run_legal_pipeline(
    user_text: str,
    *,
    pdf_text: str = "",
    source_name: str = "text",
    source_filename: str = "",
    revision_hint: str | None = None,
) -> dict[str, Any]:
    full_input = compose_input_text(user_text, pdf_text)
    result = process_legal_request(full_input, revision_hint=revision_hint)
    interaction_id = create_interaction(
        user_input=full_input,
        extracted_text=pdf_text,
        classification=result["classification"],
        risk_level=result["risk_level"],
        risk_score=int(result["risk_score"]),
        compliance_score=int(result["compliance_score"]),
        confidence_score=int(result.get("confidence_score", 0)),
        confidence_level=str(result.get("confidence_level", "LOW")),
        routing_mode=str(result.get("routing_mode", ROUTING_AUTO_REPLY)),
        review_status="pending_review" if result.get("review_required") else "not_required",
        review_decision="",
        review_notes=result.get("review_reason", ""),
        generated_response=result["final_response"],
        warnings=result["warnings"],
        retrieved_chunks=result["retrieved_chunks"],
        source_name=source_name,
        source_filename=source_filename,
        approved=1 if result.get("routing_mode") == ROUTING_AUTO_REPLY else 0,
    )
    result["interaction_id"] = interaction_id
    result["full_input"] = full_input
    result["pdf_text"] = pdf_text
    return result


def initialize_session_state() -> None:
    defaults = {
        "current_result": None,
        "current_interaction_id": None,
        "editable_response": "",
        "pending_editable_response": "",
        "user_text": "",
        "pdf_text": "",
        "pdf_filename": "",
        "last_generated_input": "",
        "last_review_notes": "",
        "pending_review_notes": "",
        "force_rerun": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.info(
        "Bu tizim sintetik O‘zbekiston huquqiy korpusi asosida ishlaydi. "
        "Agar OpenAI API ulangan bo‘lsa, LLM javobni yanada boyitadi; bo‘lmasa lokal fallback ishlaydi."
    )


def render_sidebar() -> None:
    st.sidebar.header("So‘nggi yozuvlar")
    recent = fetch_recent_interactions(limit=10)
    if not recent:
        st.sidebar.caption("Hali yozuv yo‘q.")
        return

    for item in recent:
        with st.sidebar.expander(f"#{item['id']} | {item['routing_mode']} | {item['confidence_level']}"):
            st.write(f"**Yaratilgan:** {item['created_at']}")
            st.write(f"**Risk balli:** {item['risk_score']}")
            st.write(f"**Risk darajasi:** {item['risk_level']}")
            st.write(f"**Compliance balli:** {item['compliance_score']}")
            st.write(f"**Confidence balli:** {item.get('confidence_score', 0)}")
            st.write(f"**Routing:** {item.get('routing_mode', 'AUTO_REPLY')}")
            st.write(f"**Review status:** {item.get('review_status', 'not_required')}")
            st.write(f"**Tasdiqlangan:** {'Ha' if item['approved'] else 'Yo‘q'}")
            st.write(f"**Manba:** {item['source_name']}")
            if item.get("source_filename"):
                st.write(f"**Fayl:** {item['source_filename']}")
            st.code(item["user_input"][:500], language="text")


def render_classification(classification: dict[str, Any]) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Vakolat", classification.get("authority", "other"))
    col2.metric("Niyat", classification.get("intent", "—"))
    entities = classification.get("entities") or []
    col3.metric("Entitilar", str(len(entities)))

    if entities:
        st.write("**Topilgan entitilar:**", ", ".join(entities))
    issues = classification.get("issues") or []
    if issues:
        st.write("**Xavf omillari:**", ", ".join(issues))


def render_scores(result: dict[str, Any]) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk darajasi", result.get("risk_level", "LOW"))
    col1.metric("Risk balli", int(result.get("risk_score", 0)))
    col2.metric("Compliance balli", int(result.get("compliance_score", 0)))
    col2.metric("Confidence balli", int(result.get("confidence_score", 0)))
    col3.metric("Confidence darajasi", result.get("confidence_level", "LOW"))
    col3.metric("Routing", result.get("routing_mode", ROUTING_AUTO_REPLY))

    risk_level = result.get("risk_level", "LOW")
    routing_mode = result.get("routing_mode", ROUTING_AUTO_REPLY)

    if routing_mode == ROUTING_AUTO_REPLY:
        st.success("Javob avtomatik qaytarilishi mumkin.")
    elif routing_mode == ROUTING_BANK_REVIEW:
        st.warning("Confidence o‘rtacha: bank xodimi review qilishi kerak.")
    else:
        st.error("Javob majburiy inson ko‘rigini talab qiladi.")

    if risk_level == "HIGH":
        st.error("Yuqori risk aniqlandi. Minimal oshkor etish va aniq vakolat tekshiruvi zarur.")
    elif risk_level == "MEDIUM":
        st.warning("O‘rta risk aniqlandi. Cheklangan oshkor etish tavsiya etiladi.")
    else:
        st.success("Past risk aniqlandi.")


def render_retrieval(result: dict[str, Any]) -> None:
    retrieved_chunks = result.get("retrieved_chunks") or []
    st.subheader("RAG natijalari")
    if not retrieved_chunks:
        st.caption("Mos chunk topilmadi.")
        return

    for index, chunk in enumerate(retrieved_chunks, start=1):
        with st.expander(f"{index}. {chunk.get('citation', 'Citation yo‘q')} | score={chunk.get('score', 0.0):.3f}"):
            st.write(f"**Manba fayl:** {chunk.get('source_filename', '')}")
            st.write(f"**Title:** {chunk.get('source_title', '')}")
            st.code(chunk.get("text", ""), language="text")


def render_warnings(warnings: list[str]) -> None:
    if not warnings:
        st.success("Qo‘shimcha ogohlantirish yo‘q.")
        return

    st.subheader("Ogohlantirishlar")
    for warning in warnings:
        st.warning(warning)


def handle_generate() -> None:
    user_text = st.session_state.get("user_text", "").strip()
    pdf_text = st.session_state.get("pdf_text", "").strip()
    pdf_filename = st.session_state.get("pdf_filename", "")

    if not user_text and not pdf_text:
        st.error("Kamida matn yoki PDF yuklang.")
        return

    with st.spinner("Huquqiy javob tayyorlanmoqda..."):
        result = run_legal_pipeline(
            user_text=user_text,
            pdf_text=pdf_text,
            source_name="pdf" if pdf_text else "text",
            source_filename=pdf_filename if pdf_text else "",
        )

    st.session_state.current_result = result
    st.session_state.current_interaction_id = result["interaction_id"]
    st.session_state.pending_editable_response = result["final_response"]
    st.session_state.last_generated_input = result["full_input"]
    st.session_state.pending_review_notes = result.get("review_reason", "")
    st.session_state.force_rerun = True


def handle_regenerate(revision_hint: str | None = None) -> None:
    current = st.session_state.get("current_result")
    if not current:
        st.error("Avval bir marta generatsiya qiling.")
        return

    user_text = st.session_state.get("user_text", "").strip()
    pdf_text = st.session_state.get("pdf_text", "").strip()
    pdf_filename = st.session_state.get("pdf_filename", "")

    with st.spinner("Javob qayta generatsiya qilinmoqda..."):
        result = run_legal_pipeline(
            user_text=user_text,
            pdf_text=pdf_text,
            source_name="pdf" if pdf_text else "text",
            source_filename=pdf_filename if pdf_text else "",
            revision_hint=revision_hint,
        )

    st.session_state.current_result = result
    st.session_state.current_interaction_id = result["interaction_id"]
    st.session_state.pending_editable_response = result["final_response"]
    st.session_state.last_generated_input = result["full_input"]
    st.session_state.pending_review_notes = result.get("review_reason", "")
    st.session_state.force_rerun = True


def handle_accept() -> None:
    current_result = st.session_state.get("current_result")
    interaction_id = st.session_state.get("current_interaction_id")
    editable_response = st.session_state.get("editable_response", "").strip()

    if not current_result or not interaction_id:
        st.error("Tasdiqlash uchun avval generatsiya qiling.")
        return

    if not editable_response:
        st.error("Tasdiqlash uchun javob matni bo‘sh bo‘lishi mumkin emas.")
        return

    update_generated_response(
        interaction_id,
        generated_response=editable_response,
        final_response=editable_response,
    )

    routing_mode = current_result.get("routing_mode", ROUTING_AUTO_REPLY)
    if routing_mode == ROUTING_AUTO_REPLY:
        mark_auto_approved(interaction_id, final_response=editable_response)
    else:
        approve_interaction(interaction_id, final_response=editable_response)

    st.success(f"Javob tasdiqlandi va SQLite bazasiga saqlandi. ID: {interaction_id}")
    current_result["final_response"] = editable_response
    st.session_state.current_result = current_result


def handle_reject() -> None:
    current_result = st.session_state.get("current_result")
    interaction_id = st.session_state.get("current_interaction_id")
    editable_response = st.session_state.get("editable_response", "").strip()
    review_notes = st.session_state.get("last_review_notes", "").strip()

    if not current_result or not interaction_id:
        st.error("Rad etish uchun avval generatsiya qiling.")
        return

    if not review_notes:
        st.error("Rad etish uchun review notes kerak.")
        return

    reject_interaction(
        interaction_id,
        review_notes=review_notes,
        final_response=editable_response,
    )

    stronger_hint = (
        "Bank xodimi tomonidan rad etildi. Javobni kuchaytir: "
        "normal oshkor etish emas, faqat qonuniy minimal oshkor etish, "
        "vakolat, zarurat va cheklovlarni yanada aniq yoz."
    )

    with st.spinner("AI kuchliroq javob tayyorlamoqda..."):
        handle_regenerate(revision_hint=stronger_hint)

    st.warning("So‘rov rad etildi va AI kuchliroq javob qayta tayyorladi.")


def render_review_panel() -> None:
    current_result = st.session_state.get("current_result")
    if not current_result:
        return

    routing_mode = current_result.get("routing_mode", ROUTING_AUTO_REPLY)
    review_required = bool(current_result.get("review_required", False))

    st.subheader("Javobni ko‘rib chiqish")

    pending_response = st.session_state.get("pending_editable_response", "")
    if pending_response:
        st.session_state.editable_response = pending_response
        st.session_state.pending_editable_response = ""

    if "editable_response" not in st.session_state:
        st.session_state.editable_response = current_result.get("final_response", "")

    st.text_area(
        "Rasmiy javobni tahrirlash",
        height=380,
        key="editable_response",
    )

    if routing_mode == ROUTING_BANK_REVIEW or review_required:
        pending_review_notes = st.session_state.get("pending_review_notes", "")
        if pending_review_notes:
            st.session_state.last_review_notes = pending_review_notes
            st.session_state.pending_review_notes = ""

        if "last_review_notes" not in st.session_state:
            st.session_state.last_review_notes = current_result.get("review_reason", "")

        st.text_area(
            "Review notes / qaror izohi",
            height=120,
            key="last_review_notes",
            placeholder="Masalan: bank siri bo‘yicha asos yetarli emas, moliyaviy axborot qisqartirilsin.",
        )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Accept", use_container_width=True):
            handle_accept()
    with btn_col2:
        if st.button("Regenerate", use_container_width=True):
            handle_regenerate(
                revision_hint=(
                    "Qayta ishlashda javobni kuchaytir, rasmiy uslubni saqla, "
                    "vakolat va bank siri cheklovlarini aniqroq yoz."
                )
            )

    if routing_mode == ROUTING_BANK_REVIEW:
        btn_col3, btn_col4 = st.columns(2)
        with btn_col3:
            if st.button("Reject + Stronger AI", use_container_width=True):
                handle_reject()
        with btn_col4:
            st.caption("Reject bosilganda AI kuchliroq javobni avtomatik qayta yaratadi.")

    with st.expander("Xom natija"):
        st.write("**Interaction ID:**", current_result.get("interaction_id"))
        st.write("**LLM ishlatilgan:**", current_result.get("used_llm"))
        st.write("**Model:**", current_result.get("llm_model"))
        st.write("**Qayta ishlash bosqichlari:**", current_result.get("refinement_passes"))
        st.write("**Routing:**", current_result.get("routing_mode"))
        st.write("**Confidence:**", current_result.get("confidence_level"), current_result.get("confidence_score"))
        st.write("**Full input:**")
        st.code(current_result.get("full_input", ""), language="text")
        st.write("**Draft response:**")
        st.code(current_result.get("draft_response", ""), language="text")
        st.write("**Final response:**")
        st.code(current_result.get("final_response", ""), language="text")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")
    bootstrap_application()
    initialize_session_state()
    render_sidebar()
    render_header()

    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.subheader("Kirish")
        st.text_area(
            "Huquqiy so‘rov matni",
            value=st.session_state.user_text,
            height=180,
            placeholder="Masalan: Prokuratura bankdan mijozning hisobvaraqlari va tranzaksiyalari haqida ma’lumot talab qildi",
            key="user_text",
        )

        uploaded_pdf = st.file_uploader("PDF yuklash", type=["pdf"])
        if uploaded_pdf is not None:
            st.session_state.pdf_filename = uploaded_pdf.name
            st.session_state.pdf_text = extract_pdf_text(uploaded_pdf)
        else:
            st.session_state.pdf_filename = ""
            st.session_state.pdf_text = ""

        if st.session_state.pdf_text:
            with st.expander("Ajratilgan PDF matni"):
                st.write(st.session_state.pdf_text[:12000])

        generate_clicked = st.button("Generate Legal Response", type="primary", use_container_width=True)
        if generate_clicked:
            handle_generate()

    with right_col:
        st.subheader("Natijalar")
        current_result = st.session_state.get("current_result")

        if current_result:
            render_classification(current_result.get("classification", {}))
            render_scores(current_result)
            render_warnings(current_result.get("warnings", []))
            render_retrieval(current_result)
            render_review_panel()
        else:
            st.info("Javob generatsiya qilingach, bu yerda classification, risk, compliance va tahrirlash chiqadi.")

    if st.session_state.get("force_rerun"):
        st.session_state.force_rerun = False
        st.rerun()


if __name__ == "__main__":
    main()
