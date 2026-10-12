import hashlib
import html
import os
import re
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv()

for _secret_name in (
    "OPENAI_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
):
    if _secret_name in st.secrets and not os.environ.get(_secret_name):
        os.environ[_secret_name] = str(st.secrets[_secret_name])

try:
    from langsmith import Client as LangSmithClient

    ls_client = LangSmithClient()
except Exception:
    ls_client = None

from final_backend import (
    chatbot,
    delete_thread,
    detach_document_from_thread,
    get_last_active_thread_id,
    get_thread_documents,
    ingest_pdf,
    remove_document_from_thread,
    retrieve_all_threads_info,
    set_last_active_thread_id,
    thread_document_metadata,
    thread_has_document,
)


def format_message_content(content: str) -> str:
    """
    Format LaTeX and mathematical expressions into Streamlit KaTeX-compatible
    delimiters ($$ for display equations and $ for inline variables/expressions).
    """
    if not isinstance(content, str):
        return str(content)

    # 1. Convert \[ ... \] into $$ ... $$
    content = re.sub(
        r"\\\[(.*?)\\\]",
        lambda m: f"\n\n$$\n{m.group(1).replace('%', r'\%').strip()}\n$$\n\n",
        content,
        flags=re.DOTALL,
    )

    # 2. Convert \( ... \) into $ ... $
    content = re.sub(
        r"\\\((.*?)\\\)",
        lambda m: f" ${m.group(1).replace('%', r'\%').strip()}$ ",
        content,
        flags=re.DOTALL,
    )

    # 3. Handle bracket equations like [ \text{...} ] or [ ... = ... ] containing LaTeX commands
    content = re.sub(
        r"(?<!\w)\[\s*(\\text\{.*?\}.*?|.*?\\frac\{.*?|.*?\\left\(.*?|.*?\\cdot.*?)\s*\]",
        lambda m: f"\n\n$$\n{m.group(1).replace('%', r'\%').strip()}\n$$\n\n",
        content,
        flags=re.DOTALL,
    )

    # 4. Handle isolated math expressions in parentheses like ( 2.5^{(1/4)} ) or ( n )
    content = re.sub(
        r"\(\s*([a-zA-Z]|\d+(?:\.\d+)?\^\{?[^}\n]+\}?)\s*\)",
        lambda m: f" ${m.group(1).strip()}$ ",
        content,
    )

    return content


# =========================== Page Config & Styling ===========================
st.set_page_config(
    page_title="Multi-Utility RAG & Agent Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .chat-card {
        background-color: #1E232A;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
        border: 1px solid #2E3440;
    }
    .badge-tool {
        background-color: #2e3846;
        color: #79c0ff;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
        display: inline-block;
        margin-right: 4px;
    }
    .chat-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 16px;
        background-color: #21262d;
        color: #c9d1d9;
        font-size: 13px;
        margin: 4px;
        cursor: pointer;
        border: 1px solid #30363d;
    }
    div[data-testid="stSidebarHeader"] {
        padding-bottom: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================== Utilities ===========================
def generate_thread_id() -> str:
    return str(uuid.uuid4())


def reset_chat():
    new_tid = generate_thread_id()
    st.session_state["thread_id"] = new_tid
    st.session_state["message_history"] = []
    st.session_state.pop("prefill_prompt", None)
    st.query_params["thread_id"] = new_tid
    set_last_active_thread_id(new_tid)


def load_conversation(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================

_url_tid = st.query_params.get("thread_id", None)
if not _url_tid:
    _url_tid = get_last_active_thread_id()

if _url_tid:
    st.session_state["thread_id"] = _url_tid
    st.query_params["thread_id"] = _url_tid
    set_last_active_thread_id(_url_tid)
    if "message_history" not in st.session_state:
        raw = chatbot.get_state(
            config={"configurable": {"thread_id": _url_tid}}
        ).values.get("messages", [])
        from langchain_core.messages import HumanMessage as HM, AIMessage as AM

        st.session_state["message_history"] = [
            (
                {"role": "user", "content": m.content}
                if isinstance(m, HM)
                else {"role": "assistant", "content": m.content}
            )
            for m in raw
            if (isinstance(m, HM) or (isinstance(m, AM) and m.content))
        ]
else:
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = generate_thread_id()
    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []
    st.query_params["thread_id"] = st.session_state["thread_id"]
    set_last_active_thread_id(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
doc_meta = thread_document_metadata(thread_key)


_is_fresh_thread = not bool(
    chatbot.get_state(config={"configurable": {"thread_id": thread_key}}).values.get(
        "messages", []
    )
)

selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("🤖 Multi-Utility Chatbot")
st.sidebar.caption("LangGraph • RAG • Web Search • Live Events • Calculator")

if st.sidebar.button("➕ New Chat", use_container_width=True, type="primary"):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# ================= PDF Document Ingestion =================
st.sidebar.subheader("📄 Document Ingestion (RAG)")

active_docs = get_thread_documents(thread_key)
active_by_name = {d.get("filename"): d.get("file_id") for d in active_docs}
active_by_hash = {d.get("file_hash"): d.get("file_id") for d in active_docs}

# ── Upload widget state ───────────────────────────────────────────────────────
upload_slot_key = f"upload_slot_{thread_key}"
if upload_slot_key not in st.session_state:
    st.session_state[upload_slot_key] = 0
show_uploader_key = f"show_uploader_{thread_key}"
if show_uploader_key not in st.session_state:
    st.session_state[show_uploader_key] = False


def _open_document_uploader(session_key: str) -> None:
    st.session_state[session_key] = True


st.sidebar.markdown(
    """
    <style>
    section[data-testid="stSidebar"] button[title^="Delete document"] {
        border-radius: 50% !important;
        min-height: 28px !important;
        width: 28px !important;
        padding: 0 !important;
        background: rgba(255,255,255,0.18) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: #f0f0f2 !important;
        font-size: 16px !important;
        line-height: 1 !important;
    }
    section[data-testid="stSidebar"] button[title^="Delete document"] p {
        display: block !important;
        visibility: visible !important;
        color: #f0f0f2 !important;
        font-size: 16px !important;
        line-height: 1 !important;
        margin: 0 !important;
    }
    section[data-testid="stSidebar"] button[title^="Delete document"]:hover {
        background: rgba(248,81,73,0.85) !important;
        border-color: rgba(248,81,73,0.95) !important;
        color: #fff !important;
    }
    section[data-testid="stSidebar"] button[title^="Delete document"]:hover p {
        color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ingestion_box = st.sidebar.container(border=True)
with ingestion_box:
    for doc in active_docs:
        doc_name = doc.get("filename", "Unknown")
        doc_id = str(doc.get("file_id") or doc_name)
        size_bytes = doc.get("file_size_bytes", 0)
        file_size = (
            f"{size_bytes / 1_048_576:.1f} MB"
            if size_bytes >= 1_048_576
            else (
                f"{size_bytes / 1024:.1f} KB"
                if size_bytes >= 1024
                else (
                    f"{size_bytes} B"
                    if size_bytes
                    else f"{doc.get('documents', 0)} pages"
                )
            )
        )
        display_name = (
            doc_name if len(doc_name) <= 24 else f"{doc_name[:12]}…{doc_name[-9:]}"
        )

        file_icon, file_details, remove_column = ingestion_box.columns(
            [0.18, 0.56, 0.26], vertical_alignment="center"
        )
        with file_icon:
            st.markdown(
                "<div style='width:42px;height:42px;background:#fff;border-radius:9px;"
                "display:flex;align-items:center;justify-content:center;font-size:22px;'>"
                "&#128196;</div>",
                unsafe_allow_html=True,
            )
        with file_details:
            st.markdown(
                f"<div style='font-size:13px;font-weight:600;white-space:nowrap;"
                f"overflow:hidden;text-overflow:ellipsis;'>"
                f"{html.escape(display_name)}</div>"
                f"<div style='font-size:11px;color:rgba(250,250,250,0.55);'>"
                f"{html.escape(file_size)}</div>",
                unsafe_allow_html=True,
            )
        with remove_column:
            if st.button(
                "Del",
                key=f"remove_document_{thread_key}_{doc_id}",
                help=f"Delete document {doc_name}",
                type="primary",
                use_container_width=True,
            ):
                try:
                    removed = remove_document_from_thread(thread_key, doc_id)
                except Exception as exc:
                    st.error(f"Could not remove {doc_name}: {exc}")
                else:
                    if removed:
                        st.session_state[upload_slot_key] += 1
                        if len(active_docs) == 1:
                            st.session_state[show_uploader_key] = False
                        st.rerun()
                    st.error(f"Could not remove {doc_name}. Please try again.")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    if active_docs and not st.session_state[show_uploader_key]:
        st.button(
            "＋",
            key=f"add_pdf_{thread_key}",
            help="Upload another PDF",
            use_container_width=True,
            on_click=_open_document_uploader,
            args=(show_uploader_key,),
        )

    if not active_docs or st.session_state[show_uploader_key]:
        uploaded_pdf = st.file_uploader(
            "Upload a PDF to this chat",
            type=["pdf"],
            key=f"uploader_{thread_key}_{st.session_state[upload_slot_key]}",
            label_visibility="collapsed",
        )
        if uploaded_pdf:
            uploaded_hash = hashlib.sha256(uploaded_pdf.getvalue()).hexdigest()
            if uploaded_hash in active_by_hash or uploaded_pdf.name in active_by_name:
                st.info(f"{uploaded_pdf.name} is already in this chat.")
                st.session_state[upload_slot_key] += 1
                st.rerun()

            with st.status(f"Indexing {uploaded_pdf.name}…", expanded=True) as status:
                try:
                    ingest_pdf(
                        uploaded_pdf.getvalue(),
                        thread_id=thread_key,
                        filename=uploaded_pdf.name,
                    )
                except Exception as e:
                    st.error(f"Failed to ingest {uploaded_pdf.name}: {e}")
                    status.update(
                        label="Document indexing failed", state="error", expanded=True
                    )
                else:
                    status.update(
                        label=f"✅ Indexed {uploaded_pdf.name}",
                        state="complete",
                        expanded=False,
                    )
                    st.session_state[show_uploader_key] = False
                    st.session_state[upload_slot_key] += 1
                    st.rerun()


st.sidebar.divider()

# ================= Conversation Threads Management =================
st.sidebar.subheader("💬 Past Conversations")

threads_info = retrieve_all_threads_info()

search_query = st.sidebar.text_input(
    "Search chats",
    placeholder="🔍 Filter conversations...",
    label_visibility="collapsed",
)

filtered_threads = [
    t
    for t in threads_info
    if not search_query or search_query.lower() in t.get("title", "").lower()
]

if not filtered_threads and not threads_info:
    st.sidebar.caption("No conversations yet. Start typing!")
elif not filtered_threads:
    st.sidebar.caption("No matching conversations found.")
else:
    for t in filtered_threads:
        tid = t["thread_id"]
        title = t["title"]
        is_active = tid == thread_key

        display_title = f"▶ {title}" if is_active else f"💬 {title}"

        col1, col2 = st.sidebar.columns([0.82, 0.18])
        with col1:
            if st.button(
                display_title,
                key=f"thread_btn_{tid}",
                use_container_width=True,
                help=f"Thread: {tid}",
            ):
                selected_thread = tid
        with col2:
            if st.button("🗑️", key=f"del_btn_{tid}", help="Delete this chat"):
                delete_thread(tid)
                if tid == thread_key:
                    reset_chat()
                st.rerun()

# ============================ Main Layout ========================
st.title("Multi-Utility Agentic Chatbot")
st.caption("Powered by hybrid tools (Web Search, Live Events, Precision Calculator), ")

# Quick-Prompt Suggestions
if not st.session_state["message_history"]:
    st.markdown("### 💡 Quick Actions")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button(
            "📄 What is the uploaded document about? Summarize key insights.",
            use_container_width=True,
        ):
            st.session_state["prefill_prompt"] = (
                "What is the uploaded document about? Summarize the key insights."
            )
            st.rerun()
        if st.button(
            "📈 Check latest stock price & metrics for NVDA and AAPL",
            use_container_width=True,
        ):
            st.session_state["prefill_prompt"] = (
                "What is the latest stock price and quote for NVDA and AAPL?"
            )
            st.rerun()
    with p_col2:
        if st.button(
            "🔍 Search DuckDuckGo for recent trending news",
            use_container_width=True,
        ):
            st.session_state["prefill_prompt"] = (
                "Search DuckDuckGo for the top 5 news of India today"
            )
            st.rerun()
        if st.button(
            "🧮 Compute Compound Annual Growth Rate (CAGR) for 10k to 25k over 4 years",
            use_container_width=True,
        ):
            st.session_state["prefill_prompt"] = (
                "Use the calculator to compute the growth factor and tell me the CAGR if an investment grew from 10000 to 25000 over 4 years."
            )
            st.rerun()

# Render chat history
for idx, message in enumerate(st.session_state["message_history"]):
    with st.chat_message(message["role"]):
        st.markdown(format_message_content(message["content"]))
        if message["role"] == "assistant":
            fb = st.feedback("thumbs", key=f"feedback_{thread_key}_{idx}")
            if fb is not None and not message.get("rated"):
                message["rated"] = True
                if ls_client:
                    try:
                        ls_client.create_feedback(
                            key="user_rating",
                            score=1.0 if fb == 1 else 0.0,
                            comment=(
                                "User approved response"
                                if fb == 1
                                else "User flagged response"
                            ),
                        )
                    except Exception:
                        pass

# Handle quick-prompt prefill or user input
prefill = st.session_state.pop("prefill_prompt", None)
user_input = st.chat_input(
    "Ask a question, analyze documents, or request tool execution..."
)
prompt_to_send = prefill or user_input

if prompt_to_send:
    st.session_state["message_history"].append(
        {"role": "user", "content": prompt_to_send}
    )
    with st.chat_message("user"):
        st.markdown(format_message_content(prompt_to_send))

    # LangSmith tracing and thread configuration
    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key, "source": "streamlit_ui"},
        "run_name": "agent_chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=prompt_to_send)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Running tool `{tool_name}`…", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Running tool `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool execution completed", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    active_docs = get_thread_documents(thread_key)
    if active_docs:
        doc_names = ", ".join(f"`{d.get('filename')}`" for d in active_docs)
        total_chunks = sum(d.get("chunks", 0) for d in active_docs)
        total_pages = sum(d.get("documents", 0) for d in active_docs)
        st.caption(
            f"📄 Active Source(s): {doc_names} • "
            f"Total Chunks: {total_chunks} • Pages: {total_pages}"
        )
    st.rerun()


# Handle thread switching
if selected_thread and selected_thread != thread_key:
    st.session_state["thread_id"] = selected_thread
    st.query_params["thread_id"] = selected_thread
    set_last_active_thread_id(selected_thread)
    raw_messages = load_conversation(selected_thread)

    cleaned_messages = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            cleaned_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            cleaned_messages.append({"role": "assistant", "content": msg.content})

    st.session_state["message_history"] = cleaned_messages
    st.rerun()
