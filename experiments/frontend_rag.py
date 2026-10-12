import re
import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend_rag import (
    chatbot,
    delete_thread,
    ingest_pdf,
    retrieve_all_threads_info,
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
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []


def load_conversation(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

thread_key = str(st.session_state["thread_id"])
doc_meta = thread_document_metadata(thread_key)
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("🤖 Multi-Utility Agent")
st.sidebar.caption("LangGraph • RAG • Web Search • Finance • Code")

if st.sidebar.button("➕ New Chat", use_container_width=True, type="primary"):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# PDF Document Ingestion
st.sidebar.subheader("📄 Document Ingestion (RAG)")
if doc_meta and doc_meta.get("filename"):
    st.sidebar.success(
        f"**Active Doc:** `{doc_meta.get('filename')}`\n\n"
        f"📊 {doc_meta.get('chunks', 0)} chunks • {doc_meta.get('documents', 0)} pages"
    )
else:
    st.sidebar.info("Upload a PDF to enable Document RAG.")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload document for current chat", type=["pdf"], key="pdf_uploader"
)
if uploaded_pdf:
    if doc_meta and doc_meta.get("filename") == uploaded_pdf.name:
        st.sidebar.caption(f"`{uploaded_pdf.name}` is already indexed.")
    else:
        with st.sidebar.status("Indexing document with FAISS…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            doc_meta = summary
            status_box.update(
                label=f"✅ Indexed {summary.get('documents')} pages ({summary.get('chunks')} chunks)",
                state="complete",
                expanded=False,
            )
            st.rerun()

st.sidebar.divider()

# ChatGPT-Style Past Conversations
st.sidebar.subheader("💬 Past Conversations")

threads_info = retrieve_all_threads_info()

# Search filter for conversations
search_query = st.sidebar.text_input(
    "Search chats", placeholder="🔍 Filter conversations...", label_visibility="collapsed"
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
st.caption(
    "Powered by LangGraph state machine, hybrid tools (Web, Stock Quotes, Calculator), "
    "and persistent vectorstore RAG."
)

# Starter Quick-Prompt Suggestions (displayed when conversation is empty)
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
            "🔍 Search DuckDuckGo for top AI agent frameworks in 2026",
            use_container_width=True,
        ):
            st.session_state["prefill_prompt"] = (
                "Search DuckDuckGo for the top AI agent frameworks in 2026 and summarize findings."
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
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(format_message_content(message["content"]))

# Handle quick-prompt prefill or user input
prefill = st.session_state.pop("prefill_prompt", None)
user_input = st.chat_input("Ask a question, analyze documents, or request tool execution...")
prompt_to_send = prefill or user_input

if prompt_to_send:
    st.session_state["message_history"].append({"role": "user", "content": prompt_to_send})
    with st.chat_message("user"):
        st.markdown(format_message_content(prompt_to_send))

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
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

    if doc_meta and doc_meta.get("filename"):
        st.caption(
            f"📄 Context Source: `{doc_meta.get('filename')}` • "
            f"Chunks: {doc_meta.get('chunks')} • Pages: {doc_meta.get('documents')}"
        )
    st.rerun()

# Handle thread switching
if selected_thread and selected_thread != thread_key:
    st.session_state["thread_id"] = selected_thread
    raw_messages = load_conversation(selected_thread)

    cleaned_messages = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            cleaned_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            cleaned_messages.append({"role": "assistant", "content": msg.content})

    st.session_state["message_history"] = cleaned_messages
    st.rerun()
