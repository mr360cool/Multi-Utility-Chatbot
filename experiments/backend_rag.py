from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEndpointEmbeddings,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------

   # 1. HuggingFace Models
# model = HuggingFaceEndpoint(
#     repo_id="openai/gpt-oss-20b",
#     task="text-generation",
#     provider="auto",
# )
# llm = ChatHuggingFace(llm=model)

# embeddings = HuggingFaceEndpointEmbeddings(
#     model="sentence-transformers/all-MiniLM-L6-v2",
#     task="feature-extraction",
# )


   # 2. OpenAi Models
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -------------------
# 2. PDF retriever store (per thread with disk persistence)
# -------------------
VECTORSTORE_DIR = "vectorstores"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available, loading from disk if needed."""
    if not thread_id:
        return None
    tid = str(thread_id)
    if tid in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[tid]

    # Check if persisted on disk
    thread_vs_dir = os.path.join(VECTORSTORE_DIR, tid)
    if os.path.exists(thread_vs_dir) and os.path.exists(
        os.path.join(thread_vs_dir, "index.faiss")
    ):
        try:
            vector_store = FAISS.load_local(
                thread_vs_dir, embeddings, allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            _THREAD_RETRIEVERS[tid] = retriever

            meta_path = os.path.join(thread_vs_dir, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    _THREAD_METADATA[tid] = json.load(f)
            return retriever
        except Exception as e:
            print(f"Error loading vectorstore for thread {tid}: {e}")
            return None
    return None


def ingest_pdf(
    file_bytes: bytes, thread_id: str, filename: Optional[str] = None
) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        tid = str(thread_id)
        _THREAD_RETRIEVERS[tid] = retriever
        meta = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        _THREAD_METADATA[tid] = meta

        # Persist vectorstore and metadata to disk so past conversations can reload it
        try:
            thread_vs_dir = os.path.join(VECTORSTORE_DIR, tid)
            os.makedirs(thread_vs_dir, exist_ok=True)
            vector_store.save_local(thread_vs_dir)
            with open(
                os.path.join(thread_vs_dir, "metadata.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(meta, f)
        except Exception as e:
            print(f"Warning: could not persist vectorstore to disk: {e}")

        return meta
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {
            "error": "Stock price lookup failed: ALPHA_VANTAGE_API_KEY environment variable is not configured."
        }
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    r = requests.get(url, timeout=10)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": thread_document_metadata(str(thread_id)).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)


# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful and intelligent AI assistant. For questions about the uploaded PDF or document, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. When answering using document context, cite page numbers and source details whenever available. "
            "You can also use web search, stock price lookup, and the "
            "calculator tool when helpful. Format your answers neatly using Markdown (bullet points, bold text, code blocks, or tables as appropriate). "
            "Mathematical Formatting: When writing math, formulas, or equations, ALWAYS format them using standard LaTeX with dollar signs: "
            "use `$$ ... $$` on separate lines for display equations (e.g. `$$\\text{CAGR} = \\left(\\frac{EV}{BV}\\right)^{1/n} - 1$$`), and `$ ... $` for inline variables (e.g. `$n$`, `$2.5^{1/4}$`). "
            "NEVER use square brackets like `\\[ \\]` or `[ \\text{...} ]` or parentheses like `\\( \\)` for formulas. "
            "If no document is available and the user asks about one, politely ask them to upload a PDF in the sidebar."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads_info() -> list[dict]:
    """
    Retrieve all conversation threads with ChatGPT-style titles
    (derived from the first user question), ordered from newest to oldest.
    """
    threads_dict = {}
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
        if not thread_id or thread_id in threads_dict:
            continue

        messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
        title = "New Conversation"
        for msg in messages:
            msg_type = getattr(msg, "type", "")
            if msg_type == "human" or msg.__class__.__name__ == "HumanMessage":
                content = str(msg.content).strip().replace("\n", " ")
                if content:
                    title = content[:38] + "..." if len(content) > 38 else content
                    break

        threads_dict[thread_id] = {
            "thread_id": str(thread_id),
            "title": title,
            "timestamp": checkpoint.checkpoint.get("ts"),
        }

    return list(threads_dict.values())


def delete_thread(thread_id: str) -> bool:
    """Delete a conversation thread, its checkpoints, and any saved vectorstore."""
    try:
        tid = str(thread_id)
        with sqlite3.connect("chatbot.db", check_same_thread=False) as db_conn:
            cur = db_conn.cursor()
            cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (tid,))
            cur.execute("DELETE FROM writes WHERE thread_id = ?", (tid,))
            db_conn.commit()

        _THREAD_RETRIEVERS.pop(tid, None)
        _THREAD_METADATA.pop(tid, None)

        vs_path = os.path.join(VECTORSTORE_DIR, tid)
        if os.path.exists(vs_path):
            shutil.rmtree(vs_path, ignore_errors=True)
        return True
    except Exception as e:
        print(f"Error deleting thread {thread_id}: {e}")
        return False


def thread_has_document(thread_id: str) -> bool:
    tid = str(thread_id)
    if tid in _THREAD_RETRIEVERS:
        return True
    thread_vs_dir = os.path.join(VECTORSTORE_DIR, tid)
    return os.path.exists(os.path.join(thread_vs_dir, "index.faiss"))


def thread_document_metadata(thread_id: str) -> dict:
    tid = str(thread_id)
    if tid in _THREAD_METADATA:
        return _THREAD_METADATA[tid]
    meta_path = os.path.join(VECTORSTORE_DIR, tid, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                _THREAD_METADATA[tid] = data
                return data
        except Exception:
            pass
    return {}
