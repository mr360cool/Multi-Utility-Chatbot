from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LCDoc
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()

# -------------------
# 1. LLM + Embeddings Configuration
# -------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -------------------
# 2. Vectorstore & Threat-Hardened Retrieval Storage
# -------------------
VECTORSTORE_DIR = os.path.abspath("vectorstores")
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
CHECKPOINT_DB_PATH = os.environ.get("CHATBOT_DB_PATH", "chatbot.db")

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _is_safe_thread_id(thread_id: str) -> bool:
    """Validate thread_id format to prevent directory traversal attacks."""
    if not thread_id or not isinstance(thread_id, str):
        return False
    return bool(re.match(r"^[a-zA-Z0-9_-]{1,128}$", thread_id))


def _get_thread_vs_dir(thread_id: str) -> Optional[str]:
    """Resolve and verify that the vectorstore directory stays strictly inside VECTORSTORE_DIR."""
    if not _is_safe_thread_id(thread_id):
        return None
    target_dir = os.path.abspath(os.path.join(VECTORSTORE_DIR, thread_id))
    # Path traversal check
    if os.path.commonpath([VECTORSTORE_DIR, target_dir]) != VECTORSTORE_DIR:
        return None
    return target_dir


def _get_retriever(thread_id: Optional[str]):
    """
    Fetch the retriever for a thread if available.
    Loads safely from the application-generated local directory only.
    """
    if not thread_id or not _is_safe_thread_id(thread_id):
        return None

    tid = str(thread_id)
    if tid in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[tid]

    thread_vs_dir = _get_thread_vs_dir(tid)
    if not thread_vs_dir or not os.path.isdir(thread_vs_dir):
        return None

    index_file = os.path.join(thread_vs_dir, "index.faiss")
    pkl_file = os.path.join(thread_vs_dir, "index.pkl")
    meta_file = os.path.join(thread_vs_dir, "metadata.json")

    if (
        os.path.exists(index_file)
        and os.path.exists(pkl_file)
        and os.path.exists(meta_file)
    ):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Metadata must match the thread_id
            if meta.get("thread_id") != tid:
                print(f"Security Warning: Thread ID mismatch in metadata for {tid}")
                return None

            # Deserialization safety: restricted strictly to application-generated directory
            vector_store = FAISS.load_local(
                thread_vs_dir, embeddings, allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            _THREAD_RETRIEVERS[tid] = retriever
            _THREAD_METADATA[tid] = meta
            return retriever
        except Exception as e:
            print(f"Error loading vectorstore for thread {tid}: {e}")
            return None

    return None


def ingest_pdf(
    file_bytes: bytes, thread_id: str, filename: Optional[str] = None
) -> dict:
    """
    Ingest an uploaded PDF into the thread's multi-document FAISS index.
    Supports incremental addition of multiple documents without replacing existing ones.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    if not _is_safe_thread_id(thread_id):
        raise ValueError(f"Invalid thread ID format: {thread_id}")

    tid = str(thread_id)
    # Compute SHA-256 hash of file content for reliable uniqueness and change detection
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    file_id = file_hash[:16]
    safe_filename = os.path.basename(filename) if filename else "document.pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            raise ValueError(
                "The uploaded PDF could not be read or contains no extractable text."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # Standardize metadata across chunks for clean citations
        for chunk in chunks:
            raw_page = chunk.metadata.get("page", 0)
            chunk.metadata["human_page"] = int(raw_page) + 1
            chunk.metadata["source_filename"] = safe_filename
            chunk.metadata["thread_id"] = tid
            chunk.metadata["file_id"] = file_id

        thread_vs_dir = _get_thread_vs_dir(tid)
        if not thread_vs_dir:
            raise ValueError("Failed to resolve vectorstore directory.")
        os.makedirs(thread_vs_dir, exist_ok=True)
        files_dir = os.path.join(thread_vs_dir, "files")
        os.makedirs(files_dir, exist_ok=True)

        file_record = {
            "file_id": file_id,
            "filename": safe_filename,
            "file_hash": file_hash,
            "documents": len(docs),
            "chunks": len(chunks),
            "file_size_bytes": len(file_bytes),
            "timestamp": time.time(),
            "chunks_data": [
                {"page_content": c.page_content, "metadata": c.metadata} for c in chunks
            ],
        }
        with open(
            os.path.join(files_dir, f"{file_id}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(file_record, f, indent=2)

        # Update metadata list of files
        meta = thread_document_metadata(tid)
        existing_files = [
            f for f in meta.get("files", []) if f.get("file_id") != file_id
        ]
        existing_files.append(
            {
                "file_id": file_id,
                "filename": safe_filename,
                "file_hash": file_hash,
                "documents": len(docs),
                "chunks": len(chunks),
                "file_size_bytes": len(file_bytes),
                "timestamp": time.time(),
            }
        )
        meta["thread_id"] = tid
        meta["files"] = existing_files

        # Rebuild or incrementally add to vectorstore
        index_file = os.path.join(thread_vs_dir, "index.faiss")
        pkl_file = os.path.join(thread_vs_dir, "index.pkl")
        if os.path.exists(index_file) and os.path.exists(pkl_file):
            try:
                vector_store = FAISS.load_local(
                    thread_vs_dir, embeddings, allow_dangerous_deserialization=True
                )
                vector_store.add_documents(chunks)
            except Exception:
                # Rebuild from all active files if incremental add fails
                all_chunks = []
                for f_info in existing_files:
                    f_p = os.path.join(files_dir, f"{f_info['file_id']}.json")
                    if os.path.exists(f_p):
                        with open(f_p, "r", encoding="utf-8") as rf:
                            fdata = json.load(rf)
                            for cd in fdata.get("chunks_data", []):

                                all_chunks.append(
                                    LCDoc(
                                        page_content=cd["page_content"],
                                        metadata=cd["metadata"],
                                    )
                                )
                vector_store = FAISS.from_documents(all_chunks or chunks, embeddings)
        else:
            vector_store = FAISS.from_documents(chunks, embeddings)

        vector_store.save_local(thread_vs_dir)

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[tid] = retriever
        _THREAD_METADATA[tid] = meta

        # Persist updated metadata
        with open(
            os.path.join(thread_vs_dir, "metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(meta, f, indent=2)

        return meta
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Robust, Production-Grade Tools
# -------------------
_raw_search = DuckDuckGoSearchRun(region="us-en")


@tool
def web_search(query: str) -> dict:
    """
    Search DuckDuckGo for up-to-date public web information, news, and facts.
    Includes error handling for rate limits and connection issues.
    """
    try:
        results = _raw_search.invoke(query)
        if not results or not str(results).strip():
            return {
                "query": query,
                "result": "No relevant search results were found.",
            }
        return {"query": query, "result": str(results)}
    except Exception as e:
        return {
            "query": query,
            "error": "The web search service is currently unavailable or rate-limited. Please retry shortly.",
            "details": str(e),
        }


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: 'add', 'sub', 'mul', 'div'.
    """
    try:
        op = operation.strip().lower()
        if op == "add":
            result = first_num + second_num
        elif op == "sub":
            result = first_num - second_num
        elif op == "mul":
            result = first_num * second_num
        elif op == "div":
            if second_num == 0:
                return {
                    "error": "Mathematical error: Division by zero is undefined.",
                    "operation": op,
                }
            result = first_num / second_num
        else:
            return {
                "error": f"Unsupported operation '{operation}'. Supported operations: 'add', 'sub', 'mul', 'div'."
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": op,
            "result": result,
        }
    except Exception as e:
        return {"error": f"Calculator failed: {str(e)}"}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest real-time stock quote for a ticker symbol (e.g. 'AAPL', 'NVDA')
    using Alpha Vantage. Returns validated pricing and change metrics.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {
            "error": "Stock price lookup failed: ALPHA_VANTAGE_API_KEY environment variable is not configured.",
            "symbol": symbol.upper(),
        }

    clean_symbol = symbol.strip().upper()
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": clean_symbol,
        "apikey": api_key,
    }

    # Bounded retries for transient HTTP failures
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=8)
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.Timeout:
            if attempt == max_retries:
                return {
                    "error": "Stock quote service timed out. Please try again.",
                    "symbol": clean_symbol,
                }
            time.sleep(0.5)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                return {
                    "error": f"Failed to connect to stock service: {str(e)}",
                    "symbol": clean_symbol,
                }
            time.sleep(0.5)
        except Exception as e:
            return {
                "error": f"Unexpected response error: {str(e)}",
                "symbol": clean_symbol,
            }

    if "Error Message" in data:
        return {
            "error": f"Invalid stock symbol '{clean_symbol}'. Please verify the ticker.",
            "symbol": clean_symbol,
        }

    if "Information" in data or "Note" in data:
        return {
            "error": "Alpha Vantage API call rate limit reached (standard tier is 5 calls/min). Please try again in 60 seconds.",
            "symbol": clean_symbol,
        }

    quote = data.get("Global Quote", {})
    if not quote or not quote.get("01. symbol"):
        return {
            "error": f"No active market quote found for symbol '{clean_symbol}'.",
            "symbol": clean_symbol,
        }

    return {
        "symbol": quote.get("01. symbol", clean_symbol),
        "price": quote.get("05. price"),
        "open": quote.get("02. open"),
        "high": quote.get("03. high"),
        "low": quote.get("04. low"),
        "volume": quote.get("06. volume"),
        "latest_trading_day": quote.get("07. latest trading day"),
        "previous_close": quote.get("08. previous close"),
        "change": quote.get("09. change"),
        "change_percent": quote.get("10. change percent"),
    }


# Global tracker for currently active thread to prevent tool parameter omission errors
_ACTIVE_THREAD_ID: Optional[str] = None


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF document(s) for this chat thread.
    Returns chunk contents alongside explicit document name and human-readable page citations.
    """
    effective_tid = thread_id or _ACTIVE_THREAD_ID
    retriever = _get_retriever(effective_tid)
    if retriever is None:
        return {
            "error": "No document is indexed for this chat thread. Please upload one or more PDFs in the sidebar first.",
            "query": query,
        }

    try:
        retrieved_docs = retriever.invoke(query)
        if not retrieved_docs:
            return {
                "query": query,
                "message": "The documents were searched, but no chunks matched the query relevance threshold.",
                "evidence": [],
            }

        evidence = []
        for doc in retrieved_docs:
            source_file = doc.metadata.get("source_filename") or doc.metadata.get(
                "filename", "Document"
            )
            page_num = doc.metadata.get("human_page") or (
                doc.metadata.get("page", 0) + 1
            )
            evidence.append(
                {
                    "source": source_file,
                    "page": page_num,
                    "content": doc.page_content.strip(),
                }
            )

        doc_meta = thread_document_metadata(str(effective_tid))
        files_list = doc_meta.get("files", [])
        active_filenames = (
            [f.get("filename") for f in files_list]
            if files_list
            else [doc_meta.get("filename", "Uploaded PDF")]
        )

        return {
            "query": query,
            "total_chunks_retrieved": len(evidence),
            "active_documents": active_filenames,
            "evidence": evidence,
        }
    except Exception as e:
        return {
            "error": f"Failed to retrieve document context: {str(e)}",
            "query": query,
        }


tools = [web_search, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)


# -------------------
# 4. Graph State Definition
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes & Guardrails
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that enforces grounding, citation rules, and executes tools."""
    global _ACTIVE_THREAD_ID
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id:
        _ACTIVE_THREAD_ID = str(thread_id)

    # Inspect active document status for this specific thread
    doc_meta = thread_document_metadata(str(thread_id))
    files = doc_meta.get("files", [])
    if files:
        doc_lines = []
        for idx, f in enumerate(files, 1):
            doc_lines.append(
                f"{idx}. '{f.get('filename')}' ({f.get('documents', 0)} pages, {f.get('chunks', 0)} chunks indexed)"
            )
        files_summary = "\n".join(doc_lines)
        doc_status = (
            f"Active Uploaded Documents for this chat ({len(files)} file(s)):\n"
            f"{files_summary}\n\n"
            f"- If the user asks what documents are uploaded, lists what files are active, or asks about document titles, "
            f"clearly and accurately list all active documents above.\n"
            f"- For questions about the contents, insights, concepts, comparisons, or facts inside any of these documents, "
            f"ALWAYS invoke `rag_tool` with thread_id `{thread_id}`."
        )
    else:
        doc_status = (
            "Active Uploaded Documents: None currently uploaded for this thread.\n"
            "- If the user asks about document contents or asks what doc is uploaded, politely instruct them to upload a PDF in the sidebar."
        )

    system_prompt = (
        "You are a rigorous, helpful, and intelligent AI assistant equipped with specialized tools.\n\n"
        f"### Document Context Status:\n{doc_status}\n\n"
        "### Tool Usage Guidelines:\n"
        "- For questions about the contents or details of the uploaded document, ALWAYS call `rag_tool` passing the query and "
        f"the thread_id `{thread_id}`.\n"
        "- For arithmetic or financial math, use the `calculator` tool to guarantee exact precision.\n"
        "- For public news, current events, or general web inquiries, use `web_search`.\n"
        "- For real-time ticker prices, use `get_stock_price`.\n\n"
        "### Source-Grounded Answering & Citation Rules (STRICT):\n"
        "1. When answering from document context returned by `rag_tool`:\n"
        "   - Cite the exact source and page number for every key claim using the format `[Filename, Page X]`.\n"
        "   - Distinguish verified evidence from logical inference.\n"
        "   - If the retrieved context does not contain enough evidence to answer the question, EXPLICITLY state:\n"
        "     'The uploaded document does not contain sufficient information to answer this question.'\n"
        "     Do NOT extrapolate or invent facts.\n"
        "2. If no document is currently uploaded and the user asks about document contents, politely prompt them "
        "to upload a PDF using the sidebar uploader.\n\n"
        "### Mathematical & Output Formatting:\n"
        "- Format mathematical equations using standard LaTeX with dollar signs: `$$ ... $$` for standalone equations, "
        "and `$ ... $` for inline variables.\n"
        "- Format all outputs cleanly using structured Markdown (bullet points, bold highlights, and code blocks)."
    )

    system_message = SystemMessage(content=system_prompt)
    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database=CHECKPOINT_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. StateGraph Assembly
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 8. Thread & Metadata Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads_info() -> list[dict]:
    """
    Retrieve all conversation threads with first-question derived titles,
    ordered chronologically.
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
    """Securely delete a conversation thread, its checkpoints, and any saved vectorstore."""
    if not _is_safe_thread_id(thread_id):
        return False
    try:
        tid = str(thread_id)
        with sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False) as db_conn:
            cur = db_conn.cursor()
            cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (tid,))
            cur.execute("DELETE FROM writes WHERE thread_id = ?", (tid,))
            db_conn.commit()

        _THREAD_RETRIEVERS.pop(tid, None)
        _THREAD_METADATA.pop(tid, None)

        thread_vs_dir = _get_thread_vs_dir(tid)
        if thread_vs_dir and os.path.exists(thread_vs_dir):
            shutil.rmtree(thread_vs_dir, ignore_errors=True)
        return True
    except Exception as e:
        print(f"Error deleting thread {thread_id}: {e}")
        return False


def get_thread_documents(thread_id: str) -> list[dict]:
    """Retrieve list of all active documents uploaded for the thread."""
    meta = thread_document_metadata(thread_id)
    return meta.get("files", [])


def thread_has_document(thread_id: str) -> bool:
    """Check if the given thread has one or more active documents."""
    docs = get_thread_documents(thread_id)
    return len(docs) > 0


def thread_document_metadata(thread_id: str) -> dict:
    """Retrieve metadata (files list, filename, total_chunks, total_documents) for the thread."""
    if not _is_safe_thread_id(thread_id):
        return {}
    tid = str(thread_id)
    if tid in _THREAD_METADATA:
        return _THREAD_METADATA[tid]

    thread_vs_dir = _get_thread_vs_dir(tid)
    if not thread_vs_dir:
        return {}

    meta_path = os.path.join(thread_vs_dir, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            files = meta.get("files", [])
            if not files and meta.get("filename"):
                f_hash = meta.get("file_hash", "default")
                files = [
                    {
                        "file_id": f_hash[:16],
                        "filename": meta.get("filename"),
                        "file_hash": f_hash,
                        "documents": meta.get("documents", 0),
                        "chunks": meta.get("chunks", 0),
                        "timestamp": meta.get("timestamp", 0),
                    }
                ]
                meta["files"] = files

            meta["total_files"] = len(files)
            meta["chunks"] = sum(f.get("chunks", 0) for f in files)
            meta["documents"] = sum(f.get("documents", 0) for f in files)
            meta["filename"] = ", ".join(f.get("filename", "") for f in files)
            _THREAD_METADATA[tid] = meta
            return meta
        except Exception:
            pass
    return {}


def remove_document_from_thread(thread_id: str, file_identifier: str) -> bool:
    """
    Remove a specific document from a thread's vectorstore.
    Matches by either file_id or filename.
    Rebuilds the FAISS index for any remaining documents, or cleans up if no documents remain.
    """
    if not _is_safe_thread_id(thread_id):
        return False
    tid = str(thread_id)
    thread_vs_dir = _get_thread_vs_dir(tid)
    if not thread_vs_dir or not os.path.exists(thread_vs_dir):
        return False

    meta = thread_document_metadata(tid)
    target_f_id = file_identifier
    # Find matching file record
    for f in meta.get("files", []):
        if f.get("file_id") == file_identifier or f.get("filename") == file_identifier:
            target_f_id = f.get("file_id", file_identifier)
            break

    files_dir = os.path.join(thread_vs_dir, "files")
    target_f = os.path.join(files_dir, f"{target_f_id}.json")
    if os.path.exists(target_f):
        try:
            os.remove(target_f)
        except Exception as e:
            print(f"Error removing file record {target_f_id}: {e}")

    remaining_files = [
        f
        for f in meta.get("files", [])
        if f.get("file_id") != target_f_id and f.get("filename") != file_identifier
    ]
    meta["files"] = remaining_files
    meta["total_files"] = len(remaining_files)
    meta["chunks"] = sum(f.get("chunks", 0) for f in remaining_files)
    meta["documents"] = sum(f.get("documents", 0) for f in remaining_files)
    meta["filename"] = ", ".join(f.get("filename", "") for f in remaining_files)

    if remaining_files:
        all_chunks = []
        for f_info in remaining_files:
            f_p = os.path.join(files_dir, f"{f_info['file_id']}.json")
            if os.path.exists(f_p):
                try:
                    with open(f_p, "r", encoding="utf-8") as rf:
                        fdata = json.load(rf)
                        from langchain_core.documents import Document as LCDoc

                        for cd in fdata.get("chunks_data", []):
                            all_chunks.append(
                                LCDoc(
                                    page_content=cd["page_content"],
                                    metadata=cd["metadata"],
                                )
                            )
                except Exception as e:
                    print(f"Error reading chunk file {f_p}: {e}")

        if all_chunks:
            new_vs = FAISS.from_documents(all_chunks, embeddings)
            new_vs.save_local(thread_vs_dir)
            _THREAD_RETRIEVERS[tid] = new_vs.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )

        with open(
            os.path.join(thread_vs_dir, "metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(meta, f, indent=2)
        _THREAD_METADATA[tid] = meta
    else:
        # No files remaining: clean up FAISS index files
        for fname in ["index.faiss", "index.pkl"]:
            f_path = os.path.join(thread_vs_dir, fname)
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except Exception:
                    pass
        with open(
            os.path.join(thread_vs_dir, "metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({"thread_id": tid, "files": []}, f, indent=2)
        _THREAD_RETRIEVERS.pop(tid, None)
        _THREAD_METADATA[tid] = {"thread_id": tid, "files": []}

    return True


def find_most_recent_document_metadata() -> dict:
    """Find the most recently uploaded document across any thread on disk."""
    if not os.path.exists(VECTORSTORE_DIR):
        return {}
    best_meta = {}
    best_time = 0.0
    for tid in os.listdir(VECTORSTORE_DIR):
        meta_file = os.path.join(VECTORSTORE_DIR, tid, "metadata.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    t = float(data.get("timestamp", 0))
                    if t > best_time:
                        best_time = t
                        best_meta = data
            except Exception:
                pass
    return best_meta


def copy_document_to_thread(source_thread_id: str, target_thread_id: str) -> dict:
    """Copy or reuse an existing vectorstore from source_thread_id to target_thread_id."""
    if not _is_safe_thread_id(source_thread_id) or not _is_safe_thread_id(
        target_thread_id
    ):
        return {}
    src_dir = _get_thread_vs_dir(str(source_thread_id))
    tgt_dir = _get_thread_vs_dir(str(target_thread_id))
    if not src_dir or not tgt_dir or not os.path.exists(src_dir):
        return {}
    try:
        os.makedirs(tgt_dir, exist_ok=True)
        for fname in ["index.faiss", "index.pkl", "metadata.json"]:
            src_f = os.path.join(src_dir, fname)
            if os.path.exists(src_f):
                shutil.copy2(src_f, os.path.join(tgt_dir, fname))

        meta_file = os.path.join(tgt_dir, "metadata.json")
        meta = {}
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        meta["thread_id"] = str(target_thread_id)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        _THREAD_METADATA[str(target_thread_id)] = meta
        _THREAD_RETRIEVERS.pop(str(target_thread_id), None)
        return meta
    except Exception as e:
        print(f"Error copying document to thread {target_thread_id}: {e}")
        return {}


def detach_document_from_thread(thread_id: str) -> bool:
    """Detach/remove the uploaded document vectorstore from a specific thread."""
    if not _is_safe_thread_id(thread_id):
        return False
    tid = str(thread_id)
    _THREAD_METADATA.pop(tid, None)
    _THREAD_RETRIEVERS.pop(tid, None)
    thread_vs_dir = _get_thread_vs_dir(tid)
    if thread_vs_dir and os.path.exists(thread_vs_dir):
        try:
            shutil.rmtree(thread_vs_dir)
            return True
        except Exception as e:
            print(f"Error detaching document from thread {tid}: {e}")
            return False
    return True


ACTIVE_SESSION_FILE = os.path.join(VECTORSTORE_DIR, ".active_session.json")


def get_last_active_thread_id() -> Optional[str]:
    """Retrieve the last active thread ID persisted on disk."""
    if os.path.exists(ACTIVE_SESSION_FILE):
        try:
            with open(ACTIVE_SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                tid = data.get("thread_id")
                if tid and _is_safe_thread_id(tid):
                    return str(tid)
        except Exception:
            pass
    return None


def set_last_active_thread_id(thread_id: str) -> None:
    """Persist the current active thread ID so browser refresh restores it reliably."""
    if not _is_safe_thread_id(thread_id):
        return
    try:
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        with open(ACTIVE_SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump({"thread_id": str(thread_id), "updated_at": time.time()}, f)
    except Exception:
        pass
