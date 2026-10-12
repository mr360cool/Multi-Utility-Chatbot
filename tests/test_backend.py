"""
Unit and integration tests for final_backend.py.
Run with: python -m pytest tests/
"""

import os
from unittest.mock import patch, MagicMock
import pytest

from final_backend import (
    _is_safe_thread_id,
    _get_thread_vs_dir,
    calculator,
    get_stock_price,
    web_search,
    rag_tool,
    chatbot,
    VECTORSTORE_DIR,
)


# -------------------------------------------------------------
# 1. Calculator Unit Tests
# -------------------------------------------------------------
def test_calculator_arithmetic():
    """Verify standard arithmetic operations return exact results."""
    add_res = calculator.invoke({"first_num": 10, "second_num": 5, "operation": "add"})
    assert add_res["result"] == 15

    sub_res = calculator.invoke({"first_num": 10, "second_num": 5, "operation": "sub"})
    assert sub_res["result"] == 5

    mul_res = calculator.invoke({"first_num": 4, "second_num": 6, "operation": "mul"})
    assert mul_res["result"] == 24

    div_res = calculator.invoke({"first_num": 20, "second_num": 4, "operation": "div"})
    assert div_res["result"] == 5.0


def test_calculator_division_by_zero():
    """Verify division by zero returns a clean error payload instead of raising an exception."""
    res = calculator.invoke({"first_num": 10, "second_num": 0, "operation": "div"})
    assert "error" in res
    assert "Division by zero" in res["error"]


def test_calculator_invalid_operation():
    """Verify unsupported operations return a helpful error message."""
    res = calculator.invoke({"first_num": 10, "second_num": 2, "operation": "power"})
    assert "error" in res
    assert "Unsupported operation" in res["error"]


# -------------------------------------------------------------
# 2. Stock Quote Tool & Error Handling
# -------------------------------------------------------------
def test_stock_price_missing_api_key():
    """Verify clean failure when ALPHA_VANTAGE_API_KEY is not set."""
    with patch.dict(os.environ, {}, clear=True):
        res = get_stock_price.invoke({"symbol": "AAPL"})
        assert "error" in res
        assert "ALPHA_VANTAGE_API_KEY" in res["error"]


@patch("final_backend.requests.get")
def test_stock_price_rate_limit_detection(mock_get):
    """Verify provider-level rate-limit messages ('Note') are caught and handled."""
    with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "dummy_key"}):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute."
        }
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        res = get_stock_price.invoke({"symbol": "NVDA"})
        assert "error" in res
        assert "rate limit" in res["error"].lower()


@patch("final_backend.requests.get")
def test_stock_price_valid_quote(mock_get):
    """Verify parsing of valid Global Quote payloads."""
    with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "dummy_key"}):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "Global Quote": {
                "01. symbol": "AAPL",
                "05. price": "230.50",
                "09. change": "+2.10",
                "10. change percent": "+0.92%",
            }
        }
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        res = get_stock_price.invoke({"symbol": "AAPL"})
        assert res["symbol"] == "AAPL"
        assert res["price"] == "230.50"
        assert res["change"] == "+2.10"


# -------------------------------------------------------------
# 3. Security & Path Traversal Hardening
# -------------------------------------------------------------
def test_is_safe_thread_id():
    """Verify thread ID validator permits safe tokens and rejects traversal attempts."""
    assert _is_safe_thread_id("uuid-1234-abcd") is True
    assert _is_safe_thread_id("session_42") is True
    assert _is_safe_thread_id("a1b2c3d4-e5f6") is True

    # Traversal and injection attempts must fail
    assert _is_safe_thread_id("../../../etc/passwd") is False
    assert _is_safe_thread_id("..\\..\\windows") is False
    assert _is_safe_thread_id("thread/subpath") is False
    assert _is_safe_thread_id("") is False
    assert _is_safe_thread_id(None) is False


def test_vectorstore_path_traversal_defense():
    """Verify that resolved vectorstore path never escapes the root directory."""
    assert _get_thread_vs_dir("../../evil_dir") is None
    safe_path = _get_thread_vs_dir("valid-thread-id")
    assert safe_path is not None
    assert safe_path.startswith(VECTORSTORE_DIR)


# -------------------------------------------------------------
# 4. RAG Retrieval Guardrails
# -------------------------------------------------------------
def test_rag_tool_no_indexed_doc():
    """Verify rag_tool fails gracefully when no document has been indexed for the thread."""
    res = rag_tool.invoke({"query": "What is net margin?", "thread_id": "non-existent-thread-id"})
    assert "error" in res
    assert "No document is indexed" in res["error"]


# -------------------------------------------------------------
# 5. Graph State & Tool Nodes
# -------------------------------------------------------------
def test_graph_state_initialization():
    """Verify the compiled StateGraph exposes the required entry nodes and checkpointer."""
    assert chatbot is not None
    # Compiled state graph nodes verification
    node_names = list(chatbot.get_graph().nodes.keys())
    assert "chat_node" in node_names
    assert "tools" in node_names


# -------------------------------------------------------------
# 6. Multi-Document Management Unit Tests
# -------------------------------------------------------------
def test_multi_document_helpers():
    """Verify multi-document helper functions for empty and non-existent threads."""
    from final_backend import get_thread_documents, thread_has_document, remove_document_from_thread

    # Empty thread
    docs = get_thread_documents("test-empty-thread-uuid")
    assert docs == []
    assert thread_has_document("test-empty-thread-uuid") is False

    # Removing non-existent file from non-existent thread
    removed = remove_document_from_thread("test-empty-thread-uuid", "non-existent-file-id")
    assert removed is False

