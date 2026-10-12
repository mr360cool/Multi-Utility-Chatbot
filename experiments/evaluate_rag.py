from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from statistics import mean
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from final_backend import embeddings, llm, llm_with_tools

load_dotenv()

# ----------------------------------------------------------------
# LLM-as-a-Judge Prompts & Helpers
# ----------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """\
You are an impartial, highly rigorous AI evaluation judge assessing Retrieval-Augmented Generation (RAG) components.
Give a concise rationale for your score. Do not provide hidden chain-of-thought.
You never assume outside facts not explicitly stated in the context.
Always respond with valid JSON matching the exact schema requested.
"""

FAITHFULNESS_PROMPT_TEMPLATE = """\
## Task: Faithfulness / Anti-Hallucination Evaluation

**User Question:**
{question}

**Retrieved Evidence (the ONLY allowed ground truth):**
{context}

**Generated Response:**
{answer}

## Scoring Instructions (0 to 3 scale):
Step 1: Decompose the generated response into discrete atomic claims.
Step 2: Check each claim against the retrieved evidence.
Step 3: Assign score:
  - 3: Fully faithful. 100% of claims are strictly grounded in the evidence. No hallucinations.
  - 2: Mostly faithful. Minor phrasing extrapolation, but no contradictory or fabricated facts.
  - 1: Partially faithful. Contains noticeable claims not present in or contrary to the evidence.
  - 0: Unfaithful. Major hallucinations or completely disconnected from evidence.

Output ONLY valid JSON:
{{"reasoning": "<concise step-by-step audit>", "score": <0-3>, "verdict": "<FAITHFUL|MOSTLY_FAITHFUL|PARTIALLY_FAITHFUL|UNFAITHFUL>"}}
"""

ANSWER_RELEVANCE_PROMPT_TEMPLATE = """\
## Task: Answer Relevance Evaluation

**User Question:**
{question}

**Generated Response:**
{answer}

## Scoring Instructions (0 to 3 scale):
Step 1: Identify the explicit information requested in the question.
Step 2: Assess whether the generated response directly answers it without evading or drifting off-topic.
Step 3: Assign score:
  - 3: Directly, accurately, and completely addresses the user prompt with zero fluff.
  - 2: Relevant and answers the question, with minor unnecessary tangents.
  - 1: Partially answers, but omits core aspects of the query.
  - 0: Completely off-topic or unresponsive.

Output ONLY valid JSON:
{{"reasoning": "<concise step-by-step audit>", "score": <0-3>, "verdict": "<HIGHLY_RELEVANT|RELEVANT|PARTIALLY_RELEVANT|IRRELEVANT>"}}
"""

CONTEXT_RELEVANCE_PROMPT_TEMPLATE = """\
## Task: Context Relevance Evaluation (Retriever Precision)

**User Query:**
{query}

**Retrieved Context Chunks:**
{retrieved_chunks}

## Scoring Instructions (0 to 3 scale):
Step 1: Determine whether the retrieved chunks contain the necessary information to answer the query.
Step 2: Assess the signal-to-noise ratio: do the retrieved chunks contain high-signal context or irrelevant noise?
Step 3: Assign score:
  - 3: High precision. Chunks contain directly relevant evidence needed to answer the query with minimal noise.
  - 2: Good precision. Contains the needed facts along with some adjacent background context.
  - 1: Low precision. Weakly related or contains excessive distracting noise with minimal relevance.
  - 0: Irrelevant. Chunks do not contain facts related to the query.

Output ONLY valid JSON:
{{"reasoning": "<concise analysis>", "score": <0-3>, "verdict": "<HIGH_PRECISION|MODERATE_PRECISION|LOW_PRECISION|IRRELEVANT>"}}
"""


def _judge_score(prompt: str) -> Dict[str, Any]:
    """Invoke the judge and fail loudly for invalid output instead of scoring it as zero."""
    response = llm.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    text = str(response.content).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError(f"Judge returned no JSON object: {text[:300]}")
    result = json.loads(text[start:end])
    score = result.get("score")
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 3:
        raise ValueError(f"Judge returned an invalid score: {score!r}")
    if not isinstance(result.get("reasoning"), str):
        raise ValueError("Judge JSON is missing a string 'reasoning' field")
    result["score"] = int(score)
    return result


def _percentile(values: List[float], percentile: float) -> float:
    """Return a nearest-rank percentile; empty input is reported as zero."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((percentile / 100) * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w.%$]+", " ", text.lower())).strip()


def _citation_check(answer: str, context: str) -> Dict[str, Any]:
    """Check citation syntax and that every cited source/page pair occurs in the evidence."""
    citation_pattern = re.compile(
        r"\[(?:Document:\s*)?([^,\]\n]+),\s*Page\s*:?\s*(\d+)\]", re.IGNORECASE
    )
    answer_citations = {
        (source.strip().lower(), int(page))
        for source, page in citation_pattern.findall(answer)
    }
    context_citations = {
        (source.strip().lower(), int(page))
        for source, page in citation_pattern.findall(context)
    }
    valid = (
        bool(answer_citations)
        and bool(context_citations)
        and answer_citations.issubset(context_citations)
    )
    return {
        "has_citation": bool(answer_citations),
        "citations_valid": valid,
        "citations": sorted(
            [{"source": source, "page": page} for source, page in answer_citations],
            key=lambda item: (item["source"], item["page"]),
        ),
    }


# ----------------------------------------------------------------
# Stage 1: Retrieval Evaluation (Isolated Vector Retriever)
# ----------------------------------------------------------------
def evaluate_retrieval_stage(
    corpus: List[Dict[str, Any]],
    test_cases: List[Dict[str, Any]],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Evaluates the Retriever independently from generation.
    Metrics:
    - Hit Rate @ K, Precision@K, Recall@K, MRR, and nDCG@K
    - Mean Reciprocal Rank (MRR)
    - Context Relevance (LLM-as-a-Judge CoT 0-3)
    - Latency (ms)
    """
    print(
        f"\n--- [1/4] Evaluating Retrieval in Isolation (Top-{top_k} FAISS Search) ---"
    )

    # 1. Build test vectorstore from corpus documents
    docs = [
        Document(
            page_content=item["content"],
            metadata={
                "doc_id": item["id"],
                "source": item["source"],
                "human_page": item["page"],
            },
        )
        for item in corpus
    ]
    vector_store = FAISS.from_documents(docs, embeddings)

    results = []
    hits = 0
    reciprocal_ranks = []
    precisions = []
    recalls = []
    ndcgs = []
    relevance_scores = []
    latencies = []

    for item in test_cases:
        query = item["query"]
        relevant_ids = set(
            item["relevant_doc_ids"]
            if "relevant_doc_ids" in item
            else [item["target_doc_id"]]
        )
        if not relevant_ids:
            raise ValueError(
                f"Retrieval case {item.get('id')} has no relevant document IDs"
            )
        target_doc_id = item.get("target_doc_id", sorted(relevant_ids)[0])
        target_source = item.get("target_source", "")
        target_page = item.get("target_page", 0)

        start_time = time.time()
        retrieved_docs = vector_store.similarity_search(query, k=top_k)
        latency = (time.time() - start_time) * 1000.0  # ms
        latencies.append(latency)

        retrieved_ids = [d.metadata.get("doc_id") for d in retrieved_docs]
        retrieved_text = "\n---\n".join(
            [
                f"[{d.metadata.get('source')}, Page {d.metadata.get('human_page')}]: {d.page_content}"
                for d in retrieved_docs
            ]
        )

        # Compute Hit Rate @ K
        hit = bool(relevant_ids.intersection(retrieved_ids))
        if hit:
            hits += 1
            first_rank = next(
                rank
                for rank, doc_id in enumerate(retrieved_ids, start=1)
                if doc_id in relevant_ids
            )
            rank = first_rank
            rr = 1.0 / rank
        else:
            rank = None
            rr = 0.0
        reciprocal_ranks.append(rr)
        relevant_retrieved = sum(doc_id in relevant_ids for doc_id in retrieved_ids)
        precisions.append(relevant_retrieved / max(top_k, 1))
        recalls.append(relevant_retrieved / max(len(relevant_ids), 1))
        dcg = sum(
            1.0 / math.log2(position + 1)
            for position, doc_id in enumerate(retrieved_ids, start=1)
            if doc_id in relevant_ids
        )
        ideal_count = min(len(relevant_ids), top_k)
        ideal_dcg = sum(
            1.0 / math.log2(position + 1) for position in range(1, ideal_count + 1)
        )
        ndcgs.append(dcg / ideal_dcg if ideal_dcg else 0.0)

        # Context Relevance via LLM Judge
        ctx_judge = _judge_score(
            CONTEXT_RELEVANCE_PROMPT_TEMPLATE.format(
                query=query, retrieved_chunks=retrieved_text
            )
        )
        ctx_score = ctx_judge.get("score", 0)
        relevance_scores.append(ctx_score)

        status = "HIT ✓" if hit else "MISS ✗"
        print(
            f"  [{status}] '{query[:45]}...' → Target: {target_doc_id} | "
            f"Rank: {rank if rank else 'N/A'} (RR: {rr:.2f}) | "
            f"Ctx Relevance: {ctx_score}/3 ({latency:.1f}ms)"
        )

        results.append(
            {
                "id": item["id"],
                "query": query,
                "target_doc_id": target_doc_id,
                "relevant_doc_ids": sorted(relevant_ids),
                "target_source": target_source,
                "target_page": target_page,
                "hit": hit,
                "rank": rank,
                "reciprocal_rank": round(rr, 3),
                "context_relevance_score": ctx_score,
                "context_relevance_verdict": ctx_judge.get("verdict"),
                "context_relevance_reasoning": ctx_judge.get("reasoning", "")[:180],
                "latency_ms": round(latency, 2),
            }
        )

    total = len(test_cases)
    hit_rate_pct = round((hits / max(total, 1)) * 100, 1)
    mrr = round(sum(reciprocal_ranks) / max(total, 1), 3)
    avg_relevance = round(sum(relevance_scores) / max(total, 1), 2)
    avg_relevance_pct = round((avg_relevance / 3.0) * 100, 1)
    avg_latency_ms = round(sum(latencies) / max(total, 1), 1)
    p95_latency_ms = round(_percentile(latencies, 95), 1)

    return {
        "corpus_count": len(corpus),
        "total": total,
        "hits": hits,
        "hit_rate_pct": hit_rate_pct,
        "mrr": mrr,
        "precision_at_k": round(mean(precisions), 3) if precisions else 0.0,
        "recall_at_k": round(mean(recalls), 3) if recalls else 0.0,
        "ndcg_at_k": round(mean(ndcgs), 3) if ndcgs else 0.0,
        "avg_context_relevance_score": avg_relevance,
        "avg_context_relevance_pct": avg_relevance_pct,
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "details": results,
    }


# ----------------------------------------------------------------
# Stage 2: Generation Evaluation (Isolated LLM Generator)
# ----------------------------------------------------------------
def evaluate_generation_stage(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates the LLM Generator independently from retrieval.
    Metrics:
    - Faithfulness (0-3 rubric, atomic claim verification)
    - Answer Relevance (0-3 rubric)
    - Citation Compliance ([Doc, Page X])
    - Refusal / Abstention Accuracy on unanswerable queries
    - Generation Latency (s)
    """
    print("\n--- [2/4] Evaluating Generation in Isolation (LLM-as-a-Judge CoT) ---")

    ANSWER_SYS = (
        "You are an evidence-grounded AI assistant. Answer strictly and solely from the provided context.\n"
        "Citation rules:\n"
        "1. For every factual claim, cite the document name and page number as [DocumentName, Page X].\n"
        "2. If the context does not contain sufficient facts to answer the question, respond EXACTLY with: "
        "'The uploaded document does not contain sufficient information to answer this question.'\n"
        "Do not invent, assume, or extrapolate."
    )

    results = []
    faith_total = 0
    rel_total = 0
    citations_valid = 0
    refusal_total = 0
    refusal_correct = 0
    answerable_count = 0
    answerable_refusal_count = 0
    key_fact_hits = 0
    key_fact_total = 0
    generation_error_count = 0
    latencies = []

    for item in test_cases:
        query = item["query"]
        context = item["context"]
        is_unanswerable = item.get("unanswerable", False)

        start_time = time.time()
        try:
            resp = llm.invoke(
                [
                    SystemMessage(content=ANSWER_SYS),
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
                ]
            )
            answer = str(resp.content).strip()
            latency = time.time() - start_time
            latencies.append(latency)
        except Exception as e:
            print(f"  [ERROR] Answer generation failed for {item['id']}: {e}")
            results.append({"id": item["id"], "error": str(e)})
            generation_error_count += 1
            continue

        if is_unanswerable:
            refusal_total += 1
            refused = any(
                kw in answer.lower()
                for kw in [
                    "does not contain",
                    "insufficient information",
                    "not mentioned",
                    "not found",
                    "cannot answer",
                ]
            )
            if refused:
                refusal_correct += 1
            status = "PASS ✓" if refused else "FAIL ✗"
            print(
                f"  [{status}] Refusal Test [{item['id']}] → Correctly abstained: {refused} ({latency:.2f}s)"
            )
            results.append(
                {
                    "id": item["id"],
                    "type": "refusal",
                    "passed": refused,
                    "answer_sample": answer[:100],
                    "latency_sec": round(latency, 2),
                }
            )
            continue

        answerable_count += 1
        if any(
            phrase in answer.lower()
            for phrase in (
                "does not contain sufficient information",
                "cannot answer",
                "not enough information",
                "insufficient information",
            )
        ):
            answerable_refusal_count += 1

        # Faithfulness Judge
        faith_res = _judge_score(
            FAITHFULNESS_PROMPT_TEMPLATE.format(
                question=query, context=context, answer=answer
            )
        )
        f_score = faith_res.get("score", 0)
        faith_total += f_score

        # Answer Relevance Judge
        rel_res = _judge_score(
            ANSWER_RELEVANCE_PROMPT_TEMPLATE.format(question=query, answer=answer)
        )
        r_score = rel_res.get("score", 0)
        rel_total += r_score

        # Check citation adherence
        citation_result = _citation_check(answer, context)
        has_citation = citation_result["has_citation"]
        citations_valid += int(citation_result["citations_valid"])
        expected_facts = item.get("expected_key_facts", [])
        normalized_answer = _normalized_text(answer)
        matched_facts = [
            fact
            for fact in expected_facts
            if _normalized_text(fact) in normalized_answer
        ]
        key_fact_hits += len(matched_facts)
        key_fact_total += len(expected_facts)

        print(
            f"  [ID: {item['id']}] Faithfulness: {f_score}/3 ({faith_res.get('verdict')}) | "
            f"Relevance: {r_score}/3 ({rel_res.get('verdict')}) | "
            f"Cited: {'✓' if has_citation else '✗'} ({latency:.2f}s)"
        )

        results.append(
            {
                "id": item["id"],
                "type": "grounding",
                "answer": answer,
                "faithfulness_score": f_score,
                "faithfulness_verdict": faith_res.get("verdict"),
                "faithfulness_reasoning": faith_res.get("reasoning", "")[:180],
                "relevance_score": r_score,
                "relevance_verdict": rel_res.get("verdict"),
                "relevance_reasoning": rel_res.get("reasoning", "")[:180],
                "has_citation": has_citation,
                "citations_valid": citation_result["citations_valid"],
                "citations": citation_result["citations"],
                "matched_key_facts": matched_facts,
                "expected_key_facts": expected_facts,
                "latency_sec": round(latency, 2),
            }
        )

    avg_faith = round(faith_total / max(answerable_count, 1), 2)
    avg_faith_pct = round((avg_faith / 3.0) * 100, 1)
    avg_rel = round(rel_total / max(answerable_count, 1), 2)
    avg_rel_pct = round((avg_rel / 3.0) * 100, 1)
    citation_pct = round((citations_valid / max(answerable_count, 1)) * 100, 1)
    key_fact_recall_pct = round((key_fact_hits / max(key_fact_total, 1)) * 100, 1)
    refusal_pct = round((refusal_correct / max(refusal_total, 1)) * 100, 1)
    answerable_non_abstention_pct = round(
        ((answerable_count - answerable_refusal_count) / max(answerable_count, 1))
        * 100,
        1,
    )
    avg_gen_latency = round(sum(latencies) / max(len(latencies), 1), 2)
    p95_gen_latency = round(_percentile(latencies, 95), 2)

    return {
        "total": len(test_cases),
        "answerable_count": answerable_count,
        "refusal_total": refusal_total,
        "avg_faithfulness_score": avg_faith,
        "avg_faithfulness_pct": avg_faith_pct,
        "avg_relevance_score": avg_rel,
        "avg_relevance_pct": avg_rel_pct,
        "citation_adherence_pct": citation_pct,
        "key_fact_recall_pct": key_fact_recall_pct,
        "refusal_accuracy_pct": refusal_pct,
        "answerable_non_abstention_pct": answerable_non_abstention_pct,
        "answerable_refusal_count": answerable_refusal_count,
        "generation_error_count": generation_error_count,
        "avg_latency_sec": avg_gen_latency,
        "p95_latency_sec": p95_gen_latency,
        "details": results,
    }


# ----------------------------------------------------------------
# Stage 3: Tool Routing Precision
# ----------------------------------------------------------------
def evaluate_tool_routing_stage(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Test deterministic agent routing to tools based on query intent."""
    print("\n--- [3/4] Evaluating Tool Routing Precision ---")
    results = []
    correct = 0

    for item in test_cases:
        query = item["query"]
        expected_tools = set(item.get("expected_tools", [item.get("expected_tool")]))
        expected_tools.discard(None)
        start_time = time.time()

        try:
            response = llm_with_tools.invoke([HumanMessage(content=query)])
            latency = time.time() - start_time
            called_tools = [
                tc.get("name") for tc in getattr(response, "tool_calls", [])
            ]
            passed = set(called_tools) == expected_tools
            if passed:
                correct += 1

            status = "PASS ✓" if passed else "FAIL ✗"
            print(
                f"  [{status}] '{query[:48]}...' → Expected: {sorted(expected_tools)} | Got: {called_tools} ({latency:.2f}s)"
            )
            results.append(
                {
                    "id": item["id"],
                    "query": query,
                    "expected_tools": sorted(expected_tools),
                    "expected_tool": item.get("expected_tool"),
                    "called_tools": called_tools,
                    "passed": passed,
                    "latency_sec": round(latency, 2),
                }
            )
        except Exception as e:
            print(f"  [ERROR] {item['id']}: {e}")
            results.append(
                {
                    "id": item["id"],
                    "query": query,
                    "expected_tools": sorted(expected_tools),
                    "expected_tool": item.get("expected_tool"),
                    "called_tools": [],
                    "passed": False,
                    "error": str(e),
                    "latency_sec": 0.0,
                }
            )

    accuracy = (correct / max(len(test_cases), 1)) * 100.0
    return {
        "total": len(test_cases),
        "passed": correct,
        "accuracy_percent": round(accuracy, 1),
        "details": results,
    }


# ----------------------------------------------------------------
# Stage 4: Decoupled Failure Mode Attribution Analysis
# ----------------------------------------------------------------
def generate_decoupled_attribution(
    retrieval_res: Dict[str, Any],
    generation_res: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attribute system outcomes to their root cause:
    - Retrieval Failures (Retriever Miss) vs.
    - Generation Failures (LLM Hallucination / Omission)
    """
    print("\n--- [4/4] Performing Decoupled Failure Mode Attribution ---")
    retrieval_misses = [d for d in retrieval_res["details"] if not d.get("hit")]
    low_precision_chunks = [
        d for d in retrieval_res["details"] if d.get("context_relevance_score", 3) < 2
    ]
    generation_hallucinations = [
        d
        for d in generation_res["details"]
        if d.get("type") == "grounding" and d.get("faithfulness_score", 3) < 2
    ]
    refusal_failures = [
        d
        for d in generation_res["details"]
        if d.get("type") == "refusal" and not d.get("passed")
    ]

    print(f"  • Retrieval Misses (Context not in top-k): {len(retrieval_misses)}")
    print(f"  • Low Context Precision (Distractor noise): {len(low_precision_chunks)}")
    print(
        f"  • Generator Hallucinations (Unfaithful claims): {len(generation_hallucinations)}"
    )
    print(
        f"  • Refusal Failures (Failed to abstain on unanswerable): {len(refusal_failures)}"
    )

    return {
        "retrieval_miss_count": len(retrieval_misses),
        "low_precision_count": len(low_precision_chunks),
        "hallucination_count": len(generation_hallucinations),
        "refusal_failure_count": len(refusal_failures),
    }


# ----------------------------------------------------------------
# Report Generator
# ----------------------------------------------------------------
def generate_markdown_report(
    retrieval_res: Dict[str, Any],
    generation_res: Dict[str, Any],
    routing_res: Dict[str, Any],
    attribution: Dict[str, Any],
    output_path: str = "EVALUATION_REPORT.md",
):
    """Generate professional, publication-ready benchmark report for portfolio and resume."""
    report = f"""# Decoupled RAG & Agent Evaluation Benchmark Report

> **Evaluation Methodology:** Decoupled Component Evaluation (Retrieval vs. Generation)
> Guided by AI engineering interview standards:
> *"Most people evaluate the final answer. Strong candidates evaluate retrieval and generation separately."*
> Scored using **LLM-as-a-Judge** with concise, explicit rubrics. This is a small, hand-authored benchmark over {len(retrieval_res['details'])} queries and {len(generation_res['details'])} generation cases, not a production-quality guarantee.

---

## Benchmark Executive Summary

| Evaluation Dimension | Metric | Measured Score | Target Threshold | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Retrieval Stage** | **Hit Rate @ 3** | **{retrieval_res['hit_rate_pct']}%** | > 85.0% | {'✅ Met' if retrieval_res['hit_rate_pct'] >= 85 else '⚠️ Review'} |
| **Retrieval Stage** | **Mean Reciprocal Rank (MRR)** | **{retrieval_res['mrr']}** | > 0.75 | {'✅ Met' if retrieval_res['mrr'] >= 0.75 else '⚠️ Review'} |
| **Retrieval Stage** | **Precision@3 / Recall@3** | **{retrieval_res['precision_at_k']} / {retrieval_res['recall_at_k']}** | Track trend | — |
| **Retrieval Stage** | **nDCG@3** | **{retrieval_res['ndcg_at_k']}** | Track trend | — |
| **Retrieval Stage** | **Context Relevance (Signal-to-Noise)** | **{retrieval_res['avg_context_relevance_pct']}%** | > 80.0% | {'✅ Met' if retrieval_res['avg_context_relevance_pct'] >= 80 else '⚠️ Review'} |
| **Generation Stage** | **Faithfulness (Anti-Hallucination)** | **{generation_res['avg_faithfulness_pct']}%** | > 85.0% | {'✅ Met' if generation_res['avg_faithfulness_pct'] >= 85 else '⚠️ Review'} |
| **Generation Stage** | **Answer Relevance** | **{generation_res['avg_relevance_pct']}%** | > 85.0% | {'✅ Met' if generation_res['avg_relevance_pct'] >= 85 else '⚠️ Review'} |
| **Generation Stage** | **Citation Source/Page Validity** | **{generation_res['citation_adherence_pct']}%** | Track trend | — |
| **Generation Stage** | **Expected Key-Fact Recall** | **{generation_res['key_fact_recall_pct']}%** | Track trend | — |
| **Safety Guardrail** | **Refusal / Abstention Accuracy** | **{generation_res['refusal_accuracy_pct']}%** | 100.0% | {'✅ Met' if generation_res['refusal_accuracy_pct'] == 100 else '⚠️ Review'} |
| **Safety Guardrail** | **Answerable Non-Abstention Rate** | **{generation_res['answerable_non_abstention_pct']}%** | Track trend | — |
| **Agent Intent** | **Exact Tool-Set Accuracy** | **{routing_res['accuracy_percent']}%** | Track trend | — |

---

## 1. Stage 1: Isolated Retrieval Benchmark

Evaluates whether the vectorstore (FAISS + OpenAI Embeddings) successfully surfaces the required ground-truth evidence before the LLM ever generates an answer.

- **Hit Rate @ 3:** {retrieval_res['hit_rate_pct']}% ({retrieval_res['hits']}/{retrieval_res['total']})
- **Mean Reciprocal Rank (MRR):** {retrieval_res['mrr']}
- **Context Relevance Score:** {retrieval_res['avg_context_relevance_score']}/3 ({retrieval_res['avg_context_relevance_pct']}%)
- **Mean Retrieval Latency:** {retrieval_res['avg_latency_ms']} ms
- **P95 Retrieval Latency:** {retrieval_res['p95_latency_ms']} ms
- **Precision@3 / Recall@3 / nDCG@3:** {retrieval_res['precision_at_k']} / {retrieval_res['recall_at_k']} / {retrieval_res['ndcg_at_k']}
- **Indexed corpus size:** {retrieval_res['corpus_count']} hand-authored source snippets (not the full PDF ingestion path)

| Test ID | Query Preview | Target Document | Rank | RR | Ctx Relevance | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for d in retrieval_res["details"]:
        badge = f"Rank {d['rank']}" if d.get("hit") else "MISS ❌"
        q = (d["query"][:42] + "...") if len(d["query"]) > 42 else d["query"]
        report += (
            f"| `{d['id']}` | {q} | `{d.get('target_source')}` (P.{d.get('target_page')}) | "
            f"{badge} | {d.get('reciprocal_rank')} | {d.get('context_relevance_score')}/3 | {d.get('latency_ms')}ms |\n"
        )

    report += f"""
---

## 2. Stage 2: Isolated Generation Benchmark

Evaluates the generator's ability to synthesize factual responses strictly from provided context, enforce `[Document, Page X]` source citations, and decline to answer unanswerable prompts.

- **Faithfulness (Anti-Hallucination):** {generation_res['avg_faithfulness_score']}/3 ({generation_res['avg_faithfulness_pct']}%)
- **Answer Relevance:** {generation_res['avg_relevance_score']}/3 ({generation_res['avg_relevance_pct']}%)
- **Citation Source/Page Validity:** {generation_res['citation_adherence_pct']}% (checks cited source/page pairs against context; not claim-level coverage)
- **Expected Key-Fact Recall:** {generation_res['key_fact_recall_pct']}%
- **Adversarial Refusal Accuracy:** {generation_res['refusal_accuracy_pct']}%
- **Answerable Non-Abstention Rate:** {generation_res['answerable_non_abstention_pct']}%
- **Mean Generation Latency:** {generation_res['avg_latency_sec']}s
- **P95 Generation Latency:** {generation_res['p95_latency_sec']}s
- **Generation errors:** {generation_res['generation_error_count']}

| Test ID | Test Type | Faithfulness | Relevance | Citation source/page valid? | Result | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for d in generation_res["details"]:
        if d.get("type") == "grounding":
            pass_badge = "✅ PASS" if d.get("faithfulness_score", 0) >= 2 else "⚠️ LOW"
            cited_badge = "✓" if d.get("citations_valid") else "✗"
            report += (
                f"| `{d['id']}` | Grounded QA | {d.get('faithfulness_score')}/3 ({d.get('faithfulness_verdict')}) | "
                f"{d.get('relevance_score')}/3 | {cited_badge} | {pass_badge} | {d.get('latency_sec')}s |\n"
            )
        elif d.get("type") == "refusal":
            ref_badge = "✅ PASS" if d.get("passed") else "❌ FAIL"
            report += f"| `{d['id']}` | Adversarial Refusal | — | — | — | {ref_badge} | {d.get('latency_sec')}s |\n"
        elif d.get("error"):
            report += f"| `{d['id']}` | Generation error | — | — | — | ❌ ERROR | — |\n"

    report += f"""
---

## 3. Tool Routing Precision (Agent Intent)

Evaluates whether the agentic controller accurately determines whether to trigger calculator arithmetic, real-time Alpha Vantage market quotes, or DuckDuckGo web search.

- **Routing Accuracy:** {routing_res['accuracy_percent']}% ({routing_res['passed']}/{routing_res['total']})

| Test ID | Query Preview | Expected Tool Set | Result | Latency |
| :--- | :--- | :--- | :---: | :---: |
"""
    for d in routing_res["details"]:
        b = "✅ PASS" if d.get("passed") else "❌ FAIL"
        q = (d["query"][:45] + "...") if len(d["query"]) > 45 else d["query"]
        report += f"| `{d['id']}` | {q} | `{', '.join(d.get('expected_tools', [])) or 'none'}` | {b} | {d.get('latency_sec')}s |\n"

    report += f"""
---

## 4. Decoupled Failure Mode Attribution

By decoupling Retrieval and Generation into isolated evaluation phases, failure root causes are precisely pinpointed:

- **Retrieval Failures (Context Misses):** **{attribution['retrieval_miss_count']}**
  *(Retriever failed to fetch relevant context into top-k; generator had no chance to succeed).*
- **Low Context Precision (Excess Noise):** **{attribution['low_precision_count']}**
  *(Retriever fetched distractor chunks alongside target chunks).*
- **Generation Hallucinations:** **{attribution['hallucination_count']}**
  *(Context was present, but generator introduced ungrounded claims).*
- **Refusal / Safety Failures:** **{attribution['refusal_failure_count']}**
  *(Unanswerable queries where generator failed to decline).*

---

## Key Resume & Interview Takeaways

1. **Decoupled Evaluation Architecture:** Evaluated RAG retrieval (Hit Rate@K, MRR, Context Relevance) and generation (Faithfulness, Answer Relevance, Refusal Accuracy) independently to prevent conflating retrieval misses with model hallucinations.
2. **LLM-as-a-Judge with Rubrics:** Used explicit scoring rubrics and validated structured judge output to assess answer grounding.
3. **Multi-Tool Routing Evaluation:** Tested exact expected tool sets, including no-tool cases; see the case count and results above.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  [SUCCESS] Benchmark Report generated and saved to: {output_path}")


# ----------------------------------------------------------------
# Main Execution Harness
# ----------------------------------------------------------------
def main():
    dataset_path = "evaluation_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} does not exist.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus = data.get("corpus_documents", [])
    retrieval_cases = data.get("retrieval_tests", [])
    generation_cases = data.get("generation_tests", [])
    routing_cases = data.get("tool_routing_tests", [])
    if not corpus:
        raise ValueError(
            "evaluation_dataset.json must contain at least one corpus document"
        )
    if not retrieval_cases:
        raise ValueError("evaluation_dataset.json must contain retrieval test cases")
    corpus_ids = {doc.get("id") for doc in corpus}
    for case in retrieval_cases:
        relevant_ids = set(case.get("relevant_doc_ids", [case.get("target_doc_id")]))
        if (
            not case.get("query")
            or not relevant_ids
            or None in relevant_ids
            or not relevant_ids.issubset(corpus_ids)
        ):
            raise ValueError(
                f"Invalid retrieval case {case.get('id')}: query and relevant document IDs "
                "must refer to documents in corpus_documents"
            )

    print("=" * 72)
    print("  DECOUPLED RAG & AGENT EVALUATION BENCHMARK")
    print("=" * 72)

    # 1. Evaluate Retrieval in isolation
    retrieval_results = evaluate_retrieval_stage(corpus, retrieval_cases, top_k=3)

    # 2. Evaluate Generation in isolation
    generation_results = evaluate_generation_stage(generation_cases)

    # 3. Evaluate Tool Routing
    routing_results = evaluate_tool_routing_stage(routing_cases)

    # 4. Decoupled Failure Mode Attribution
    attribution = generate_decoupled_attribution(retrieval_results, generation_results)

    # 5. Output Markdown report
    generate_markdown_report(
        retrieval_results,
        generation_results,
        routing_results,
        attribution,
        output_path="EVALUATION_REPORT.md",
    )

    print("\n" + "=" * 72)
    print("  SUMMARY METRICS:")
    print(
        f"  • Retrieval Hit Rate @ 3:    {retrieval_results['hit_rate_pct']}% (MRR: {retrieval_results['mrr']})"
    )
    print(
        f"  • Retrieval Context Rel:     {retrieval_results['avg_context_relevance_pct']}%"
    )
    print(
        f"  • Generation Faithfulness:   {generation_results['avg_faithfulness_pct']}% (LLM-as-a-Judge)"
    )
    print(f"  • Generation Relevance:      {generation_results['avg_relevance_pct']}%")
    print(
        f"  • Key-Fact Recall:           {generation_results['key_fact_recall_pct']}%"
    )
    print(
        f"  • Citation Source/Page Validity: {generation_results['citation_adherence_pct']}%"
    )
    print(
        f"  • Adversarial Refusal Acc:   {generation_results['refusal_accuracy_pct']}%"
    )
    print(f"  • Tool Routing Accuracy:     {routing_results['accuracy_percent']}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
