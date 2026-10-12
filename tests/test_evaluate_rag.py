from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.documents import Document
from ragas import EvaluationDataset

import evaluate_rag


def test_build_ragas_dataset_shapes_retrieval_and_generation_cases(monkeypatch):
    class FakeVectorStore:
        @classmethod
        def from_documents(cls, documents, _embeddings):
            assert len(documents) == 1
            return cls()

        def similarity_search(self, _query, k):
            assert k == evaluate_rag.TOP_K
            return [
                Document(
                    page_content="Revenue was $10.",
                    metadata={
                        "doc_id": "doc1",
                        "source": "report.pdf",
                        "human_page": 4,
                    },
                )
            ]

    monkeypatch.setattr(evaluate_rag, "FAISS", FakeVectorStore)
    monkeypatch.setattr(
        evaluate_rag,
        "embeddings",
        object(),
    )
    monkeypatch.setattr(
        evaluate_rag,
        "llm",
        MagicMock(invoke=MagicMock(return_value=MagicMock(content="Revenue was $10. [report.pdf, Page 4]"))),
    )

    dataset, details = evaluate_rag.build_ragas_dataset(
        corpus=[
            {
                "id": "doc1",
                "source": "report.pdf",
                "page": 4,
                "content": "Revenue was $10.",
            }
        ],
        retrieval_cases=[
            {
                "id": "retrieval_1",
                "query": "What was revenue?",
                "ground_truth_fact": "Revenue was $10.",
            }
        ],
        generation_cases=[
            {
                "id": "generation_1",
                "category": "generation_grounding",
                "query": "What was revenue?",
                "reference": "Revenue was $10.",
                "unanswerable": False,
            }
        ],
    )

    samples = dataset.to_list()
    assert len(samples) == 2
    assert samples[0] == {
        "user_input": "What was revenue?",
        "retrieved_contexts": ["[report.pdf, Page 4] Revenue was $10."],
        "response": "Revenue was $10. [report.pdf, Page 4]",
        "reference": "Revenue was $10.",
    }
    assert details[0]["test_id"] == "retrieval_1"
    assert details[1]["test_id"] == "generation_1"


def test_ragas_evaluation_scores_each_case_with_four_metrics(monkeypatch):
    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc_value, _traceback):
            return None

    class FakeScorer:
        value = 0.0

        def __init__(self, **_kwargs):
            pass

        async def ascore(self, **kwargs):
            return SimpleNamespace(value=self.value)

    class PrecisionScorer(FakeScorer):
        value = 0.1

    class RecallScorer(FakeScorer):
        value = 0.2

    class FaithfulnessScorer(FakeScorer):
        value = 0.3

    class RelevancyScorer(FakeScorer):
        value = 0.4

    monkeypatch.setattr(evaluate_rag, "AsyncOpenAI", FakeAsyncClient)
    monkeypatch.setattr(
        evaluate_rag, "llm_factory", lambda *_args, **_kwargs: "llm"
    )
    monkeypatch.setattr(
        evaluate_rag, "embedding_factory", lambda *_args, **_kwargs: "embeddings"
    )
    monkeypatch.setattr(evaluate_rag, "ContextPrecision", PrecisionScorer)
    monkeypatch.setattr(evaluate_rag, "ContextRecall", RecallScorer)
    monkeypatch.setattr(evaluate_rag, "Faithfulness", FaithfulnessScorer)
    monkeypatch.setattr(evaluate_rag, "AnswerRelevancy", RelevancyScorer)

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": "question",
                "retrieved_contexts": ["context"],
                "response": "answer",
                "reference": "reference",
            }
        ]
    )
    scores = evaluate_rag.run_ragas_evaluation(dataset)

    assert scores == [
        {
            "context_precision": 0.1,
            "context_recall": 0.2,
            "faithfulness": 0.3,
            "answer_relevancy": 0.4,
        }
    ]


def test_write_report_records_per_case_scores(tmp_path):
    scores = [
        {
            "context_precision": 1.0,
            "context_recall": 1.0,
            "faithfulness": 0.8,
            "answer_relevancy": 0.9,
        },
        {
            "context_precision": 0.5,
            "context_recall": 1.0,
            "faithfulness": 1.0,
            "answer_relevancy": 0.7,
        },
    ]
    details = [
        {
            "test_id": "case1",
            "category": "retrieval",
            "user_input": "question 1",
            "reference": "answer 1",
            "response": "response 1",
            "retrieved_contexts": ["context 1"],
            "retrieval_latency_ms": 10.0,
            "generation_latency_sec": 0.2,
            "retrieved_doc_ids": ["doc1"],
            "unanswerable": False,
        },
        {
            "test_id": "case2",
            "category": "generation_refusal",
            "user_input": "question 2",
            "reference": "answer 2",
            "response": "response 2",
            "retrieved_contexts": ["context 2"],
            "retrieval_latency_ms": 15.0,
            "generation_latency_sec": 0.3,
            "retrieved_doc_ids": ["doc2"],
            "unanswerable": True,
        },
    ]
    report = tmp_path / "report.md"
    csv = tmp_path / "details.csv"

    evaluate_rag.write_report(
        scores,
        details,
        corpus_count=5,
        report_path=str(report),
        csv_path=str(csv),
    )

    assert report.is_file()
    assert "context_precision" in report.read_text(encoding="utf-8")
    assert csv.is_file()
    assert "test_id" in csv.read_text(encoding="utf-8")
