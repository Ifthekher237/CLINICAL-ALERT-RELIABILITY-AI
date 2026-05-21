"""Focused tests for Step 20 local RAG engine."""

from __future__ import annotations

from pathlib import Path

from src.llm.rag_engine import (
    DEFAULT_KNOWLEDGE_FILES,
    RAGEngine,
    RetrievedContext,
    create_rag_engine,
    ensure_default_knowledge_base,
)


def test_default_knowledge_base_files_exist_or_are_created(tmp_path: Path) -> None:
    kb_dir = tmp_path / "knowledge_base"
    ensure_default_knowledge_base(kb_dir)

    for filename in DEFAULT_KNOWLEDGE_FILES:
        assert (kb_dir / filename).exists()


def test_documents_load_successfully(tmp_path: Path) -> None:
    kb_dir = tmp_path / "knowledge_base"
    ensure_default_knowledge_base(kb_dir)
    engine = RAGEngine(knowledge_base_dir=kb_dir)
    documents = engine.load_documents()

    assert len(documents) == 4
    assert {"source_file", "text"}.issubset(documents[0])


def test_chunks_are_created(tmp_path: Path) -> None:
    kb_dir = tmp_path / "knowledge_base"
    ensure_default_knowledge_base(kb_dir)
    engine = RAGEngine(knowledge_base_dir=kb_dir)
    documents = engine.load_documents()
    chunks = engine.split_into_chunks(documents)

    assert chunks
    assert {"source_file", "section_title", "text"}.issubset(chunks[0])


def test_index_builds_successfully(tmp_path: Path) -> None:
    kb_dir = tmp_path / "knowledge_base"
    ensure_default_knowledge_base(kb_dir)
    engine = RAGEngine(knowledge_base_dir=kb_dir)
    engine.build_index()

    assert engine.vectorizer is not None
    assert engine.index_matrix is not None
    assert len(engine.chunks) > 0


def test_retrieve_returns_retrieved_context_objects(tmp_path: Path) -> None:
    kb_dir = tmp_path / "knowledge_base"
    ensure_default_knowledge_base(kb_dir)
    engine = create_rag_engine(kb_dir)
    contexts = engine.retrieve("critical alert human review", top_k=2)

    assert contexts
    assert all(isinstance(context, RetrievedContext) for context in contexts)
    assert all(context.source_file for context in contexts)
    assert all(isinstance(context.score, float) for context in contexts)


def test_build_grounded_context_returns_source_markers(tmp_path: Path) -> None:
    kb_dir = tmp_path / "knowledge_base"
    ensure_default_knowledge_base(kb_dir)
    engine = create_rag_engine(kb_dir)
    grounded = engine.build_grounded_context("critical simulated alert review", top_k=2)

    assert "[Source:" in grounded
    assert "Score:" in grounded
    assert "critical" in grounded.lower() or "review" in grounded.lower()


def test_source_summary_returns_filenames(tmp_path: Path) -> None:
    kb_dir = tmp_path / "knowledge_base"
    ensure_default_knowledge_base(kb_dir)
    engine = create_rag_engine(kb_dir)
    contexts = engine.retrieve("workflow burden response delay", top_k=2)
    summary = engine.get_source_summary(contexts)

    assert summary
    assert {"source_file", "section_title", "score"}.issubset(summary[0])
    assert summary[0]["source_file"].endswith(".md")


def test_no_internet_llm_or_ollama_required(tmp_path: Path) -> None:
    kb_dir = tmp_path / "knowledge_base"
    ensure_default_knowledge_base(kb_dir)
    engine = RAGEngine(knowledge_base_dir=kb_dir)
    engine.build_index()
    contexts = engine.retrieve("model drift and alert fatigue")

    assert contexts


def test_safety_query_retrieves_safety_or_limitation_content() -> None:
    engine = create_rag_engine("knowledge_base")
    contexts = engine.retrieve(
        "Why should a critical simulated alert require human review?",
        top_k=3,
    )
    source_files = {context.source_file for context in contexts}
    combined_text = " ".join(context.text.lower() for context in contexts)

    assert source_files.intersection({"safety_rules.md", "system_limitations.md", "alert_escalation_rules.md"})
    assert "human review" in combined_text or "critical" in combined_text


def test_explain_retrieval_is_readable_and_deterministic(tmp_path: Path) -> None:
    kb_dir = tmp_path / "knowledge_base"
    ensure_default_knowledge_base(kb_dir)
    engine = create_rag_engine(kb_dir)
    explanation = engine.explain_retrieval("LLM outputs must be constrained")

    assert explanation["retrieval_method"] == "tfidf_cosine_similarity"
    assert explanation["contexts_returned"] > 0
    assert "simulation_only_note" in explanation
