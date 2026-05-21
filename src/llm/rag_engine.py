"""Lightweight local RAG retrieval for simulated healthcare AI rules.

Step 20 retrieves internal project rules from local Markdown documents. It does
not call an LLM, search the internet, generate medical advice, or use paid APIs.
The retrieval output is intended to ground later explanations and action logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_KNOWLEDGE_BASE_DIR = Path("knowledge_base")
DEFAULT_KNOWLEDGE_FILES = {
    "safety_rules.md": """# Safety Rules

## Simulation Boundary

- This is a simulated healthcare AI engineering project.
- The system must not diagnose, recommend treatment, or replace clinicians.
- Human review is required for safety-sensitive or uncertain alerts.
- Critical alerts must never be suppressed.
""",
    "alert_escalation_rules.md": """# Alert Escalation Rules

## Escalation Rules

- Critical and immediate alerts require escalation in the simulation.
- High alerts require urgent review.
- Medium and low alerts may be monitored, grouped, or reviewed depending on context.
- Uncertain or noisy alerts require manual verification.
""",
    "workflow_rules.md": """# Workflow Rules

## Workflow Rules

- Alerts may move through triage, nurse review, clinician review, escalated review, and closed states.
- Repeated low-value alerts can increase fatigue and workflow burden.
- Response delays and ignored alerts should feed reliability monitoring.
""",
    "system_limitations.md": """# System Limitations

## Limitations

- The data is simulated only and not clinically validated.
- The system is not for real patient use.
- Drift, alert fatigue, and LLM uncertainty can reduce reliability.
- LLM outputs can be incomplete and must be constrained to internal rules.
""",
}

SIMULATION_ONLY_NOTE = (
    "Local project retrieval only; not clinical advice and not a medical device."
)


@dataclass
class RetrievedContext:
    """One retrieved knowledge-base context chunk."""

    source_file: str
    section_title: str
    text: str
    score: float


class RAGEngine:
    """Deterministic TF-IDF retriever over local project knowledge-base docs."""

    def __init__(
        self,
        knowledge_base_dir: str | Path = DEFAULT_KNOWLEDGE_BASE_DIR,
        max_contexts: int = 3,
    ) -> None:
        self.knowledge_base_dir = _resolve_project_path(knowledge_base_dir)
        self.max_contexts = int(max_contexts)
        self.documents: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.index_matrix = None

    def load_documents(self) -> list[dict[str, Any]]:
        """Load local Markdown knowledge-base documents."""
        ensure_default_knowledge_base(self.knowledge_base_dir)
        documents: list[dict[str, Any]] = []
        for path in sorted(self.knowledge_base_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                documents.append(
                    {
                        "source_file": path.name,
                        "path": str(path),
                        "text": text,
                    }
                )
        self.documents = documents
        return documents

    def split_into_chunks(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split Markdown documents into readable section chunks."""
        chunks: list[dict[str, Any]] = []
        for document in documents:
            source_file = str(document["source_file"])
            current_title = "Overview"
            current_lines: list[str] = []

            for line in str(document["text"]).splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    if current_lines:
                        chunks.append(
                            _make_chunk(source_file, current_title, current_lines)
                        )
                        current_lines = []
                    current_title = stripped.lstrip("#").strip() or "Untitled Section"
                else:
                    current_lines.append(line)

            if current_lines:
                chunks.append(_make_chunk(source_file, current_title, current_lines))

        self.chunks = [
            chunk
            for chunk in chunks
            if len(chunk["text"].strip()) >= 20
        ]
        return self.chunks

    def build_index(self) -> None:
        """Build a local TF-IDF index over knowledge-base chunks."""
        documents = self.load_documents()
        chunks = self.split_into_chunks(documents)
        if not chunks:
            self.vectorizer = None
            self.index_matrix = None
            return

        corpus = [
            f"{chunk['section_title']} {chunk['text']}"
            for chunk in chunks
        ]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.index_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedContext]:
        """Retrieve relevant local contexts for a query."""
        if self.vectorizer is None or self.index_matrix is None or not self.chunks:
            self.build_index()

        top_k = int(top_k or self.max_contexts)
        if top_k < 1:
            return []
        if self.vectorizer is None or self.index_matrix is None or not self.chunks:
            return []

        clean_query = query.strip() or "simulation safety limitations human review"
        query_vector = self.vectorizer.transform([clean_query])
        scores = cosine_similarity(query_vector, self.index_matrix).flatten()
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), self.chunks[index]["source_file"]),
        )

        contexts = [
            self._context_from_chunk(index, float(scores[index]))
            for index in ranked_indices[:top_k]
            if float(scores[index]) > 0.01
        ]

        if not contexts:
            contexts = self._fallback_contexts(top_k)
        elif len(contexts) < top_k:
            existing = {(context.source_file, context.section_title) for context in contexts}
            for fallback in self._fallback_contexts(top_k):
                key = (fallback.source_file, fallback.section_title)
                if key not in existing:
                    contexts.append(fallback)
                    existing.add(key)
                if len(contexts) >= top_k:
                    break

        return contexts[:top_k]

    def build_grounded_context(self, query: str, top_k: int = 3) -> str:
        """Build readable grounded context with source markers."""
        contexts = self.retrieve(query, top_k=top_k)
        if not contexts:
            return (
                "[No source available]\n"
                "No local project context was found. Preserve simulation-only safety limits "
                "and human review."
            )

        sections = []
        for context in contexts:
            sections.append(
                f"[Source: {context.source_file} | Section: {context.section_title} | "
                f"Score: {context.score:.4f}]\n{context.text}"
            )
        return "\n\n".join(sections)

    def get_source_summary(
        self,
        contexts: list[RetrievedContext],
    ) -> list[dict[str, Any]]:
        """Return source metadata for retrieved contexts."""
        return [
            {
                "source_file": context.source_file,
                "section_title": context.section_title,
                "score": round(float(context.score), 4),
            }
            for context in contexts
        ]

    def explain_retrieval(self, query: str) -> dict[str, Any]:
        """Return an explainable retrieval summary for demos/tests."""
        contexts = self.retrieve(query, top_k=self.max_contexts)
        return {
            "query": query,
            "retrieval_method": "tfidf_cosine_similarity",
            "knowledge_base_dir": str(self.knowledge_base_dir),
            "documents_loaded": len(self.documents),
            "chunks_indexed": len(self.chunks),
            "contexts_returned": len(contexts),
            "sources": self.get_source_summary(contexts),
            "simulation_only_note": SIMULATION_ONLY_NOTE,
        }

    def _context_from_chunk(self, index: int, score: float) -> RetrievedContext:
        chunk = self.chunks[index]
        return RetrievedContext(
            source_file=str(chunk["source_file"]),
            section_title=str(chunk["section_title"]),
            text=str(chunk["text"]),
            score=round(max(float(score), 0.0), 4),
        )

    def _fallback_contexts(self, top_k: int) -> list[RetrievedContext]:
        fallback_order = [
            "safety_rules.md",
            "system_limitations.md",
            "alert_escalation_rules.md",
            "workflow_rules.md",
        ]
        fallback_chunks = [
            chunk
            for source_file in fallback_order
            for chunk in self.chunks
            if chunk["source_file"] == source_file
        ]
        return [
            RetrievedContext(
                source_file=str(chunk["source_file"]),
                section_title=str(chunk["section_title"]),
                text=str(chunk["text"]),
                score=0.0,
            )
            for chunk in fallback_chunks[:top_k]
        ]


def ensure_default_knowledge_base(
    knowledge_base_dir: str | Path = DEFAULT_KNOWLEDGE_BASE_DIR,
) -> None:
    """Create missing default knowledge-base files with safe project content."""
    directory = _resolve_project_path(knowledge_base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for filename, content in DEFAULT_KNOWLEDGE_FILES.items():
        path = directory / filename
        if not path.exists():
            path.write_text(content.strip() + "\n", encoding="utf-8")


def create_rag_engine(
    knowledge_base_dir: str | Path = DEFAULT_KNOWLEDGE_BASE_DIR,
) -> RAGEngine:
    """Create and index the default RAG engine."""
    engine = RAGEngine(knowledge_base_dir=knowledge_base_dir)
    engine.build_index()
    return engine


def run_rag_demo() -> None:
    """Run a small local retrieval demo without LLM or internet calls."""
    query = "Why should a critical simulated alert require human review?"
    engine = create_rag_engine()
    contexts = engine.retrieve(query)
    print("Retrieved sources:")
    for source in engine.get_source_summary(contexts):
        print(f"  {source['source_file']} | {source['section_title']} | score={source['score']}")

    print("\nGrounded context:")
    print(engine.build_grounded_context(query))

    print("\nRetrieval explanation:")
    explanation = engine.explain_retrieval(query)
    for key, value in explanation.items():
        print(f"{key}: {value}")


def retrieve_context(query: str = "simulation safety human review") -> list[RetrievedContext]:
    """Compatibility wrapper for older placeholder imports."""
    return create_rag_engine().retrieve(query)


def _make_chunk(
    source_file: str,
    section_title: str,
    lines: list[str],
) -> dict[str, Any]:
    text = "\n".join(line.strip() for line in lines if line.strip()).strip()
    return {
        "source_file": source_file,
        "section_title": section_title,
        "text": text,
    }


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[2] / candidate


if __name__ == "__main__":
    run_rag_demo()
