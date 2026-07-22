from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from backend.shared.models import ContextPack
from backend.shared.utils import count_tokens


@dataclass(frozen=True)
class SourceTaggedDocument:
    source: str
    text: str


@dataclass
class SourceTaggedRagIndex:
    """Small deterministic RAG fake that preserves source exclusion semantics."""

    documents: list[SourceTaggedDocument] = field(default_factory=list)
    calls: list[dict[str, object]] = field(default_factory=list)

    def add(self, source: str, text: str) -> None:
        self.documents.append(SourceTaggedDocument(source=source, text=text))

    async def retrieve(
        self,
        *,
        query: str,
        max_tokens: int,
        exclude_sources: Iterable[str] | None = None,
        chunk_size: int | None = None,
        **_: object,
    ) -> ContextPack:
        excluded = set(exclude_sources or ())
        self.calls.append(
            {
                "query": query,
                "max_tokens": max_tokens,
                "chunk_size": chunk_size,
                "exclude_sources": tuple(sorted(excluded)),
            }
        )

        selected: list[SourceTaggedDocument] = []
        used_tokens = 0
        for document in self.documents:
            if document.source in excluded:
                continue
            rendered = f"[SOURCE: {document.source}]\n{document.text}"
            document_tokens = count_tokens(rendered)
            if selected and used_tokens + document_tokens > max_tokens:
                continue
            if not selected and document_tokens > max_tokens:
                continue
            selected.append(document)
            used_tokens += document_tokens

        text = "\n\n".join(
            f"[SOURCE: {document.source}]\n{document.text}" for document in selected
        )
        return ContextPack(
            text=text,
            evidence=[
                {"source": document.source, "text": document.text}
                for document in selected
            ],
            source_map={
                str(index): document.source
                for index, document in enumerate(selected)
            },
            coverage=1.0 if selected else 0.0,
            answerability=1.0 if selected else 0.0,
        )

    async def retrieve_for_mode(
        self,
        *,
        query: str,
        mode: str,
        max_tokens: int | None = None,
        exclude_sources: Iterable[str] | None = None,
        role_context_window: int | None = None,
        role_max_output_tokens: int | None = None,
        **kwargs: object,
    ) -> ContextPack:
        del mode, role_context_window, role_max_output_tokens
        return await self.retrieve(
            query=query,
            max_tokens=max_tokens or 100_000,
            exclude_sources=exclude_sources,
            **kwargs,
        )
