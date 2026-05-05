"""
Extract proof dependencies from verified Lean 4 code.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Iterable, List

from backend.shared.models import MathlibLemmaHint, ProofDependency

if TYPE_CHECKING:
    from backend.autonomous.agents.lemma_search_agent import MathlibLemmaSearchAgent

logger = logging.getLogger(__name__)

_DOTTED_NAME_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_']*(?:\.[A-Za-z][A-Za-z0-9_']*)+\b")


class ProofDependencyExtractor:
    """Best-effort dependency extraction for verified proofs."""

    @staticmethod
    def _contains_symbol(search_text: str, symbol: str) -> bool:
        if not search_text or not symbol:
            return False
        pattern = re.compile(rf"(?<![A-Za-z0-9_'.]){re.escape(symbol)}(?![A-Za-z0-9_'])")
        return bool(pattern.search(search_text))

    @staticmethod
    def _dependency_source_ref(file_path: str, line_number: int) -> str:
        if file_path and line_number:
            return f"{file_path}:{line_number}"
        return file_path or ""

    @staticmethod
    def _dedupe_dependencies(dependencies: Iterable[ProofDependency]) -> List[ProofDependency]:
        seen = set()
        deduped: List[ProofDependency] = []
        for dependency in dependencies:
            key = (dependency.kind, dependency.name, dependency.source_ref)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(dependency)
        return deduped

    @staticmethod
    def _extract_dotted_names(lean_code: str) -> List[str]:
        return sorted(set(_DOTTED_NAME_RE.findall(lean_code or "")))

    @staticmethod
    def _strip_comments_and_strings(lean_code: str) -> str:
        if not lean_code:
            return ""

        result: list[str] = []
        index = 0
        length = len(lean_code)
        block_comment_depth = 0
        in_string = False

        while index < length:
            current = lean_code[index]
            pair = lean_code[index:index + 2]

            if block_comment_depth > 0:
                if pair == "/-":
                    block_comment_depth += 1
                    result.extend([" ", " "])
                    index += 2
                    continue
                if pair == "-/":
                    block_comment_depth -= 1
                    result.extend([" ", " "])
                    index += 2
                    continue
                result.append("\n" if current == "\n" else " ")
                index += 1
                continue

            if in_string:
                if current == "\\" and index + 1 < length:
                    result.extend([" ", " "])
                    index += 2
                    continue
                if current == "\"":
                    in_string = False
                    result.append(" ")
                    index += 1
                    continue
                result.append("\n" if current == "\n" else " ")
                index += 1
                continue

            if pair == "/-":
                block_comment_depth = 1
                result.extend([" ", " "])
                index += 2
                continue

            if pair == "--":
                while index < length and lean_code[index] != "\n":
                    result.append(" ")
                    index += 1
                continue

            if current == "\"":
                in_string = True
                result.append(" ")
                index += 1
                continue

            result.append(current)
            index += 1

        return "".join(result)

    @classmethod
    def _extract_search_text(cls, lean_code: str) -> str:
        sanitized = cls._strip_comments_and_strings(lean_code)
        if not sanitized:
            return ""

        proof_match = re.search(r":=\s*by\b(?P<body>.*)", sanitized, flags=re.DOTALL)
        if proof_match:
            return proof_match.group("body")

        theorem_by_match = re.search(
            r"^\s*(?:protected\s+)?(?:theorem|lemma)\b[^\n]*\bby\b(?P<body>.*)",
            sanitized,
            flags=re.DOTALL | re.MULTILINE,
        )
        if theorem_by_match:
            return theorem_by_match.group("body")

        return sanitized

    async def extract_dependencies(
        self,
        *,
        lean_code: str,
        theorem_name: str,
        proof_database,
        lemma_search_agent: "MathlibLemmaSearchAgent",
        relevant_lemmas: Iterable[MathlibLemmaHint] = (),
        current_proof_id: str = "",
    ) -> List[ProofDependency]:
        dependencies: List[ProofDependency] = []
        search_text = self._extract_search_text(lean_code)
        if not search_text:
            return dependencies

        try:
            existing_proofs = await proof_database.get_all_proofs()
            for proof in existing_proofs:
                if current_proof_id and proof.proof_id == current_proof_id:
                    continue
                candidate_name = str(proof.theorem_name or "").strip()
                if not candidate_name or candidate_name == theorem_name:
                    continue
                if not self._contains_symbol(search_text, candidate_name):
                    continue
                dependencies.append(
                    ProofDependency(
                        kind="moto",
                        name=candidate_name,
                        source_ref=proof.proof_id,
                    )
                )
        except Exception as exc:
            logger.debug("Failed to extract MOTO proof ancestry for %s: %s", theorem_name or "[unnamed theorem]", exc)

        mathlib_index_ready = False
        try:
            mathlib_index_ready = await lemma_search_agent.ensure_index_loaded()
        except Exception as exc:
            logger.debug("Mathlib dependency extraction skipped: %s", exc)

        if not mathlib_index_ready:
            return self._dedupe_dependencies(dependencies)

        for hint in relevant_lemmas or []:
            short_name = (hint.full_name or hint.requested_name).split(".")[-1]
            if not (
                self._contains_symbol(search_text, hint.full_name)
                or self._contains_symbol(search_text, short_name)
            ):
                continue
            dependencies.append(
                ProofDependency(
                    kind="mathlib",
                    name=hint.full_name or hint.requested_name,
                    source_ref=self._dependency_source_ref(hint.file_path, hint.line_number),
                )
            )

        for dotted_name in self._extract_dotted_names(search_text):
            for hit in lemma_search_agent.lookup_candidate_name(dotted_name):
                dependencies.append(
                    ProofDependency(
                        kind="mathlib",
                        name=hit.full_name or hit.requested_name,
                        source_ref=self._dependency_source_ref(hit.file_path, hit.line_number),
                    )
                )

        return self._dedupe_dependencies(dependencies)
