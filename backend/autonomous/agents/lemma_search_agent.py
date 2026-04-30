"""
Mathlib lemma search agent for Lean 4 proof generation.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from backend.autonomous.prompts.proof_prompts import build_lemma_search_prompt
from backend.shared.api_client_manager import api_client_manager
from backend.shared.json_parser import parse_json
from backend.shared.lean4_client import get_lean4_client
from backend.shared.models import MathlibLemmaHint, ProofCandidate
from backend.shared.openrouter_client import FreeModelExhaustedError
from backend.shared.utils import count_tokens

logger = logging.getLogger(__name__)

_NAMESPACE_RE = re.compile(r"^\s*namespace\s+([A-Za-z0-9_'.]+)")
_END_RE = re.compile(r"^\s*end(?:\s+([A-Za-z0-9_'.]+))?\s*$")
_DECL_RE = re.compile(r"^\s*(?:protected\s+)?(?:theorem|lemma)\s+([A-Za-z0-9_']+)")


class MathlibLemmaSearchAgent:
    """Suggest and locally confirm relevant Mathlib lemmas for a theorem."""

    def __init__(
        self,
        model_id: str,
        context_window: int,
        max_output_tokens: int,
        role_id: str,
    ) -> None:
        self.model_id = model_id
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.role_id = role_id
        self.task_sequence = 0
        self._by_full_name: Dict[str, List[dict]] = {}
        self._by_short_name: Dict[str, List[dict]] = {}
        self._index_loaded = False

    def get_current_task_id(self) -> str:
        return f"proof_lemma_{self.task_sequence:03d}"

    @staticmethod
    def _build_source_excerpt(theorem_statement: str, source_content: str) -> str:
        statement = (theorem_statement or "").strip()
        content = source_content or ""
        if not content:
            return ""

        search_token = statement[:80]
        if search_token:
            match_index = content.find(search_token)
            if match_index >= 0:
                start = max(0, match_index - 2000)
                end = min(len(content), match_index + max(len(statement), 1) + 2000)
                return content[start:end]

        return content[:5000]

    @staticmethod
    def _dedupe_hits(hits: List[MathlibLemmaHint]) -> List[MathlibLemmaHint]:
        seen = set()
        deduped: List[MathlibLemmaHint] = []
        for hit in hits:
            key = (hit.full_name, hit.file_path, hit.line_number)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
        return deduped

    @staticmethod
    def _relative_mathlib_path(root: Path, file_path: Path) -> str:
        try:
            return str(file_path.relative_to(root))
        except ValueError:
            return str(file_path)

    def _register_index_entry(self, entry: dict) -> None:
        full_name = str(entry.get("full_name", "")).strip()
        short_name = str(entry.get("short_name", "")).strip()
        if full_name:
            self._by_full_name.setdefault(full_name, []).append(entry)
        if short_name:
            self._by_short_name.setdefault(short_name, []).append(entry)

    def _load_index_from_cache(self, index_path: Path) -> bool:
        if not index_path.exists():
            return False

        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            for entry in payload.get("entries", []) or []:
                if isinstance(entry, dict):
                    self._register_index_entry(entry)
            self._index_loaded = True
            return True
        except Exception as exc:
            logger.warning("Failed to load cached Mathlib declaration index: %s", exc)
            return False

    def _write_index_cache(self, index_path: Path) -> None:
        entries = []
        for entry_list in self._by_full_name.values():
            entries.extend(entry_list)
        index_path.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")

    def _build_index_from_sources(self, roots: List[Path], index_path: Path) -> None:
        entries: List[dict] = []
        for root in roots:
            for file_path in root.rglob("*.lean"):
                namespace_stack: List[str] = []
                try:
                    with file_path.open("r", encoding="utf-8") as handle:
                        for line_number, raw_line in enumerate(handle, start=1):
                            namespace_match = _NAMESPACE_RE.match(raw_line)
                            if namespace_match:
                                namespace_stack.append(namespace_match.group(1))
                                continue

                            end_match = _END_RE.match(raw_line)
                            if end_match:
                                label = (end_match.group(1) or "").strip()
                                if label and namespace_stack:
                                    current_full = ".".join(namespace_stack)
                                    current_leaf = namespace_stack[-1].split(".")[-1]
                                    if label == current_full or label == current_leaf:
                                        namespace_stack.pop()
                                continue

                            decl_match = _DECL_RE.match(raw_line)
                            if not decl_match:
                                continue

                            short_name = decl_match.group(1)
                            full_name = ".".join(namespace_stack + [short_name]) if namespace_stack else short_name
                            entries.append(
                                {
                                    "full_name": full_name,
                                    "short_name": short_name,
                                    "declaration": raw_line.strip(),
                                    "file_path": self._relative_mathlib_path(root, file_path),
                                    "line_number": line_number,
                                }
                            )
                except Exception as exc:
                    logger.debug("Skipping Mathlib file %s during index build: %s", file_path, exc)

        for entry in entries:
            self._register_index_entry(entry)
        self._write_index_cache(index_path)
        self._index_loaded = True
        logger.info("Built Mathlib declaration index with %s entries", len(entries))

    async def ensure_index_loaded(self) -> bool:
        if self._index_loaded:
            return True

        client = get_lean4_client()
        workspace_ready = await client.ensure_workspace()
        if not workspace_ready:
            return False

        roots = client.get_mathlib_source_roots()
        if not roots:
            return False

        index_path = client.get_mathlib_index_path()
        if self._load_index_from_cache(index_path):
            return True

        self._build_index_from_sources(roots, index_path)
        return self._index_loaded

    async def _ensure_index_loaded(self) -> bool:
        """Backward-compatible private alias for older callers."""
        return await self.ensure_index_loaded()

    def lookup_candidate_name(self, requested_name: str) -> List[MathlibLemmaHint]:
        requested_name = str(requested_name or "").strip()
        if not requested_name:
            return []

        matched_entries = list(self._by_full_name.get(requested_name, []))
        if not matched_entries:
            short_name = requested_name.split(".")[-1]
            matched_entries = list(self._by_short_name.get(short_name, []))

        hits = [
            MathlibLemmaHint(
                requested_name=requested_name,
                full_name=str(entry.get("full_name", "")).strip() or requested_name,
                declaration=str(entry.get("declaration", "")).strip(),
                file_path=str(entry.get("file_path", "")).strip(),
                line_number=int(entry.get("line_number", 0) or 0),
            )
            for entry in matched_entries[:3]
        ]
        return self._dedupe_hits(hits)

    def _lookup_candidate_name(self, requested_name: str) -> List[MathlibLemmaHint]:
        """Backward-compatible private alias for older callers."""
        return self.lookup_candidate_name(requested_name)

    async def suggest_relevant_lemmas(
        self,
        user_research_prompt: str,
        source_type: str,
        theorem_candidate: ProofCandidate,
        source_content: str,
        *,
        max_candidates: int = 8,
    ) -> List[MathlibLemmaHint]:
        """Return locally confirmed Mathlib hints for the target theorem."""
        if not await self.ensure_index_loaded():
            return []

        source_excerpt = theorem_candidate.source_excerpt or self._build_source_excerpt(
            theorem_candidate.statement,
            source_content,
        )
        prompt = build_lemma_search_prompt(
            user_prompt=user_research_prompt,
            source_type=source_type,
            theorem_statement=theorem_candidate.statement,
            formal_sketch=theorem_candidate.formal_sketch,
            source_excerpt=source_excerpt,
        )

        max_input_tokens = self.context_window - self.max_output_tokens
        prompt_tokens = count_tokens(prompt)
        while prompt_tokens > max_input_tokens and len(source_excerpt) > 1200:
            source_excerpt = source_excerpt[: max(len(source_excerpt) // 2, 1200)]
            prompt = build_lemma_search_prompt(
                user_prompt=user_research_prompt,
                source_type=source_type,
                theorem_statement=theorem_candidate.statement,
                formal_sketch=theorem_candidate.formal_sketch,
                source_excerpt=source_excerpt,
            )
            prompt_tokens = count_tokens(prompt)

        if prompt_tokens > max_input_tokens:
            return []

        task_id = self.get_current_task_id()
        self.task_sequence += 1

        try:
            response = await api_client_manager.generate_completion(
                task_id=task_id,
                role_id=self.role_id,
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                temperature=0.0,
            )
            if not response or not response.get("choices"):
                return []

            message = response["choices"][0].get("message", {})
            content = message.get("content") or message.get("reasoning") or ""
            if not content:
                return []

            data = parse_json(content)
            if isinstance(data, list):
                data = data[0] if data else {}

            raw_names = data.get("lemma_names") or data.get("lemmas") or []
            candidate_names = []
            for raw_name in raw_names:
                if isinstance(raw_name, dict):
                    value = raw_name.get("name") or raw_name.get("lemma_name")
                else:
                    value = raw_name
                value = str(value or "").strip()
                if value:
                    candidate_names.append(value)

            confirmed_hits: List[MathlibLemmaHint] = []
            for candidate_name in candidate_names:
                confirmed_hits.extend(self.lookup_candidate_name(candidate_name))
                confirmed_hits = self._dedupe_hits(confirmed_hits)
                if len(confirmed_hits) >= max_candidates:
                    break

            return confirmed_hits[:max_candidates]
        except FreeModelExhaustedError:
            raise
        except Exception as exc:
            logger.warning(
                "MathlibLemmaSearchAgent failed for theorem %s: %s",
                theorem_candidate.theorem_id,
                exc,
            )
            return []
