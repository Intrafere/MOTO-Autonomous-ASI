"""
Autonomous RAG Manager - Context management for autonomous research mode.
Wraps the aggregator RAG manager with autonomous-specific functionality.

CRITICAL: This manager follows the "DIRECT INJECTION FIRST, RAG SECOND" principle:
- Content that fits in context is directly injected
- Content that doesn't fit is retrieved via RAG semantic search
- NO truncation is used as fallback
"""
import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from backend.shared.config import system_config, rag_config
from backend.shared.utils import count_tokens
from backend.aggregator.core.rag_manager import rag_manager
from backend.autonomous.memory.brainstorm_memory import brainstorm_memory
from backend.autonomous.memory.paper_library import paper_library

logger = logging.getLogger(__name__)


class AutonomousRAGManager:
    """
    Manages RAG context for autonomous research mode.
    Handles per-brainstorm RAG databases and reference paper context.
    
    Follows the "DIRECT INJECTION FIRST, RAG SECOND" principle:
    - Tries direct injection first for all content
    - Falls back to RAG semantic search when content doesn't fit
    - NEVER truncates content (would lose important information)
    """
    
    def __init__(self):
        self._initialized = False
        self._current_topic_id: Optional[str] = None
        # Track which brainstorms have been indexed for RAG
        self._brainstorms_indexed: set = set()
        # Track which papers have been indexed for RAG
        self._papers_indexed: set = set()
    
    def reset(self) -> None:
        """
        Reset tracking state for a fresh session.
        
        CRITICAL: This resets the indexed tracking sets that would otherwise
        cause stale content to persist between sessions. Call this when
        starting a new autonomous research session or when RAG is cleared.
        """
        logger.info("Resetting AutonomousRAGManager tracking state...")
        self._brainstorms_indexed.clear()
        self._papers_indexed.clear()
        self._current_topic_id = None
        self._initialized = False  # Force re-initialization
        logger.info("AutonomousRAGManager state reset")
    
    async def initialize(self) -> None:
        """Initialize the autonomous RAG manager."""
        if self._initialized:
            return
        
        await brainstorm_memory.initialize()
        await paper_library.initialize()
        
        self._initialized = True
        logger.info("AutonomousRAGManager initialized")
    
    async def get_brainstorm_context(
        self,
        topic_id: str,
        max_tokens: int = 50000,
        query: str = "",
        exclude_sources: Optional[List[str]] = None
    ) -> Tuple[str, bool]:
        """
        Get brainstorm database content for context.
        
        Implements DIRECT INJECTION FIRST, RAG SECOND principle:
        - If content fits in max_tokens: returns full content (direct inject)
        - If content exceeds max_tokens: returns RAG-retrieved relevant chunks
        
        Args:
            topic_id: Topic ID to get context for
            max_tokens: Maximum tokens available for this content
            query: Query for RAG retrieval if needed (e.g., user research prompt)
            exclude_sources: Source names to skip during RAG packing
        
        Returns:
            Tuple of (content string, used_rag boolean)
        """
        content = await brainstorm_memory.get_database_content(topic_id, strip_proofs=True)
        
        if not content:
            return "", False
        
        # Count actual tokens
        content_tokens = count_tokens(content)
        
        if content_tokens <= max_tokens:
            # Content fits - use direct injection
            logger.debug(f"Brainstorm context: Direct injection ({content_tokens} tokens <= {max_tokens} max)")
            return content, False
        else:
            # Content doesn't fit - use RAG retrieval
            logger.info(f"Brainstorm context: Using RAG ({content_tokens} tokens > {max_tokens} max)")
            
            # Ensure brainstorm is indexed in RAG
            await self._ensure_brainstorm_indexed(topic_id, content)
            
            # Retrieve relevant chunks via RAG
            if not query:
                query = f"brainstorm topic {topic_id}"
            
            context_pack = await rag_manager.retrieve(
                query=query,
                chunk_size=rag_config.validator_chunk_size,  # 512 for consistency
                max_tokens=max_tokens,
                exclude_sources=exclude_sources
            )
            
            if context_pack and context_pack.text:
                logger.info(f"Brainstorm RAG: Retrieved {len(context_pack.evidence)} evidence chunks, "
                           f"coverage={context_pack.coverage:.2f}")
                return context_pack.text, True
            else:
                # RAG returned empty - use summary as fallback
                logger.warning("Brainstorm RAG returned empty - using summary fallback")
                summary = await self.get_brainstorm_summary(topic_id, max_chars=int(max_tokens * 4))
                return summary, True
    
    async def _ensure_brainstorm_indexed(self, topic_id: str, content: str) -> None:
        """Ensure brainstorm content is indexed in RAG for retrieval."""
        if topic_id in self._brainstorms_indexed:
            return
        
        try:
            source_name = f"brainstorm_{topic_id}"
            
            # Add to RAG with all 4 chunk sizes for flexibility
            await rag_manager.add_text(
                content,
                source_name,
                chunk_sizes=rag_config.submitter_chunk_intervals,
                is_permanent=False
            )
            
            self._brainstorms_indexed.add(topic_id)
            logger.info(f"Indexed brainstorm {topic_id} in RAG")
            
        except Exception as e:
            logger.error(f"Failed to index brainstorm {topic_id}: {e}")
    
    async def get_brainstorm_summary(
        self,
        topic_id: str,
        max_chars: int = 5000
    ) -> str:
        """
        Get a summary of brainstorm content for context.
        
        Args:
            topic_id: Topic ID to summarize
            max_chars: Maximum characters in summary
        
        Returns:
            Summary of brainstorm content
        """
        submissions = await brainstorm_memory.get_submissions_list(topic_id)
        
        if not submissions:
            return "No submissions yet."
        
        summary_parts = [f"Brainstorm has {len(submissions)} accepted submissions:"]
        
        chars_used = len(summary_parts[0])
        
        for i, sub in enumerate(submissions, 1):
            # Get preview of each submission
            content = sub.get("content", "")[:500]
            preview = f"\n\n{i}. {content}"
            
            if chars_used + len(preview) > max_chars:
                summary_parts.append(f"\n\n... and {len(submissions) - i + 1} more submissions")
                break
            
            summary_parts.append(preview)
            chars_used += len(preview)
        
        return "".join(summary_parts)
    
    async def get_reference_papers_context(
        self,
        paper_ids: List[str],
        max_total_tokens: int = 60000,
        query: str = "",
        include_outlines: bool = True,
        exclude_sources: Optional[List[str]] = None
    ) -> Tuple[str, bool]:
        """
        Get reference papers content for context.
        
        Implements DIRECT INJECTION FIRST, RAG SECOND principle:
        - If all papers fit in max_total_tokens: returns full content (direct inject)
        - If papers exceed max_total_tokens: returns RAG-retrieved relevant chunks
        
        Args:
            paper_ids: List of paper IDs to include
            max_total_tokens: Maximum tokens for all reference papers combined
            query: Query for RAG retrieval if needed
            exclude_sources: Source names to skip during RAG packing
        
        Returns:
            Tuple of (content string, used_rag boolean)
        """
        if not paper_ids:
            return "", False
        
        # First, collect all paper content
        papers_content = []
        total_tokens = 0
        
        for paper_id in paper_ids:
            content = await paper_library.get_paper_content(paper_id, strip_proofs=True)
            metadata = await paper_library.get_metadata(paper_id)
            
            if content and metadata:
                paper_tokens = count_tokens(content)
                papers_content.append({
                    "id": paper_id,
                    "title": metadata.title,
                    "content": content,
                    "tokens": paper_tokens
                })
                total_tokens += paper_tokens
        
        if not papers_content:
            return "", False
        
        # Check if all papers fit - use direct injection
        if total_tokens <= max_total_tokens:
            parts = ["REFERENCE PAPERS:\n" + "=" * 60]
            for paper in papers_content:
                parts.append(f"\n\n{'=' * 40}")
                parts.append(f"\nPaper: {paper['title']}")
                parts.append(f"\n{'=' * 40}")
                
                # Include outline if requested
                if include_outlines:
                    outline = await paper_library.get_outline(paper['id'])
                    if outline:
                        parts.append(f"\n\nOUTLINE:\n{outline}\n")
                        parts.append(f"\n{'=' * 40}")
                
                parts.append(f"\n\n{paper['content']}")
            
            logger.debug(f"Reference papers: Direct injection ({total_tokens} tokens <= {max_total_tokens} max)")
            return "".join(parts), False
        
        # Papers don't fit - use RAG retrieval
        logger.info(f"Reference papers: Using RAG ({total_tokens} tokens > {max_total_tokens} max)")
        
        # Ensure all papers are indexed in RAG
        for paper in papers_content:
            await self._ensure_paper_indexed(paper["id"], paper["content"], paper["title"])
        
        # Retrieve relevant chunks via RAG
        if not query:
            query = "reference paper content mathematical research"
        
        # Get outline for context enhancement (use first paper's outline)
        outline = None
        if include_outlines and papers_content:
            outline = await paper_library.get_outline(papers_content[0]['id'])
        
        # Enhance query with outline context
        enhanced_query = query
        if outline:
            # Use first 500 chars of outline to enhance semantic search
            enhanced_query = f"{query}\n\nPaper structure: {outline[:500]}"
        
        context_pack = await rag_manager.retrieve(
            query=enhanced_query,
            chunk_size=rag_config.validator_chunk_size,
            max_tokens=max_total_tokens,
            exclude_sources=exclude_sources
        )
        
        if context_pack and context_pack.text:
            logger.info(f"Reference papers RAG: Retrieved {len(context_pack.evidence)} evidence chunks")
            
            # Include outline summary if available
            result_parts = [f"REFERENCE PAPERS (RAG Retrieved):\n{'=' * 60}"]
            if outline and include_outlines:
                result_parts.append(f"\n\nOUTLINE SUMMARY:\n{outline[:300]}...")
                result_parts.append(f"\n{'=' * 40}")
            result_parts.append(f"\n\n{context_pack.text}")
            
            return "".join(result_parts), True
        else:
            # RAG returned empty - return abstracts only
            logger.warning("Reference papers RAG returned empty - using abstracts fallback")
            parts = ["REFERENCE PAPERS (Abstracts Only):\n" + "=" * 60]
            for paper in papers_content:
                metadata = await paper_library.get_metadata(paper["id"])
                if metadata and metadata.abstract:
                    parts.append(f"\n\n- {paper['title']}")
                    parts.append(f"\n  Abstract: {metadata.abstract[:1000]}")
            return "".join(parts), True
    
    async def _ensure_paper_indexed(self, paper_id: str, content: str, title: str) -> None:
        """Ensure paper content is indexed in RAG for retrieval."""
        source_name = f"reference_paper_{paper_id}"
        has_document_entry = source_name in rag_manager.document_access_order
        has_validator_chunks = any(
            chunk.source_file == source_name
            for chunk in rag_manager.chunks_by_size[rag_config.validator_chunk_size]
        )

        if paper_id in self._papers_indexed and has_document_entry and has_validator_chunks:
            return
        
        try:
            # If the tracking set says this paper was indexed but its active RAG entry
            # has been evicted, remove any partial remnants and rebuild it.
            if paper_id in self._papers_indexed:
                self._papers_indexed.discard(paper_id)

            if has_document_entry:
                await rag_manager.remove_document(source_name)

            await rag_manager.add_text(
                content,
                source_name,
                chunk_sizes=rag_config.submitter_chunk_intervals,
                is_permanent=False
            )
            self._papers_indexed.add(paper_id)
            logger.debug(f"Indexed reference paper {paper_id}: {title}")
            
        except Exception as e:
            logger.error(f"Failed to index reference paper {paper_id}: {e}")
    
    async def get_all_papers_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all papers for topic selection context."""
        return await paper_library.get_papers_summary()
    
    async def get_all_brainstorms_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all brainstorms for topic selection context."""
        brainstorms = await brainstorm_memory.get_all_brainstorms()
        
        return [
            {
                "topic_id": b.topic_id,
                "topic_prompt": b.topic_prompt,
                "status": b.status,
                "submission_count": b.submission_count,
                "papers_generated": b.papers_generated,
                "created_at": b.created_at.isoformat() if b.created_at else None,
                "last_activity": b.last_activity.isoformat() if b.last_activity else None
            }
            for b in brainstorms
        ]
    
    async def prepare_compiler_context(
        self,
        topic_id: str,
        reference_paper_ids: List[str],
        current_outline: str,
        current_paper: str,
        context_budget: int = 100000,
        query: str = ""
    ) -> Dict[str, Any]:
        """
        Prepare context for paper compilation.
        
        Implements DIRECT INJECTION FIRST, RAG SECOND with priority ordering:
        1. Current outline (ALWAYS direct inject - never RAG, non-negotiable)
        2. Brainstorm database (direct if fits, else RAG)
        3. Current paper progress (direct if fits, else RAG)
        4. Reference papers (direct if fits, else RAG - lowest priority)
        
        NO TRUNCATION IS USED - content either fits (direct) or uses RAG.
        
        Args:
            topic_id: Source brainstorm topic ID
            reference_paper_ids: Selected reference paper IDs
            current_outline: Current paper outline (always injected)
            current_paper: Current paper progress
            context_budget: Total context budget in tokens
            query: Query for RAG retrieval if needed
        
        Returns:
            Dictionary with context components and allocation info
        """
        # Calculate available budget after outline (mandatory - NEVER RAGed)
        outline_tokens = count_tokens(current_outline)
        system_overhead = 5000  # Reserve for system prompts, JSON schema, etc.
        remaining_budget = context_budget - outline_tokens - system_overhead
        
        if remaining_budget <= 0:
            logger.error(f"Outline ({outline_tokens} tokens) + overhead ({system_overhead}) exceeds context budget ({context_budget})")
            raise ValueError(f"Context budget too small for outline. Need at least {outline_tokens + system_overhead} tokens.")
        
        # Initialize context structure
        context = {
            "outline": current_outline,  # ALWAYS fully injected - non-negotiable
            "brainstorm": "",
            "reference_papers": "",
            "current_paper": "",
            "use_rag_for_brainstorm": False,
            "use_rag_for_papers": False,
            "use_rag_for_reference": False
        }
        
        # RAG query for retrievals
        rag_query = query or f"mathematical research paper compilation"
        rag_exclude_sources: List[str] = []
        
        # Priority 1: Brainstorm database (highest priority after outline)
        brainstorm_budget = int(remaining_budget * 0.5)  # Allocate 50% to brainstorm
        brainstorm_content, used_rag = await self.get_brainstorm_context(
            topic_id,
            max_tokens=brainstorm_budget,
            query=rag_query
        )
        context["brainstorm"] = brainstorm_content
        context["use_rag_for_brainstorm"] = used_rag
        brainstorm_tokens = count_tokens(brainstorm_content)
        remaining_budget -= brainstorm_tokens

        # If brainstorm was direct-injected, exclude its RAG sources from later retrievals.
        if brainstorm_content and not used_rag:
            rag_exclude_sources.extend([
                f"brainstorm_{topic_id}",
                f"brainstorm_{topic_id}.txt"
            ])
        
        # Priority 2: Current paper progress
        paper_tokens = count_tokens(current_paper) if current_paper else 0
        paper_budget = int(remaining_budget * 0.7)  # Allocate 70% of remaining to paper
        
        if paper_tokens <= paper_budget and paper_tokens > 0:
            # Paper fits - direct injection
            context["current_paper"] = current_paper
            remaining_budget -= paper_tokens
            logger.debug(f"Compiler context: Paper direct injection ({paper_tokens} tokens)")
            rag_exclude_sources.append("compiler_current_paper")
        elif paper_tokens > 0:
            # Paper doesn't fit - use RAG
            context["use_rag_for_papers"] = True
            await self._ensure_paper_indexed_for_compiler(current_paper)
            
            paper_pack = await rag_manager.retrieve(
                query=rag_query,
                chunk_size=rag_config.validator_chunk_size,
                max_tokens=paper_budget,
                exclude_sources=list(dict.fromkeys(rag_exclude_sources)) if rag_exclude_sources else None
            )
            
            if paper_pack and paper_pack.text:
                context["current_paper"] = f"[CURRENT PAPER - RAG Retrieved]\n{paper_pack.text}"
                remaining_budget -= count_tokens(context["current_paper"])
                logger.info(f"Compiler context: Paper RAG ({len(paper_pack.evidence)} chunks)")
            else:
                # Keep last portion of paper as fallback
                last_chars = int(paper_budget * 4)
                context["current_paper"] = f"[CURRENT PAPER - Last Section]\n{current_paper[-last_chars:]}" if current_paper else ""
                remaining_budget -= count_tokens(context["current_paper"])
                logger.warning("Compiler context: Paper RAG empty, using last section")
        
        # Priority 3: Reference papers (lowest priority)
        if reference_paper_ids and remaining_budget > 2000:
            ref_content, ref_used_rag = await self.get_reference_papers_context(
                reference_paper_ids,
                max_total_tokens=remaining_budget,
                query=rag_query,
                exclude_sources=list(dict.fromkeys(rag_exclude_sources)) if rag_exclude_sources else None
            )
            context["reference_papers"] = ref_content
            context["use_rag_for_reference"] = ref_used_rag
        
        # Log final allocation
        logger.info(
            f"Prepared compiler context: outline={outline_tokens}t (direct), "
            f"brainstorm={brainstorm_tokens}t (RAG={context['use_rag_for_brainstorm']}), "
            f"paper={count_tokens(context['current_paper'])}t (RAG={context['use_rag_for_papers']}), "
            f"reference={count_tokens(context['reference_papers'])}t (RAG={context['use_rag_for_reference']})"
        )
        
        return context
    
    async def _ensure_paper_indexed_for_compiler(self, paper_content: str) -> None:
        """Ensure current paper is indexed in RAG for retrieval during compilation."""
        if not paper_content:
            return
        
        try:
            source_name = "compiler_current_paper"
            
            # Remove old version first
            await rag_manager.remove_document(source_name)
            
            # Add current version
            await rag_manager.add_text(
                paper_content,
                source_name,
                chunk_sizes=[rag_config.validator_chunk_size],  # 512 for consistency
                is_permanent=False
            )
            logger.debug("Indexed current paper in RAG for compiler")
            
        except Exception as e:
            logger.error(f"Failed to index current paper for compiler: {e}")
    
    async def remove_brainstorm_from_rag(self, topic_id: str) -> None:
        """Remove a brainstorm from RAG index (cleanup)."""
        if topic_id in self._brainstorms_indexed:
            try:
                source_name = f"brainstorm_{topic_id}"
                await rag_manager.remove_document(source_name)
                self._brainstorms_indexed.discard(topic_id)
                logger.info(f"Removed brainstorm {topic_id} from RAG")
            except Exception as e:
                logger.error(f"Failed to remove brainstorm {topic_id} from RAG: {e}")


# Global instance
autonomous_rag_manager = AutonomousRAGManager()

