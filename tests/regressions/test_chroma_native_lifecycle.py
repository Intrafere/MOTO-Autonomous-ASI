from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows native-binding release blocker")
def test_real_chroma_rebuild_reopen_survives_subprocess(tmp_path: Path) -> None:
    """A native access violation becomes a deterministic nonzero subprocess exit."""
    script = textwrap.dedent(
        """
        import asyncio
        from pathlib import Path

        from backend.aggregator.core.rag_manager import RAGManager
        from backend.shared.config import bind_runtime_roots

        root = Path(r"{root}")
        bind_runtime_roots(data_root=root / "data", logs_root=root / "logs")

        async def main():
            manager = RAGManager()
            await manager.ensure_initialized()
            for collection in manager.collections.values():
                collection.upsert(
                    ids=["first"],
                    embeddings=[[0.0, 1.0, 0.0]],
                    documents=["native lifecycle"],
                    metadatas=[{{"source_file": "source.txt"}}],
                )
                await manager.remove_document("source.txt")
                assert all(collection.count() == 0 for collection in manager.collections.values())
            await manager.close()
            await manager.ensure_initialized()
            assert all(collection.count() == 0 for collection in manager.collections.values())
            await manager.close()

        asyncio.run(main())
        """
    ).format(root=str(tmp_path).replace("\\", "\\\\"))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (
        f"native Chroma subprocess exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows process-generation cache contract")
def test_windows_process_generation_never_opens_prior_cache(tmp_path: Path) -> None:
    """A new backend generation replaces the old cache before native open."""
    root = str(tmp_path).replace("\\", "\\\\")
    create_script = textwrap.dedent(
        f"""
        import asyncio
        from pathlib import Path
        from backend.aggregator.core.rag_manager import RAGManager
        from backend.shared.config import bind_runtime_roots

        root = Path(r"{root}")
        bind_runtime_roots(data_root=root / "data", logs_root=root / "logs")

        async def main():
            manager = RAGManager()
            await manager.ensure_initialized()
            collection = manager.collections[256]
            collection.upsert(
                ids=["prior"],
                embeddings=[[0.0, 1.0, 0.0]],
                documents=["prior generation"],
                metadatas=[{{"source_file": "prior.txt"}}],
            )
            assert collection.count() == 1
            await manager.close()

        asyncio.run(main())
        """
    )
    first = subprocess.run(
        [sys.executable, "-c", create_script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert first.returncode == 0, first.stderr

    reopen_script = textwrap.dedent(
        f"""
        import asyncio
        from pathlib import Path
        from backend.aggregator.core.rag_manager import RAGManager
        from backend.shared.config import bind_runtime_roots

        root = Path(r"{root}")
        bind_runtime_roots(data_root=root / "data", logs_root=root / "logs")

        async def main():
            manager = RAGManager()
            await manager.prepare_process_generation_cache()
            assert manager.collections[256].count() == 0
            manager.collections[256].upsert(
                ids=["current"],
                embeddings=[[1.0, 0.0, 0.0]],
                documents=["current generation"],
                metadatas=[{{"source_file": "current.txt"}}],
            )
            assert manager.collections[256].count() == 1
            await manager.close()

        asyncio.run(main())
        """
    )
    second = subprocess.run(
        [sys.executable, "-c", reopen_script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert second.returncode == 0, (
        f"replacement subprocess exited {second.returncode}\n"
        f"stdout:\n{second.stdout}\nstderr:\n{second.stderr}"
    )
