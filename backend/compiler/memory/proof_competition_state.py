"""Autonomous compiler-only durable, atomic competition cursor."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from pathlib import Path


class CompilerCompetitionState:
    def __init__(self, session_dir: Path, run_id: str, source_id: str):
        self.identity = {"run_id": run_id, "source_id": source_id, "schema": 1}
        key = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.path = session_dir / "compiler_competition" / f"{key}.json"
        self.data = {}

    async def load(self):
        def read():
            if not self.path.exists():
                return {}
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("identity") != self.identity:
                raise ValueError("Compiler competition checkpoint identity mismatch")
            return payload["state"]
        self.data = await asyncio.to_thread(read)
        return self.data

    async def save(self):
        payload = json.dumps({"identity": self.identity, "state": self.data}, ensure_ascii=False)
        def write():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("w", encoding="utf-8") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
        # Retain write ownership on cancellation so a later save cannot be
        # overwritten by a still-running thread from the cancelled call.
        task = asyncio.create_task(asyncio.to_thread(write))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
