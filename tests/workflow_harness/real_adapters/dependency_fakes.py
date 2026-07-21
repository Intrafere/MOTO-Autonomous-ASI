from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from backend.autonomous.core.proof_verification_stage import ProofVerificationProviderPause
from backend.shared.models import ProofCandidate, ProofStageResult


@dataclass
class EventCollector:
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def broadcast(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        self.events.append((event_type, dict(payload or {})))

    def payloads(self, event_type: str) -> list[dict[str, Any]]:
        return [payload for current_type, payload in self.events if current_type == event_type]


@dataclass
class RoleConfigCapture:
    roles: dict[str, Any] = field(default_factory=dict)

    def configure(self, role_id: str, config: Any) -> None:
        self.roles[role_id] = config


class FakeProofStage:
    def __init__(
        self,
        *,
        result: ProofStageResult | None = None,
        pause: bool = False,
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        self.result = result
        self.pause = pause
        self.checkpoint = checkpoint
        self.calls: list[dict[str, Any]] = []
        self.pause_count = 0

    async def run(self, **kwargs: Any) -> ProofStageResult:
        self.calls.append(kwargs)
        if self.checkpoint and kwargs.get("checkpoint_callback"):
            await kwargs["checkpoint_callback"](dict(self.checkpoint))
        if self.pause and self.pause_count == 0:
            self.pause_count += 1
            raise ProofVerificationProviderPause(
                "OpenRouter credit exhausted",
                remaining_candidates=[
                    ProofCandidate(
                        theorem_id="remaining_candidate",
                        statement="A remaining prompt-relevant theorem.",
                        expected_novelty_tier="novel_variant",
                    )
                ],
            )
        if kwargs.get("append_proof_callback"):
            await kwargs["append_proof_callback"](
                {
                    "proof_id": "manual-proof-1",
                    "source_type": kwargs.get("source_type"),
                    "source_id": kwargs.get("source_id"),
                }
            )
        return self.result or ProofStageResult(
            source_type=kwargs["source_type"],
            source_id=kwargs["source_id"],
            total_candidates=1,
            verified_count=1,
            novel_count=1,
        )


class FakeResearchMetadata:
    def __init__(self) -> None:
        self.proof_checkpoint: dict[str, Any] | None = None
        self.saved_proof_checkpoints: list[dict[str, Any]] = []
        self.workflow_states: list[dict[str, Any]] = []
        self.completed_triggers: list[str] = []

    async def save_proof_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.proof_checkpoint = dict(checkpoint)
        self.saved_proof_checkpoints.append(dict(checkpoint))

    async def get_proof_checkpoint(
        self,
        source_type: str | None = None,
        source_id: str | None = None,
        trigger: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.proof_checkpoint:
            return None
        if source_type and self.proof_checkpoint.get("source_type") != source_type:
            return None
        if source_id and self.proof_checkpoint.get("source_id") != source_id:
            return None
        if trigger and self.proof_checkpoint.get("trigger") != trigger:
            return None
        return dict(self.proof_checkpoint)

    async def mark_proof_checkpoint_trigger_complete(
        self,
        source_type: str,
        source_id: str,
        trigger: str,
        source_title: str = "",
    ) -> None:
        self.completed_triggers.append(trigger)
        self.proof_checkpoint = {
            **(self.proof_checkpoint or {}),
            "source_type": source_type,
            "source_id": source_id,
            "source_title": source_title,
            "trigger": trigger,
            "status": "trigger_complete",
        }

    async def save_workflow_state(self, state: dict[str, Any]) -> None:
        self.workflow_states.append(dict(state))

    async def set_current_brainstorm(self, _topic_id: str | None) -> None:
        return None


def route_workflow_state(*, is_running: bool = False, current_tier: str = "idle") -> SimpleNamespace:
    return SimpleNamespace(is_running=is_running, current_tier=current_tier, is_active=is_running)
