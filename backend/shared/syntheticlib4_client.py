"""SyntheticLib4 corpus client used by the proof-search build slice.

The production SyntheticLib.com service is still under construction, so this
client implements the MOTO-side contract against offline/mock data while keeping
the same public methods that the live adapter will use later.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from backend.shared.config import system_config
from backend.shared.path_safety import validate_single_path_component

logger = logging.getLogger(__name__)

SYNTHETICLIB4_CONTRACT_VERSION = "moto-syntheticlib4-v1"
SYNTHETICLIB4_SCHEMA_VERSION = "syntheticlib4.mock_client.v1"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "syntheticlib4"
_BUILTIN_RELEASE_ID = "stable-2026-06-11"
_MOCK_SCOPES = [
    "proofs:read",
    "releases:read",
    "deltas:read",
    "usage:write",
    "account:status",
    "user_proofs:read",
]


class SyntheticLib4ClientError(RuntimeError):
    """Raised when mock SyntheticLib4 fixture data is unavailable or invalid."""


class SyntheticLib4Client:
    """
    Minimal contract-first SyntheticLib4 adapter.

    This first build slice intentionally supports offline fixtures only. It gives
    MOTO a stable client surface before production auth/download endpoints exist.
    """

    def __init__(self, fixture_dir: Path | None = None) -> None:
        self.fixture_dir = Path(fixture_dir) if fixture_dir else _DEFAULT_FIXTURE_DIR
        self._fixture_dir_explicit = fixture_dir is not None
        self._memory_api_key: str | None = None

    @property
    def snapshot_root(self) -> Path:
        """Return the active data-root SyntheticLib4 cache directory."""
        return Path(system_config.data_dir) / "syntheticlib4"

    def get_status(self) -> dict[str, Any]:
        """Return non-secret mock account status."""
        source_dir, source_kind = self._current_source_dir()
        status_path = source_dir / "account_status_response.json" if source_dir else None
        if status_path and status_path.exists():
            status = self._load_json(status_path)
            auth_mode = "local_snapshot" if source_kind == "data_root_snapshot" else "offline_fixture"
        else:
            status = self._builtin_account_status()
            auth_mode = "built_in_offline_fixture"

        credential_configured = self.has_configured_credentials()
        return {
            **status,
            "credential_configured": credential_configured,
            "auth_mode": "api_key" if credential_configured else auth_mode,
            "hosted_auth_connected": False,
            "production_contract_pending": True,
        }

    def set_api_key(self, api_key: str) -> dict[str, Any]:
        """Store a SyntheticLib4 API key through the mode-appropriate secret path."""
        normalized = (api_key or "").strip()
        if not normalized:
            raise SyntheticLib4ClientError("SyntheticLib4 API key is required")
        if system_config.generic_mode:
            self._memory_api_key = normalized
        else:
            from backend.shared.secret_store import store_syntheticlib4_api_key

            store_syntheticlib4_api_key(normalized)
        return self.get_status()

    def clear_credentials(self) -> dict[str, Any]:
        """Clear the configured SyntheticLib4 credential without touching snapshots."""
        self._memory_api_key = None
        if not system_config.generic_mode:
            from backend.shared.secret_store import clear_syntheticlib4_api_key

            clear_syntheticlib4_api_key()
        return self.get_status()

    def has_configured_credentials(self) -> bool:
        """Return whether a SyntheticLib4 credential is configured without exposing it."""
        if system_config.generic_mode:
            return bool(self._memory_api_key)
        try:
            from backend.shared.secret_store import load_syntheticlib4_api_key

            return bool(load_syntheticlib4_api_key())
        except Exception as exc:
            logger.debug("SyntheticLib4 credential status unavailable: %s", exc)
            return False

    def list_releases(self, channel: str | None = None) -> dict[str, Any]:
        """Return a mock release list derived from the fixture manifest."""
        manifest = self.get_release_manifest()
        requested_channel = (channel or manifest.get("channel") or "stable").strip()
        release_channel = str(manifest.get("channel") or "stable")
        releases = []
        if requested_channel == release_channel:
            releases.append(
                {
                    "release_id": manifest.get("release_id", ""),
                    "channel": release_channel,
                    "created_at": manifest.get("generated_at", ""),
                    "lean_toolchain": manifest.get("lean_toolchain", ""),
                    "mathlib_revision": manifest.get("mathlib_revision", ""),
                    "syntheticlib4_revision": manifest.get("syntheticlib4_revision", ""),
                    "proof_count": manifest.get("proof_count", 0),
                    "schema_version": manifest.get("schema_version", ""),
                    "compatible_moto_contract_versions": manifest.get(
                        "compatible_moto_contract_versions", []
                    ),
                    "manifest_url": self._manifest_uri(),
                }
            )
        return {
            "contract_version": SYNTHETICLIB4_CONTRACT_VERSION,
            "schema_version": "syntheticlib4.releases.v1",
            "releases": releases,
        }

    def get_release_manifest(self) -> dict[str, Any]:
        """Load the local mock release manifest."""
        source_dir, _source_kind = self._current_source_dir()
        manifest_path = source_dir / "release_manifest.json" if source_dir else None
        if manifest_path and manifest_path.exists():
            return self._load_json(manifest_path)
        return self._builtin_release_manifest()

    def load_proof_metadata(self) -> list[dict[str, Any]]:
        """Load SyntheticLib4 proof metadata JSONL fixture records."""
        source_dir, _source_kind = self._current_source_dir()
        metadata_path = source_dir / "proof_metadata.jsonl" if source_dir else None
        if not metadata_path or not metadata_path.exists():
            return self._builtin_proof_metadata()

        records: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SyntheticLib4ClientError(
                    f"Invalid SyntheticLib4 metadata JSONL at line {line_number}: {exc}"
                ) from exc
            self._validate_proof_record(record)
            records.append(record)
        return records

    def validate_local_snapshot(self) -> dict[str, Any]:
        """
        Validate the available local fixture/snapshot before activating search.

        The first MOTO-side build supports JSONL fixture metadata, not full
        archive download/extraction. Real hashes are checked when present; mock
        hash placeholders are reported but not treated as failures.
        """
        source_dir, source_kind = self._current_source_dir()
        manifest = self.get_release_manifest()
        records = self.load_proof_metadata()
        required_manifest_fields = [
            "contract_version",
            "schema_version",
            "release_id",
            "channel",
            "proof_count",
            "compatible_moto_contract_versions",
        ]
        missing = [field for field in required_manifest_fields if manifest.get(field) in (None, "")]
        if missing:
            raise SyntheticLib4ClientError(
                f"SyntheticLib4 manifest is missing required fields: {', '.join(missing)}"
            )
        compatible = manifest.get("compatible_moto_contract_versions") or []
        if SYNTHETICLIB4_CONTRACT_VERSION not in compatible:
            raise SyntheticLib4ClientError(
                f"SyntheticLib4 release is not compatible with {SYNTHETICLIB4_CONTRACT_VERSION}"
            )
        expected_count = int(manifest.get("proof_count") or 0)
        if expected_count and expected_count != len(records):
            raise SyntheticLib4ClientError(
                f"SyntheticLib4 proof count mismatch: manifest={expected_count}, metadata={len(records)}"
            )

        file_checks: list[dict[str, Any]] = []
        for entry in manifest.get("files") or []:
            if not isinstance(entry, dict):
                continue
            name = validate_single_path_component(str(entry.get("name") or ""), "SyntheticLib4 manifest file")
            path = source_dir / name if source_dir else Path()
            expected_hash = str(entry.get("sha256") or "").strip()
            exists = path.exists()
            actual_hash = ""
            hash_verified = False
            if exists and expected_hash and not expected_hash.startswith("mocksha256"):
                actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual_hash != expected_hash:
                    raise SyntheticLib4ClientError(
                        f"SyntheticLib4 snapshot hash mismatch for {name}"
                    )
                hash_verified = True
            file_checks.append(
                {
                    "name": name,
                    "exists": exists,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                    "hash_verified": hash_verified,
                    "mock_hash_placeholder": expected_hash.startswith("mocksha256"),
                }
            )

        return {
            "valid": True,
            "contract_version": SYNTHETICLIB4_CONTRACT_VERSION,
            "release_id": manifest.get("release_id", ""),
            "channel": manifest.get("channel", "stable"),
            "proof_count": len(records),
            "fixture_source": source_kind,
            "snapshot_dir": str(source_dir) if source_dir else "",
            "file_checks": file_checks,
        }

    def import_snapshot_directory(self, source_dir: Path | str, *, channel: str = "stable") -> dict[str, Any]:
        """
        Validate and activate a local SyntheticLib4 snapshot directory.

        Expected input files are `release_manifest.json` and
        `proof_metadata.jsonl`, with optional account-proof fixtures and
        `proofs/*.json` hydration records. The existing active snapshot is
        preserved unless the staged snapshot validates successfully.
        """
        safe_channel = validate_single_path_component(channel or "stable", "SyntheticLib4 channel")
        source_path = Path(source_dir).resolve()
        if not source_path.exists() or not source_path.is_dir():
            raise SyntheticLib4ClientError(f"SyntheticLib4 snapshot source is not a directory: {source_path}")
        self._validate_snapshot_source_tree(source_path)

        releases_root = self.snapshot_root / "releases"
        target_dir = releases_root / safe_channel
        staging_dir = releases_root / f".{safe_channel}.staging"
        previous_dir = releases_root / f".{safe_channel}.previous"

        self._remove_tree(staging_dir)
        releases_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path, staging_dir)

        staged_client = SyntheticLib4Client(staging_dir)
        staged_validation = staged_client.validate_local_snapshot()

        target_moved = False
        try:
            self._remove_tree(previous_dir)
            if target_dir.exists():
                shutil.move(str(target_dir), str(previous_dir))
                target_moved = True
            shutil.move(str(staging_dir), str(target_dir))
        except Exception:
            self._remove_tree(target_dir)
            if target_moved and previous_dir.exists():
                shutil.move(str(previous_dir), str(target_dir))
            raise

        return {
            "success": True,
            "activated_channel": safe_channel,
            "snapshot_dir": str(target_dir),
            "previous_snapshot_preserved": previous_dir.exists(),
            "validation": staged_validation,
        }

    def retrieve_batch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return up to 7 fixture proofs, honoring cursors and excluded fingerprints."""
        limit = min(max(int(request.get("limit") or 7), 1), 7)
        excluded = {str(value) for value in request.get("excluded_fingerprints", [])}
        cursor = str(request.get("cursor") or "").strip()
        include_full_code = bool(request.get("include_full_code", True))
        offset = 0
        if cursor.startswith("cursor_mock_"):
            try:
                offset = int(cursor.removeprefix("cursor_mock_"))
            except ValueError:
                offset = 0

        records = [
            record
            for record in self.load_proof_metadata()
            if str(record.get("fingerprint", "")) not in excluded
        ]
        selected = records[offset : offset + limit]
        next_offset = offset + len(selected)
        next_cursor = f"cursor_mock_{next_offset}" if next_offset < len(records) else None

        proofs = []
        for record in selected:
            payload = dict(record)
            if not include_full_code:
                payload["lean_code"] = ""
            proofs.append(payload)

        manifest = self.get_release_manifest()
        return {
            "contract_version": SYNTHETICLIB4_CONTRACT_VERSION,
            "schema_version": "syntheticlib4.retrieve_batch.v1",
            "retrieval_batch_id": f"rb_mock_{offset + 1:03d}",
            "release_id": manifest.get("release_id", ""),
            "channel": manifest.get("channel", "stable"),
            "lean_toolchain": manifest.get("lean_toolchain", ""),
            "mathlib_revision": manifest.get("mathlib_revision", ""),
            "syntheticlib4_revision": manifest.get("syntheticlib4_revision", ""),
            "proofs": proofs,
            "next_cursor": next_cursor,
            "exhausted": next_cursor is None,
            "exhaustion_reason": None if next_cursor else "fixture_exhausted",
            "quota_remaining": {
                "api_requests_remaining_day": 1999,
                "text_searches_remaining_month": 1999,
                "semantic_searches_remaining_month": 199,
            },
        }

    def hydrate_proof(self, fingerprint: str) -> dict[str, Any] | None:
        """Return one fixture proof by fingerprint, including any available Lean code."""
        safe_fingerprint = validate_single_path_component(fingerprint, "SyntheticLib4 fingerprint")
        source_dir, _source_kind = self._current_source_dir()
        for record in self.load_proof_metadata():
            if record.get("fingerprint") == safe_fingerprint:
                hydration_url = str(record.get("hydration_url") or "")
                if hydration_url.startswith("fixture://syntheticlib4/proofs/"):
                    hydrated_path = source_dir / "proofs" / f"{safe_fingerprint}.json" if source_dir else Path()
                    if hydrated_path.exists():
                        hydrated = self._load_json(hydrated_path)
                        self._validate_proof_record(hydrated)
                        return {**record, **hydrated}
                return record
        return None

    def list_account_proofs(
        self,
        *,
        cursor: str | None = None,
        limit: int = 50,
        release_id: str | None = None,
        channel: str | None = None,
    ) -> dict[str, Any]:
        """Return mock accepted user proofs using the planned account-proof shape."""
        source_dir, _source_kind = self._current_source_dir()
        fixture_path = source_dir / "account_proofs_response.json" if source_dir else None
        if fixture_path and fixture_path.exists():
            return self._load_json(fixture_path)
        return self._account_proofs_from_metadata(
            query="",
            cursor=cursor,
            limit=limit,
            release_id=release_id,
            channel=channel,
            schema_version="syntheticlib4.account_proofs.v1",
        )

    def search_user_proofs(
        self,
        *,
        query: str = "",
        module: str | None = None,
        novelty_rank: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Search mock accepted user proofs using the planned account-proof shape."""
        source_dir, _source_kind = self._current_source_dir()
        fixture_path = source_dir / "account_proofs_search_response.json" if source_dir else None
        if fixture_path and fixture_path.exists():
            return self._load_json(fixture_path)
        search_query = " ".join(part for part in [query, module, novelty_rank] if part)
        return self._account_proofs_from_metadata(
            query=search_query,
            cursor=cursor,
            limit=limit,
            release_id=None,
            channel=None,
            schema_version="syntheticlib4.account_proofs.v1",
        )

    def _load_json(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SyntheticLib4ClientError(f"SyntheticLib4 fixture missing: {path}") from exc
        except json.JSONDecodeError as exc:
            raise SyntheticLib4ClientError(f"Invalid SyntheticLib4 fixture JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise SyntheticLib4ClientError(f"SyntheticLib4 fixture is not an object: {path}")
        return payload

    def _current_source_dir(self) -> tuple[Path | None, str]:
        if not self._fixture_dir_explicit:
            data_snapshot = self._active_data_snapshot_dir()
            if data_snapshot is not None:
                return data_snapshot, "data_root_snapshot"
        if (self.fixture_dir / "release_manifest.json").exists() or (self.fixture_dir / "proof_metadata.jsonl").exists():
            return self.fixture_dir, "filesystem"
        return None, "built_in"

    def _active_data_snapshot_dir(self, channel: str = "stable") -> Path | None:
        candidate = self.snapshot_root / "releases" / validate_single_path_component(channel, "SyntheticLib4 channel")
        if (candidate / "release_manifest.json").exists() and (candidate / "proof_metadata.jsonl").exists():
            return candidate
        return None

    def _manifest_uri(self) -> str:
        source_dir, source_kind = self._current_source_dir()
        if source_kind == "data_root_snapshot" and source_dir is not None:
            return f"file://{source_dir / 'release_manifest.json'}"
        if source_kind == "filesystem":
            return "fixture://syntheticlib4/release_manifest.json"
        return "builtin://syntheticlib4/release_manifest.json"

    def _validate_snapshot_source_tree(self, source_path: Path) -> None:
        required = {"release_manifest.json", "proof_metadata.jsonl"}
        found = {path.name for path in source_path.iterdir() if path.is_file()}
        missing = sorted(required - found)
        if missing:
            raise SyntheticLib4ClientError(
                f"SyntheticLib4 snapshot directory is missing required files: {', '.join(missing)}"
            )

        allowed_root_files = {
            "release_manifest.json",
            "proof_metadata.jsonl",
            "account_status_response.json",
            "account_proofs_response.json",
            "account_proofs_search_response.json",
        }
        max_file_bytes = 64 * 1024 * 1024
        for path in source_path.rglob("*"):
            relative = path.relative_to(source_path)
            if path.is_symlink():
                raise SyntheticLib4ClientError(f"SyntheticLib4 snapshot contains a symlink: {relative}")
            if path.is_dir():
                if relative.parts and relative.parts[0] != "proofs":
                    raise SyntheticLib4ClientError(f"SyntheticLib4 snapshot contains an unsupported directory: {relative}")
                continue
            if path.stat().st_size > max_file_bytes:
                raise SyntheticLib4ClientError(f"SyntheticLib4 snapshot file is too large: {relative}")
            if len(relative.parts) == 1:
                if relative.name not in allowed_root_files:
                    raise SyntheticLib4ClientError(f"SyntheticLib4 snapshot contains an unsupported file: {relative}")
                continue
            if len(relative.parts) == 2 and relative.parts[0] == "proofs" and relative.name.endswith(".json"):
                validate_single_path_component(relative.name, "SyntheticLib4 hydration proof file")
                continue
            raise SyntheticLib4ClientError(f"SyntheticLib4 snapshot contains an unsupported path: {relative}")

        # Validate staged content against the normal contract before copying.
        staging_client = SyntheticLib4Client(source_path)
        staging_client.validate_local_snapshot()

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)

    def _validate_proof_record(self, record: dict[str, Any]) -> None:
        required = [
            "fingerprint",
            "theorem_statement",
            "theorem_statement_hash",
            "lean_code_hash",
            "release_membership",
            "license_terms_id",
        ]
        missing = [field for field in required if not str(record.get(field, "")).strip()]
        if missing:
            raise SyntheticLib4ClientError(
                f"SyntheticLib4 proof record {record.get('fingerprint') or '<unknown>'} "
                f"is missing required fields: {', '.join(missing)}"
            )

    def _account_proofs_from_metadata(
        self,
        *,
        query: str,
        cursor: str | None,
        limit: int,
        release_id: str | None,
        channel: str | None,
        schema_version: str,
    ) -> dict[str, Any]:
        records = self.load_proof_metadata()
        manifest = self.get_release_manifest()
        if release_id and release_id != manifest.get("release_id"):
            records = []
        if channel and channel != manifest.get("channel"):
            records = []
        terms = [term.lower() for term in (query or "").split() if term.strip()]
        if terms:
            def _matches(record: dict[str, Any]) -> bool:
                haystack = " ".join(
                    str(record.get(field) or "")
                    for field in (
                        "fingerprint",
                        "display_title",
                        "theorem_name",
                        "theorem_statement",
                        "informal_statement",
                        "proof_description",
                        "module",
                        "source_path",
                        "novelty_rank",
                    )
                ).lower()
                return all(term in haystack for term in terms)

            records = [record for record in records if _matches(record)]

        offset = 0
        raw_cursor = (cursor or "").strip()
        if raw_cursor.startswith("account_cursor_"):
            try:
                offset = int(raw_cursor.removeprefix("account_cursor_"))
            except ValueError:
                offset = 0
        capped_limit = min(max(int(limit or 50), 1), 100)
        selected = records[offset : offset + capped_limit]
        next_offset = offset + len(selected)
        next_cursor = f"account_cursor_{next_offset}" if next_offset < len(records) else None
        return {
            "contract_version": SYNTHETICLIB4_CONTRACT_VERSION,
            "schema_version": schema_version,
            "proofs": selected,
            "next_cursor": next_cursor,
            "quota_remaining": {
                "api_requests_remaining_day": 1999,
                "text_searches_remaining_month": 1999,
                "semantic_searches_remaining_month": 199,
            },
        }

    def _builtin_account_status(self) -> dict[str, Any]:
        return {
            "contract_version": SYNTHETICLIB4_CONTRACT_VERSION,
            "schema_version": "syntheticlib4.account_status.v1",
            "authenticated": True,
            "membership_active": True,
            "membership_tier": "offline_mock",
            "access_expires_at": "",
            "scopes": list(_MOCK_SCOPES),
            "quota": {
                "api_requests_remaining_day": 2000,
                "text_searches_remaining_month": 2000,
                "semantic_searches_remaining_month": 200,
            },
        }

    def _builtin_release_manifest(self) -> dict[str, Any]:
        return {
            "contract_version": SYNTHETICLIB4_CONTRACT_VERSION,
            "schema_version": "syntheticlib4.release_manifest.v1",
            "release_id": _BUILTIN_RELEASE_ID,
            "channel": "stable",
            "generated_at": "2026-06-11T00:00:00Z",
            "lean_toolchain": "leanprover/lean4:v4.18.0",
            "mathlib_revision": "mock-mathlib-rev",
            "syntheticlib4_revision": "built-in-mock",
            "license_terms_id": "syntheticlib4-member-license-v1",
            "proof_count": 30,
            "novelty_distribution": {
                "novel_formalization": 20,
                "novel_reformulation": 7,
                "minor_mathematical_discovery": 3,
            },
            "compatible_moto_contract_versions": [SYNTHETICLIB4_CONTRACT_VERSION],
            "files": [],
        }

    def _builtin_proof_metadata(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index in range(1, 31):
            theorem_name = f"SyntheticLib4.Mock.builtin_helper_{index:03d}"
            theorem_statement = f"theorem builtin_helper_{index:03d} : True"
            lean_code = (
                "import Mathlib\n\n"
                f"theorem builtin_helper_{index:03d} : True := by\n"
                "  trivial\n"
            )
            fingerprint = f"sl4_builtin_fp_{index:03d}"
            statement_hash = hashlib.sha256(theorem_statement.encode("utf-8")).hexdigest()
            code_hash = hashlib.sha256(lean_code.encode("utf-8")).hexdigest()
            metadata_only = index > 20
            records.append(
                {
                    "fingerprint": fingerprint,
                    "display_title": f"Built-in SyntheticLib4 fixture proof {index}",
                    "theorem_name": theorem_name,
                    "theorem_statement": theorem_statement,
                    "informal_statement": "A built-in offline fixture proof for MOTO proof-search smoke tests.",
                    "proof_description": "Uses `trivial` to close a True goal.",
                    "theorem_statement_hash": statement_hash,
                    "lean_code": "" if metadata_only else lean_code,
                    "lean_code_hash": code_hash,
                    "imports": ["Mathlib"],
                    "dependency_names": ["True.intro"],
                    "topic_tags": ["fixture"],
                    "domain_tags": ["logic"],
                    "module": "SyntheticLib4.Mock",
                    "source_path": "SyntheticLib4/Mock.lean",
                    "line_range": {"start": index, "end": index + 2},
                    "novelty_rank": "novel_formalization",
                    "novelty_confidence": 0.5,
                    "validation_record_id": f"builtin_val_{index:03d}",
                    "release_membership": "stable",
                    "license_terms_id": "syntheticlib4-member-license-v1",
                    "hydration_url": None,
                }
            )
        return records


syntheticlib4_client = SyntheticLib4Client()

