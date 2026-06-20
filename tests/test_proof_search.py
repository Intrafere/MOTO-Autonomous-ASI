import asyncio
import hashlib
import json
import tempfile
import time
from pathlib import Path
from unittest import TestCase, mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.main import app as full_app
from backend.api.routes import proof_search as proof_search_route
from backend.api.routes import syntheticlib4 as syntheticlib4_route
from backend.shared.proof_search.indexer import ProofSearchIndexer
from backend.shared.proof_search import moto_sources
from backend.shared.proof_search.models import ProofSearchRequest, UnifiedProofSearchRecord
from backend.shared.proof_search.search_service import ProofSearchService
from backend.shared.proof_search.syntheticlib4_sources import load_syntheticlib4_fixture_records
from backend.shared.proof_search.tool_adapter import (
    SEARCH_LEAN_PROOFS_TOOL_SCHEMA,
    execute_search_lean_proofs,
)
from backend.shared.config import system_config
from backend.shared.models import ProofCandidate
from backend.shared.syntheticlib4_client import SyntheticLib4Client
from backend.autonomous.agents import proof_formalization_agent as formalization_module


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "syntheticlib4"


def _write_snapshot_fixture(root: Path, *, fingerprint: str, theorem_name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    theorem_statement = f"theorem {theorem_name.split('.')[-1]} : True"
    lean_code = "import Mathlib\n\ntheorem imported_snapshot_helper : True := by\n  trivial\n"
    record = {
        "fingerprint": fingerprint,
        "display_title": theorem_name,
        "theorem_name": theorem_name,
        "theorem_statement": theorem_statement,
        "informal_statement": "Imported local snapshot proof.",
        "proof_description": "A local snapshot proof used for import testing.",
        "theorem_statement_hash": hashlib.sha256(theorem_statement.encode("utf-8")).hexdigest(),
        "lean_code": lean_code,
        "lean_code_hash": hashlib.sha256(lean_code.encode("utf-8")).hexdigest(),
        "imports": ["Mathlib"],
        "dependency_names": ["True.intro"],
        "topic_tags": ["imported"],
        "domain_tags": ["logic"],
        "module": "SyntheticLib4.Imported",
        "source_path": "SyntheticLib4/Imported.lean",
        "line_range": {"start": 1, "end": 3},
        "novelty_rank": "novel_formalization",
        "novelty_confidence": 0.5,
        "validation_record_id": f"val_{fingerprint}",
        "release_membership": "stable",
        "license_terms_id": "syntheticlib4-member-license-v1",
        "hydration_url": None,
    }
    metadata = json.dumps(record) + "\n"
    metadata_path = root / "proof_metadata.jsonl"
    metadata_path.write_text(metadata, encoding="utf-8")
    metadata_bytes = metadata_path.read_bytes()
    manifest = {
        "contract_version": "moto-syntheticlib4-v1",
        "schema_version": "syntheticlib4.release_manifest.v1",
        "release_id": "imported-test-release",
        "channel": "stable",
        "generated_at": "2026-06-12T00:00:00Z",
        "lean_toolchain": "leanprover/lean4:v4.18.0",
        "mathlib_revision": "mock-mathlib-rev",
        "syntheticlib4_revision": "imported-fixture",
        "license_terms_id": "syntheticlib4-member-license-v1",
        "proof_count": 1,
        "compatible_moto_contract_versions": ["moto-syntheticlib4-v1"],
        "files": [
            {
                "name": "proof_metadata.jsonl",
                "sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "size_bytes": len(metadata_bytes),
            }
        ],
    }
    (root / "release_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class SyntheticLib4FixtureTests(TestCase):
    def test_fixture_client_loads_contract_records_and_caps_retrieve_batch(self) -> None:
        client = SyntheticLib4Client(FIXTURE_DIR)

        records = client.load_proof_metadata()
        response = client.retrieve_batch({"limit": 99, "include_full_code": True})

        self.assertEqual(len(records), 30)
        self.assertGreaterEqual(sum(1 for record in records if record.get("lean_code")), 10)
        self.assertGreaterEqual(sum(1 for record in records if not record.get("lean_code")), 5)
        self.assertEqual(len(response["proofs"]), 7)
        self.assertEqual(response["contract_version"], "moto-syntheticlib4-v1")
        self.assertEqual(client.get_status()["membership_active"], True)

    def test_syntheticlib4_records_normalize_for_search(self) -> None:
        records = load_syntheticlib4_fixture_records(SyntheticLib4Client(FIXTURE_DIR))

        self.assertEqual(len(records), 30)
        first = records[0]
        self.assertEqual(first.corpus, "syntheticlib4")
        self.assertEqual(first.external_fingerprint, "sl4_mock_fp_001")
        self.assertEqual(first.release_id, "stable-2026-06-11")
        self.assertIn("Finset.sum_congr", first.dependency_names)

    def test_fixture_client_hydrates_metadata_only_record(self) -> None:
        client = SyntheticLib4Client(FIXTURE_DIR)

        hydrated = client.hydrate_proof("sl4_mock_fp_013")

        self.assertIsNotNone(hydrated)
        self.assertIn("domain_restrict_agree_mock", hydrated["lean_code"])
        self.assertEqual(hydrated["lean_code_hash"], "code_hash_013")

    def test_client_has_built_in_offline_records_when_test_fixtures_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = SyntheticLib4Client(Path(temp_dir) / "missing-fixtures")

            manifest = client.get_release_manifest()
            records = client.load_proof_metadata()
            validation = client.validate_local_snapshot()

        self.assertEqual(manifest["contract_version"], "moto-syntheticlib4-v1")
        self.assertEqual(len(records), 30)
        self.assertEqual(validation["fixture_source"], "built_in")
        self.assertTrue(validation["valid"])

    def test_import_snapshot_directory_activates_data_root_snapshot(self) -> None:
        old_data_dir = system_config.data_dir
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                system_config.data_dir = str(Path(temp_dir) / "data")
                source = Path(temp_dir) / "candidate"
                _write_snapshot_fixture(
                    source,
                    fingerprint="sl4_imported_fp_001",
                    theorem_name="SyntheticLib4.Imported.helper",
                )

                result = SyntheticLib4Client().import_snapshot_directory(source)
                records = SyntheticLib4Client().load_proof_metadata()
                validation = SyntheticLib4Client().validate_local_snapshot()

            self.assertTrue(result["success"])
            self.assertEqual(result["validation"]["proof_count"], 1)
            self.assertEqual(records[0]["fingerprint"], "sl4_imported_fp_001")
            self.assertEqual(validation["fixture_source"], "data_root_snapshot")
        finally:
            system_config.data_dir = old_data_dir

    def test_failed_snapshot_import_preserves_previous_active_snapshot(self) -> None:
        old_data_dir = system_config.data_dir
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                system_config.data_dir = str(Path(temp_dir) / "data")
                first_source = Path(temp_dir) / "first"
                _write_snapshot_fixture(
                    first_source,
                    fingerprint="sl4_imported_fp_keep",
                    theorem_name="SyntheticLib4.Imported.keep",
                )
                SyntheticLib4Client().import_snapshot_directory(first_source)

                bad_source = Path(temp_dir) / "bad"
                bad_source.mkdir()
                (bad_source / "release_manifest.json").write_text("{}", encoding="utf-8")

                with self.assertRaises(Exception):
                    SyntheticLib4Client().import_snapshot_directory(bad_source)

                records = SyntheticLib4Client().load_proof_metadata()

            self.assertEqual(records[0]["fingerprint"], "sl4_imported_fp_keep")
        finally:
            system_config.data_dir = old_data_dir

    def test_snapshot_import_rejects_unsupported_paths_before_activation(self) -> None:
        old_data_dir = system_config.data_dir
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                system_config.data_dir = str(Path(temp_dir) / "data")
                source = Path(temp_dir) / "unsafe"
                _write_snapshot_fixture(
                    source,
                    fingerprint="sl4_imported_fp_unsafe",
                    theorem_name="SyntheticLib4.Imported.unsafe",
                )
                (source / "unexpected.exe").write_text("not allowed", encoding="utf-8")

                with self.assertRaises(Exception):
                    SyntheticLib4Client().import_snapshot_directory(source)

                records = SyntheticLib4Client().load_proof_metadata()

            self.assertEqual(len(records), 30)
            self.assertNotEqual(records[0]["fingerprint"], "sl4_imported_fp_unsafe")
        finally:
            system_config.data_dir = old_data_dir


class ProofSearchIndexerTests(TestCase):
    def test_search_enforces_seven_result_cap_and_dedupes(self) -> None:
        records = load_syntheticlib4_fixture_records(SyntheticLib4Client(FIXTURE_DIR))
        duplicate = records[0].model_copy(
            update={
                "search_id": "syntheticlib4:duplicate-fp-001",
                "proof_id": "duplicate-fp-001",
                "display_title": "Duplicate finite sum cancellation",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            indexer = ProofSearchIndexer(Path(temp_dir) / "proof_search.sqlite")
            indexer.rebuild([duplicate, *records])
            response = indexer.search(
                ProofSearchRequest(
                    query="finite sums",
                    corpora=["syntheticlib4"],
                    limit=99,
                    hydrate_lean_code=False,
                )
            )
            capped_response = indexer.search(
                ProofSearchRequest(
                    query="Mathlib",
                    corpora=["syntheticlib4"],
                    limit=99,
                    hydrate_lean_code=False,
                )
            )

        self.assertLessEqual(capped_response.result_count, 7)
        self.assertEqual(
            sum(result.external_fingerprint == "sl4_mock_fp_001" for result in response.results),
            1,
        )
        self.assertTrue(all(result.lean_code == "" for result in capped_response.results))

    def test_overview_reports_syntheticlib4_corpus(self) -> None:
        records = load_syntheticlib4_fixture_records(SyntheticLib4Client(FIXTURE_DIR))

        with tempfile.TemporaryDirectory() as temp_dir:
            indexer = ProofSearchIndexer(Path(temp_dir) / "proof_search.sqlite")
            indexer.rebuild(records)
            overview = indexer.overview()

        self.assertEqual(overview.total_records, 30)
        self.assertIn("syntheticlib4", {corpus["id"] for corpus in overview.corpora})
        self.assertEqual(overview.result_cap, 7)

    def test_large_index_still_caps_results(self) -> None:
        base_records = load_syntheticlib4_fixture_records(SyntheticLib4Client(FIXTURE_DIR))
        generated = []
        for index in range(1200):
            base = base_records[index % len(base_records)]
            generated.append(
                base.model_copy(
                    update={
                        "search_id": f"syntheticlib4:large-{index}",
                        "proof_id": f"large-{index}",
                        "external_fingerprint": f"large-fp-{index}",
                        "theorem_name": f"SyntheticLib4.Large.generated_{index}",
                        "theorem_statement": f"theorem generated_{index} : True",
                        "theorem_statement_hash": f"large-stmt-{index}",
                        "lean_code_hash": f"large-code-{index}",
                    }
                )
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            indexer = ProofSearchIndexer(Path(temp_dir) / "proof_search.sqlite")
            indexer.rebuild(generated)
            response = indexer.search(
                ProofSearchRequest(
                    query="generated True",
                    corpora=["syntheticlib4"],
                    limit=500,
                    hydrate_lean_code=False,
                )
            )
            overview = indexer.overview()

        self.assertEqual(overview.total_records, 1200)
        self.assertLessEqual(response.result_count, 7)

    def test_generated_50k_index_keeps_search_bounded(self) -> None:
        base_records = load_syntheticlib4_fixture_records(SyntheticLib4Client(FIXTURE_DIR))
        generated = []
        for index in range(50_000):
            base = base_records[index % len(base_records)]
            generated.append(
                base.model_copy(
                    update={
                        "search_id": f"syntheticlib4:scale-{index}",
                        "proof_id": f"scale-{index}",
                        "external_fingerprint": f"scale-fp-{index}",
                        "theorem_name": f"SyntheticLib4.Scale.generated_{index}",
                        "theorem_statement": f"theorem scale_generated_{index} : True",
                        "theorem_statement_hash": f"scale-stmt-{index}",
                        "lean_code": "",
                        "lean_code_hash": f"scale-code-{index}",
                        "module": f"SyntheticLib4.Scale.Module{index % 17}",
                        "source_path": f"SyntheticLib4/Scale/Module{index % 17}.lean",
                        "dependency_names": [f"Scale.dep_{index % 23}", "True.intro"],
                        "topic_tags": [f"scale-topic-{index % 11}"],
                    }
                )
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            indexer = ProofSearchIndexer(Path(temp_dir) / "proof_search.sqlite")
            rebuild_started = time.perf_counter()
            indexer.rebuild(generated)
            rebuild_elapsed = time.perf_counter() - rebuild_started

            search_started = time.perf_counter()
            response = indexer.search(
                ProofSearchRequest(
                    query="scale generated True",
                    corpora=["syntheticlib4"],
                    dependency_names=["Scale.dep_7"],
                    module_filters=["SyntheticLib4.Scale"],
                    limit=500,
                    hydrate_lean_code=False,
                )
            )
            search_elapsed = time.perf_counter() - search_started
            overview = indexer.overview()

        self.assertEqual(overview.total_records, 50_000)
        self.assertLessEqual(response.result_count, 7)
        self.assertTrue(all(not result.lean_code for result in response.results))
        self.assertLess(
            rebuild_elapsed,
            120,
            f"50k proof-search rebuild took too long: {rebuild_elapsed:.2f}s",
        )
        self.assertLess(
            search_elapsed,
            10,
            f"50k proof-search query took too long: {search_elapsed:.2f}s",
        )

    def test_indexer_fetches_one_record_for_detail_lookup(self) -> None:
        records = load_syntheticlib4_fixture_records(SyntheticLib4Client(FIXTURE_DIR))

        with tempfile.TemporaryDirectory() as temp_dir:
            indexer = ProofSearchIndexer(Path(temp_dir) / "proof_search.sqlite")
            indexer.rebuild(records)
            record = indexer.get_record(
                corpus="syntheticlib4",
                proof_id="sl4_mock_fp_013",
            )

        self.assertIsNotNone(record)
        self.assertEqual(record.theorem_name, "SyntheticLib4.Finset.domain_restrict_agree_mock")
        self.assertEqual(record.lean_code, "")

    def test_moto_source_loader_includes_autonomous_history_with_hydrated_code(self) -> None:
        library_entry = {
            "session_id": "session_a",
            "proof_id": "proof_001",
            "theorem_name": "History.example",
            "theorem_statement": "theorem example : True",
            "source_type": "paper",
            "source_id": "paper_001",
            "source_title": "Archived Paper",
            "novel": True,
            "novelty_tier": "novel_formulation",
            "dependencies": [{"kind": "mathlib", "name": "Mathlib.Init"}],
        }
        hydrated_entry = {
            **library_entry,
            "lean_code": "import Mathlib\n\ntheorem example : True := by\n  trivial\n",
        }

        async def _list_library(novel_only: bool = False):
            return [library_entry]

        async def _get_library_proof(session_id: str, proof_id: str):
            self.assertEqual(session_id, "session_a")
            self.assertEqual(proof_id, "proof_001")
            return hydrated_entry

        with mock.patch.object(moto_sources.proof_database, "list_proof_library", _list_library), \
            mock.patch.object(moto_sources.proof_database, "get_library_proof", _get_library_proof):
            records = asyncio.run(moto_sources._records_from_autonomous_history())

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].corpus, "moto")
        self.assertEqual(records[0].corpus_scope, "history")
        self.assertIn("theorem example", records[0].lean_code)
        self.assertIn("Mathlib.Init", records[0].dependency_names)


class ProofSearchRouteTests(TestCase):
    def test_overview_and_search_routes_use_shared_service(self) -> None:
        async def _prepare_service(path: Path) -> ProofSearchService:
            service = ProofSearchService(index_path=path)
            await service.rebuild_index()
            return service

        with tempfile.TemporaryDirectory() as temp_dir:
            service = asyncio.run(_prepare_service(Path(temp_dir) / "proof_search.sqlite"))
            app = FastAPI()
            app.include_router(proof_search_route.router)

            with mock.patch.object(proof_search_route, "proof_search_service", service):
                client = TestClient(app)
                overview = client.get("/api/proof-search/overview")
                search = client.post(
                    "/api/proof-search/search",
                    json={
                        "query": "finite sums",
                        "corpora": ["syntheticlib4"],
                        "limit": 7,
                        "hydrate_lean_code": False,
                    },
                )

        self.assertEqual(overview.status_code, 200)
        self.assertEqual(search.status_code, 200)
        self.assertLessEqual(search.json()["result_count"], 7)
        self.assertEqual(search.json()["searched_corpora"], ["syntheticlib4"])

    def test_proof_detail_route_hydrates_syntheticlib4_record(self) -> None:
        async def _prepare_service(path: Path) -> ProofSearchService:
            service = ProofSearchService(index_path=path)
            await service.rebuild_index()
            return service

        with tempfile.TemporaryDirectory() as temp_dir:
            service = asyncio.run(_prepare_service(Path(temp_dir) / "proof_search.sqlite"))
            app = FastAPI()
            app.include_router(proof_search_route.router)

            with mock.patch.object(proof_search_route, "proof_search_service", service):
                client = TestClient(app)
                response = client.get(
                    "/api/proof-search/proofs/syntheticlib4/sl4_mock_fp_013"
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["external_fingerprint"], "sl4_mock_fp_013")
        self.assertIn("domain_restrict_agree_mock", payload["lean_code"])

    def test_disabled_corpus_toggles_filter_overview_search_and_detail_routes(self) -> None:
        old_syntheticlib4_enabled = system_config.syntheticlib4_enabled
        old_memory_enabled = system_config.agent_conversation_memory_enabled

        async def _prepare_service(path: Path) -> ProofSearchService:
            service = ProofSearchService(index_path=path)
            await service.rebuild_index()
            return service

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                service = asyncio.run(_prepare_service(Path(temp_dir) / "proof_search.sqlite"))
                app = FastAPI()
                app.include_router(proof_search_route.router)
                system_config.syntheticlib4_enabled = False
                system_config.agent_conversation_memory_enabled = False

                with mock.patch.object(proof_search_route, "proof_search_service", service):
                    client = TestClient(app)
                    search = client.post(
                        "/api/proof-search/search",
                        json={
                            "query": "finite sums",
                            "corpora": ["syntheticlib4", "moto", "manual", "leanoj"],
                            "limit": 7,
                            "hydrate_lean_code": False,
                        },
                    )
                    detail = client.get(
                        "/api/proof-search/proofs/syntheticlib4/sl4_mock_fp_013"
                    )
                    overview = client.get("/api/proof-search/overview")
        finally:
            system_config.syntheticlib4_enabled = old_syntheticlib4_enabled
            system_config.agent_conversation_memory_enabled = old_memory_enabled

        self.assertEqual(overview.status_code, 200)
        self.assertEqual(overview.json()["total_records"], 0)
        self.assertEqual(overview.json()["corpora"], [])
        self.assertEqual(search.status_code, 200)
        self.assertEqual(search.json()["result_count"], 0)
        self.assertEqual(search.json()["searched_corpora"], [])
        self.assertEqual(detail.status_code, 404)

    def test_public_route_openapi_exposes_seven_result_caps(self) -> None:
        schema = full_app.openapi()
        proof_limit = (
            schema["components"]["schemas"]["PublicProofSearchRequest"]["properties"]["limit"]
        )
        synthetic_limit = (
            schema["components"]["schemas"]["SyntheticLib4RetrieveBatchRequest"]["properties"]["limit"]
        )

        self.assertEqual(proof_limit["maximum"], 7)
        self.assertEqual(synthetic_limit["maximum"], 7)

    def test_public_proof_search_rejects_limits_above_result_cap(self) -> None:
        app = FastAPI()
        app.include_router(proof_search_route.router)
        client = TestClient(app)

        response = client.post(
            "/api/proof-search/search",
            json={"query": "finite sums", "limit": 99},
        )

        self.assertEqual(response.status_code, 422)


class ProofSearchFreshnessTests(TestCase):
    def test_service_rebuilds_when_source_files_are_newer_than_index(self) -> None:
        old_data_dir = system_config.data_dir
        first_record = UnifiedProofSearchRecord(
            search_id="moto::old",
            corpus="moto",
            source_kind="verified_proof",
            proof_id="old",
            theorem_name="Old.proof",
            theorem_statement="theorem old : True",
            lean_code="theorem old : True := by\n  trivial",
            canonical_uri="moto-proof://old",
        )
        second_record = first_record.model_copy(
            update={
                "search_id": "moto::new",
                "proof_id": "new",
                "theorem_name": "New.proof",
                "theorem_statement": "theorem new : True",
                "lean_code": "theorem new : True := by\n  trivial",
                "canonical_uri": "moto-proof://new",
            }
        )
        records = [[first_record], [second_record]]

        async def _load_records():
            return records.pop(0) if records else [second_record]

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                data_root = Path(temp_dir) / "data"
                system_config.data_dir = str(data_root)
                proof_file = data_root / "proofs" / "proof_old.json"
                proof_file.parent.mkdir(parents=True)
                proof_file.write_text("old", encoding="utf-8")
                service = ProofSearchService(index_path=data_root / "proof_search" / "proof_search.sqlite")
                service._load_records = _load_records  # type: ignore[method-assign]

                first = asyncio.run(
                    service.search(ProofSearchRequest(query="old", corpora=["moto"]))
                )
                time.sleep(0.02)
                proof_file.write_text("new", encoding="utf-8")
                second = asyncio.run(
                    service.search(ProofSearchRequest(query="new", corpora=["moto"]))
                )
        finally:
            system_config.data_dir = old_data_dir

        self.assertEqual(first.results[0].proof_id, "old")
        self.assertEqual(second.results[0].proof_id, "new")


class SyntheticLib4RouteTests(TestCase):
    def test_full_app_registers_syntheticlib4_and_proof_search_routes(self) -> None:
        paths = set(full_app.openapi().get("paths", {}))

        self.assertIn("/api/syntheticlib4/status", paths)
        self.assertIn("/api/syntheticlib4/import-local-snapshot", paths)
        self.assertIn("/api/syntheticlib4/retrieve-batch", paths)
        self.assertIn("/api/syntheticlib4/account/proofs", paths)
        self.assertIn("/api/proof-search/overview", paths)
        self.assertIn("/api/proof-search/search", paths)

    def test_status_releases_and_reindex_routes_expose_fixture_snapshot(self) -> None:
        async def _prepare_service(path: Path) -> ProofSearchService:
            service = ProofSearchService(index_path=path)
            await service.rebuild_index()
            return service

        with tempfile.TemporaryDirectory() as temp_dir:
            service = asyncio.run(_prepare_service(Path(temp_dir) / "proof_search.sqlite"))
            app = FastAPI()
            app.include_router(syntheticlib4_route.router)

            with mock.patch.object(syntheticlib4_route, "proof_search_service", service):
                client = TestClient(app)
                status = client.get("/api/syntheticlib4/status")
                releases = client.get("/api/syntheticlib4/releases")
                reindex = client.post("/api/syntheticlib4/reindex")

        self.assertEqual(status.status_code, 200)
        self.assertTrue(status.json()["local_snapshot"]["available"])
        self.assertEqual(status.json()["proof_index"]["result_cap"], 7)
        self.assertEqual(releases.status_code, 200)
        self.assertGreaterEqual(len(releases.json()["releases"]), 1)
        self.assertEqual(reindex.status_code, 200)
        self.assertTrue(reindex.json()["success"])

    def test_auth_routes_store_and_clear_api_key_without_live_validation(self) -> None:
        app = FastAPI()
        app.include_router(syntheticlib4_route.router)
        client = TestClient(app)
        old_generic_mode = system_config.generic_mode
        system_config.generic_mode = True

        try:
            start = client.post("/api/syntheticlib4/auth/start", json={})
            api_key = client.post("/api/syntheticlib4/api-key", json={"api_key": "sl4_test"})
            clear = client.delete("/api/syntheticlib4/auth")
        finally:
            system_config.generic_mode = old_generic_mode

        self.assertEqual(start.status_code, 501)
        self.assertIn("mock/offline build", start.json()["detail"])
        self.assertEqual(api_key.status_code, 200)
        self.assertTrue(api_key.json()["status"]["credential_configured"])
        self.assertEqual(clear.status_code, 200)
        self.assertTrue(clear.json()["success"])
        self.assertFalse(clear.json()["status"]["credential_configured"])

    def test_retrieve_batch_and_account_proof_routes(self) -> None:
        app = FastAPI()
        app.include_router(syntheticlib4_route.router)
        client = TestClient(app)

        retrieve = client.post(
            "/api/syntheticlib4/retrieve-batch",
            json={"limit": 7, "include_full_code": False},
        )
        account = client.get("/api/syntheticlib4/account/proofs")
        account_search = client.get("/api/syntheticlib4/account/proofs/search?q=finite")

        self.assertEqual(retrieve.status_code, 200)
        self.assertEqual(len(retrieve.json()["proofs"]), 7)
        self.assertTrue(all(not proof.get("lean_code") for proof in retrieve.json()["proofs"]))
        self.assertEqual(account.status_code, 200)
        self.assertIn("proofs", account.json())
        self.assertEqual(account_search.status_code, 200)
        self.assertIn("proofs", account_search.json())

    def test_public_retrieve_batch_rejects_limits_above_result_cap(self) -> None:
        app = FastAPI()
        app.include_router(syntheticlib4_route.router)
        client = TestClient(app)

        response = client.post(
            "/api/syntheticlib4/retrieve-batch",
            json={"limit": 99, "include_full_code": False},
        )

        self.assertEqual(response.status_code, 422)

    def test_import_local_snapshot_route_activates_staged_data_root_snapshot(self) -> None:
        old_data_dir = system_config.data_dir
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                system_config.data_dir = str(Path(temp_dir) / "data")
                source = Path(system_config.data_dir) / "syntheticlib4" / "imports" / "snapshot_a"
                _write_snapshot_fixture(
                    source,
                    fingerprint="sl4_route_import_fp_001",
                    theorem_name="SyntheticLib4.Route.imported",
                )
                app = FastAPI()
                app.include_router(syntheticlib4_route.router)
                client = TestClient(app)

                response = client.post(
                    "/api/syntheticlib4/import-local-snapshot",
                    json={"source_name": "snapshot_a", "channel": "stable"},
                )

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["success"])
            self.assertEqual(response.json()["validation"]["proof_count"], 1)
        finally:
            system_config.data_dir = old_data_dir


class ProofSearchToolAdapterTests(TestCase):
    def test_tool_schema_exposes_search_lean_proofs(self) -> None:
        self.assertEqual(
            SEARCH_LEAN_PROOFS_TOOL_SCHEMA["function"]["name"],
            "search_lean_proofs",
        )
        action_schema = SEARCH_LEAN_PROOFS_TOOL_SCHEMA["function"]["parameters"]["properties"]["action"]
        self.assertIn("search", action_schema["enum"])
        self.assertIn("hydrate", action_schema["enum"])

    def test_tool_overview_search_and_hydrate(self) -> None:
        async def _prepare_service(path: Path) -> ProofSearchService:
            service = ProofSearchService(index_path=path)
            await service.rebuild_index()
            return service

        with tempfile.TemporaryDirectory() as temp_dir:
            service = asyncio.run(_prepare_service(Path(temp_dir) / "proof_search.sqlite"))
            overview = asyncio.run(
                execute_search_lean_proofs({"action": "overview"}, service=service)
            )
            search = asyncio.run(
                execute_search_lean_proofs(
                    {
                        "action": "search",
                        "query": "finite",
                        "corpora": ["autonomous", "syntheticlib4"],
                        "limit": 99,
                        "hydrate_lean_code": False,
                    },
                    service=service,
                )
            )
            hydrate = asyncio.run(
                execute_search_lean_proofs(
                    {
                        "action": "hydrate",
                        "source": "syntheticlib4",
                        "proof_id": "sl4_mock_fp_013",
                    },
                    service=service,
                )
            )

        self.assertTrue(overview["success"])
        self.assertGreaterEqual(overview["overview"]["total_records"], 30)
        self.assertTrue(search["success"])
        self.assertLessEqual(len(search["results"]), 7)
        self.assertIn("syntheticlib4", search["searched_corpora"])
        self.assertTrue(hydrate["success"])
        self.assertIn("domain_restrict_agree_mock", hydrate["results"][0]["lean_code"])

    def test_tool_persists_usage_attestation_locally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = asyncio.run(
                execute_search_lean_proofs(
                    {
                        "action": "attest_usage",
                        "usage_attestation": {
                            "retrieval_batch_id": "rb_mock_001",
                            "used_proofs": [
                                {
                                    "fingerprint": "sl4_mock_fp_013",
                                    "theorem_statement_hash": "stmt_hash_013",
                                    "lean_code_hash": "code_hash_013",
                                }
                            ],
                            "entire_code_used": True,
                            "moto_artifact_hash": "artifact_hash",
                        },
                    },
                    usage_root=Path(temp_dir),
                )
            )
            usage_file = Path(temp_dir) / "usage_attestations.jsonl"
            saved = usage_file.read_text(encoding="utf-8")

        self.assertTrue(result["success"])
        self.assertTrue(result["usage_attestation"]["persisted"])
        self.assertIn("sl4_mock_fp_013", saved)
        self.assertIn("stmt_hash_013", saved)
        self.assertIn("code_hash_013", saved)

    def test_tool_rejects_whole_code_attestation_without_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = asyncio.run(
                execute_search_lean_proofs(
                    {
                        "action": "attest_usage",
                        "usage_attestation": {
                            "retrieval_batch_id": "rb_mock_001",
                            "used_fingerprints": ["sl4_mock_fp_013"],
                            "entire_code_used": True,
                            "moto_artifact_hash": "artifact_hash",
                        },
                    },
                    usage_root=Path(temp_dir),
                )
            )

        self.assertFalse(result["success"])
        self.assertIn("theorem_statement_hash", result["error"])

    def test_formalization_agent_records_syntheticlib_context_exposure(self) -> None:
        agent = formalization_module.ProofFormalizationAgent(
            model_id="test-model",
            context_window=8192,
            max_output_tokens=1024,
            role_id="test_role",
        )
        candidate = ProofCandidate(
            theorem_id="candidate_001",
            statement="theorem target : True",
            formal_sketch="Prove True.",
            expected_novelty_tier="novel_formulation",
            prompt_relevance_rationale="Relevant to the prompt.",
            novelty_rationale="A test target.",
            why_not_standard_known_result="Fixture-only test.",
        )
        captured = {}
        original_execute = formalization_module.execute_search_lean_proofs

        async def fake_execute_search_lean_proofs(arguments):
            captured.update(arguments)
            return {"success": True, "usage_attestation": {"persisted": True}}

        try:
            formalization_module.execute_search_lean_proofs = fake_execute_search_lean_proofs
            asyncio.run(
                agent._record_syntheticlib4_context_exposure(
                    [
                        {
                            "corpus": "syntheticlib4",
                            "fingerprint": "sl4_mock_fp_001",
                            "theorem_statement_hash": "stmt_hash",
                            "lean_code_hash": "code_hash",
                            "lean_code": "import Mathlib\n\ntheorem helper : True := by\n  trivial\n",
                        },
                        {
                            "corpus": "moto",
                            "proof_id": "local_proof",
                            "lean_code": "import Mathlib\n",
                        },
                    ],
                    theorem_candidate=candidate,
                    lean_code="import Mathlib\n\ntheorem target : True := by\n  trivial\n",
                )
            )
        finally:
            formalization_module.execute_search_lean_proofs = original_execute

        self.assertEqual(captured["action"], "attest_usage")
        attestation = captured["usage_attestation"]
        self.assertEqual(attestation["usage_type"], "model_visible_context")
        self.assertFalse(attestation["entire_code_used"])
        self.assertEqual(attestation["used_proofs"][0]["fingerprint"], "sl4_mock_fp_001")
        self.assertEqual(attestation["used_proofs"][0]["theorem_statement_hash"], "stmt_hash")
        self.assertEqual(attestation["used_proofs"][0]["lean_code_hash"], "code_hash")

    def test_formalization_agent_does_not_expose_in_role_proof_search_tool_loop(self) -> None:
        agent = formalization_module.ProofFormalizationAgent(
            model_id="test-model",
            context_window=8192,
            max_output_tokens=1024,
            role_id="test_role",
        )

        self.assertFalse(hasattr(agent, "_generate_completion_with_proof_search_tool"))
        self.assertFalse(hasattr(agent, "_retrieve_proof_search_context"))

    def test_tactic_script_uses_assistant_without_normal_path_in_role_search(self) -> None:
        agent = formalization_module.ProofFormalizationAgent(
            model_id="test-model",
            context_window=8192,
            max_output_tokens=1024,
            role_id="test_role",
        )
        candidate = ProofCandidate(
            theorem_id="candidate_tactic_001",
            statement="theorem target : True",
            formal_sketch="Prove True with a tactic script.",
            expected_novelty_tier="novel_formulation",
            prompt_relevance_rationale="Relevant to the prompt.",
            novelty_rationale="A test target.",
            why_not_standard_known_result="Fixture-only test.",
        )
        original_generate = formalization_module.api_client_manager.generate_completion
        original_execute = formalization_module.execute_search_lean_proofs
        original_get_lean = formalization_module.get_lean4_client
        original_assistant = formalization_module.assistant_proof_search_coordinator
        calls = []

        class FakeAssistantCoordinator:
            def __init__(self) -> None:
                self.snapshots = []

            def submit_target(self, snapshot):
                self.snapshots.append(snapshot)
                return "target_hash"

            def get_latest_pack(self, target_hash=None):
                return None

        class FakeLeanResult:
            success = True
            tactic_error_slice = ""
            error_output = ""
            goal_states = ""

        class FakeLeanClient:
            async def check_tactic_script(self, theorem_header, tactic_commands, timeout):
                self.last_header = theorem_header
                self.last_tactics = tactic_commands
                return FakeLeanResult()

        fake_assistant = FakeAssistantCoordinator()
        fake_lean = FakeLeanClient()

        async def fake_generate_completion(**kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({
                                "theorem_name": "target",
                                "theorem_header": "theorem target : True := by",
                                "tactics": [{"tactic": "trivial", "reasoning": "True is immediate."}],
                                "reasoning": "Use the simplest tactic script.",
                            }),
                        }
                    }
                ]
            }

        async def fail_execute_search_lean_proofs(arguments):
            raise AssertionError("normal tactic-script path must not call search_lean_proofs directly")

        try:
            formalization_module.api_client_manager.generate_completion = fake_generate_completion
            formalization_module.execute_search_lean_proofs = fail_execute_search_lean_proofs
            formalization_module.get_lean4_client = lambda: fake_lean
            formalization_module.assistant_proof_search_coordinator = fake_assistant
            success, theorem_name, lean_code, attempts = asyncio.run(
                agent.prove_candidate_tactic_script(
                    user_research_prompt="Prove the target.",
                    source_type="paper",
                    theorem_candidate=candidate,
                    source_content="A source asks for theorem target : True.",
                    max_attempts=1,
                    source_title="Assistant tactic test",
                )
            )
        finally:
            formalization_module.api_client_manager.generate_completion = original_generate
            formalization_module.execute_search_lean_proofs = original_execute
            formalization_module.get_lean4_client = original_get_lean
            formalization_module.assistant_proof_search_coordinator = original_assistant

        self.assertTrue(success)
        self.assertEqual(theorem_name, "target")
        self.assertIn("theorem target", lean_code)
        self.assertEqual(attempts[-1].strategy, "tactic_script")
        self.assertEqual(fake_lean.last_tactics, ["trivial"])
        self.assertEqual(len(fake_assistant.snapshots), 1)
        self.assertEqual(fake_assistant.snapshots[0].target_statement, candidate.statement)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0].get("tools"))
        self.assertIsNone(calls[0].get("tool_choice"))

    def test_compiler_aggregator_formalization_roles_are_manual_assistant_targets(self) -> None:
        self.assertEqual(
            formalization_module._assistant_workflow_mode_for_role(
                "autonomous_proof_formalization_compiler_aggregator"
            ),
            "manual_proof_check",
        )
        self.assertEqual(
            formalization_module._assistant_workflow_mode_for_role(
                "autonomous_proof_formalization_paper_1"
            ),
            "autonomous",
        )

