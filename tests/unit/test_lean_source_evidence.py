import unittest

from backend.shared.lean_source_evidence import (
    parse_explicit_import_modules,
    receipt_digest,
    source_sha256,
    transformations_between,
)
from backend.shared.models import LeanCompileReceipt, PortableProofDependency


class LeanSourceEvidenceTests(unittest.TestCase):
    def test_import_parser_ignores_comments_and_strings(self) -> None:
        source = (
            '/- import Fake.Block -/\n'
            '-- import Fake.Line\n'
            'def text := "import Fake.String"\n'
            "import Mathlib Data.Nat.Basic\n"
            "import Mathlib\n"
        )
        self.assertEqual(
            parse_explicit_import_modules(source),
            ["Mathlib", "Data.Nat.Basic"],
        )

    def test_source_hash_binds_exact_utf8_and_newlines(self) -> None:
        self.assertNotEqual(source_sha256("α"), source_sha256("α\n"))
        self.assertNotEqual(source_sha256("a\r\n"), source_sha256("a\n"))

    def test_transformations_report_inserted_import(self) -> None:
        labels = transformations_between(
            "theorem t : True := by trivial",
            "import Mathlib\n\ntheorem t : True := by trivial\n",
        )
        self.assertIn("insert_default_import", labels)
        self.assertIn("normalize_final_newline", labels)

    def test_receipt_digest_detects_mutation(self) -> None:
        payload = {
            "schema_version": "moto-lean-compile-receipt-v1",
            "result": "accepted",
            "checker_mode": "subprocess",
        }
        digest = receipt_digest(payload)
        payload["checker_mode"] = "lsp"
        self.assertNotEqual(digest, receipt_digest(payload))

    def test_mathlib_portable_dependency_rejects_missing_revision(self) -> None:
        with self.assertRaises(ValueError):
            PortableProofDependency(
                kind="mathlib",
                declaration_name="Nat.add_comm",
            )

    def test_receipt_rejects_unknown_version(self) -> None:
        with self.assertRaises(ValueError):
            LeanCompileReceipt.model_validate(
                {
                    "schema_version": "unknown",
                    "receipt_digest": "0" * 64,
                    "checker_mode": "subprocess",
                    "original_source_sha256": "0" * 64,
                    "original_source_byte_length": 0,
                    "checked_source_sha256": "0" * 64,
                    "checked_source_byte_length": 0,
                    "lean_version": "Lean 4",
                    "mathlib_revision": "a" * 40,
                    "moto_version": "test",
                    "moto_build_commit": "dev",
                    "moto_api_contract_version": "test",
                    "timeout_seconds": 1,
                    "completed_at": "2026-10-07T00:00:00Z",
                }
            )


if __name__ == "__main__":
    unittest.main()
