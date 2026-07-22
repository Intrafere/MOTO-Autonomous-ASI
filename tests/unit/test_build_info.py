import json
from pathlib import Path
import tempfile
from unittest import TestCase, main, mock

from backend.shared import build_info
import moto_updater


class BuildInfoArchiveIdentityTests(TestCase):
    def tearDown(self) -> None:
        build_info.get_build_info.cache_clear()

    def test_get_build_info_uses_archival_head_for_zip_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            manifest_path = repo_root / "moto-update-manifest.json"
            package_path = repo_root / "package.json"
            archival_path = repo_root / ".git_archival.txt"
            manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "version": "1.0.9",
                        "build_commit": "stale-local-commit",
                        "update_channel": "main",
                        "api_contract_version": "build5-v29",
                    }
                ),
                encoding="utf-8",
            )
            package_path.write_text(json.dumps({"version": "1.0.9"}), encoding="utf-8")
            archival_path.write_text(
                "node: 0123456789abcdef0123456789abcdef01234567\n",
                encoding="utf-8",
            )

            with mock.patch.object(build_info, "REPO_ROOT", repo_root):
                with mock.patch.object(build_info, "BUILD_MANIFEST_PATH", manifest_path):
                    with mock.patch.object(build_info, "PACKAGE_JSON_PATH", package_path):
                        with mock.patch.object(build_info, "ARCHIVAL_METADATA_PATH", archival_path):
                            build_info.get_build_info.cache_clear()
                            resolved = build_info.get_build_info()

        self.assertEqual(resolved.build_commit, "0123456789abcdef0123456789abcdef01234567")

    def test_default_and_manifest_contract_versions_match(self) -> None:
        manifest = json.loads(build_info.BUILD_MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            build_info._DEFAULT_BUILD_INFO["api_contract_version"],
            "build5-v73",
        )
        self.assertEqual(
            manifest["api_contract_version"],
            build_info._DEFAULT_BUILD_INFO["api_contract_version"],
        )
        self.assertEqual(
            moto_updater._DEFAULT_MANIFEST["api_contract_version"],
            build_info._DEFAULT_BUILD_INFO["api_contract_version"],
        )


if __name__ == "__main__":
    main()
