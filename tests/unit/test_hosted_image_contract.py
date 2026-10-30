"""The hosted image bakes the tiktoken encoding in, so a new sandbox never downloads it."""
from __future__ import annotations

import inspect
import posixpath
import re
import sys
import types
from pathlib import Path
from unittest import TestCase, main, mock

from backend.shared import utils

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"


def _dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def _dockerfile_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in _dockerfile_text().replace("\\\n", " ").splitlines():
        if line.startswith("ENV "):
            env.update(re.findall(r"(\w+)=(\S+)", line[len("ENV "):]))
    return env


def _build_time_python() -> str:
    match = re.search(r"<<'PY'\n(.*?)\nPY\n", _dockerfile_text(), re.S)
    assert match, "Dockerfile has no build-time Python heredoc"
    return match.group(1)


class HostedImageTokenizerCacheTests(TestCase):
    def test_cache_dir_is_a_fixed_image_path_outside_the_data_root(self) -> None:
        env = _dockerfile_env()
        cache_dir = env["TIKTOKEN_CACHE_DIR"]
        data_root = env["MOTO_DATA_ROOT"]

        self.assertTrue(posixpath.isabs(cache_dir))
        self.assertNotEqual(posixpath.commonpath([cache_dir, data_root]), data_root)
        # The build step must already see the cache dir, or the file lands in /tmp instead.
        text = _dockerfile_text()
        self.assertLess(text.index("TIKTOKEN_CACHE_DIR="), text.index("<<'PY'"))

    def test_build_step_loads_the_encoding_count_tokens_uses(self) -> None:
        fake_fastembed = types.ModuleType("fastembed")
        fake_fastembed.TextEmbedding = mock.Mock()

        with mock.patch.dict(sys.modules, {"fastembed": fake_fastembed}):
            with mock.patch("tiktoken.get_encoding") as get_encoding:
                exec(compile(_build_time_python(), str(DOCKERFILE), "exec"), {})

        get_encoding.assert_called_once_with(utils.TIKTOKEN_ENCODING_NAME)
        default = inspect.signature(utils.count_tokens).parameters["encoding_name"].default
        self.assertEqual(default, utils.TIKTOKEN_ENCODING_NAME)

    def test_entrypoint_keeps_the_cache_dir(self) -> None:
        cache_dir = _dockerfile_env()["TIKTOKEN_CACHE_DIR"]

        self.assertIn(
            f'export TIKTOKEN_CACHE_DIR="${{TIKTOKEN_CACHE_DIR:-{cache_dir}}}"',
            ENTRYPOINT.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    main()
