"""Authoritative cross-process ownership for one mutable MOTO data root."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
from typing import Iterator


class RuntimeRootInUseError(RuntimeError):
    """Raised when another backend already owns the requested data root."""


_PROCESS_GUARD = threading.Lock()
_PROCESS_OWNED: set[str] = set()


class RuntimeRootLease:
    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root).resolve()
        self.lock_path = self.data_root / ".moto_backend.lock"
        self._file = None
        self._identity = os.path.normcase(os.path.realpath(str(self.data_root)))

    def acquire(self) -> "RuntimeRootLease":
        self.data_root.mkdir(parents=True, exist_ok=True)
        with _PROCESS_GUARD:
            if self._identity in _PROCESS_OWNED:
                raise RuntimeRootInUseError(
                    f"MOTO data root is already owned by this backend process: {self.data_root}"
                )
            _PROCESS_OWNED.add(self._identity)

        file_obj = None
        try:
            file_obj = self.lock_path.open("a+b")
            file_obj.seek(0)
            if file_obj.read(1) == b"":
                file_obj.seek(0)
                file_obj.write(b"\0")
                file_obj.flush()
            file_obj.seek(0)
            self._lock_file(file_obj)
            payload = json.dumps(
                {"pid": os.getpid(), "data_root": str(self.data_root)},
                separators=(",", ":"),
            ).encode("utf-8")
            file_obj.seek(1)
            file_obj.write(payload)
            file_obj.truncate()
            file_obj.flush()
            self._file = file_obj
            return self
        except BaseException as exc:
            if file_obj is not None:
                file_obj.close()
            with _PROCESS_GUARD:
                _PROCESS_OWNED.discard(self._identity)
            if isinstance(exc, RuntimeRootInUseError):
                raise
            raise RuntimeRootInUseError(
                f"MOTO data root is already in use by another backend process: {self.data_root}"
            ) from exc

    @staticmethod
    def _lock_file(file_obj) -> None:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeRootInUseError("Windows runtime-root lock is held") from exc
        else:
            import fcntl

            try:
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeRootInUseError("POSIX runtime-root lock is held") from exc

    def release(self) -> None:
        file_obj = self._file
        self._file = None
        try:
            if file_obj is not None:
                file_obj.seek(0)
                if os.name == "nt":
                    import msvcrt

                    try:
                        msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        # Closing the descriptor also releases all Windows byte
                        # locks. This covers runtimes that reject explicit
                        # unlock after writes changed the current file pointer.
                        pass
                else:
                    import fcntl

                    fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
                file_obj.close()
        finally:
            with _PROCESS_GUARD:
                _PROCESS_OWNED.discard(self._identity)

    def __enter__(self) -> "RuntimeRootLease":
        return self.acquire()

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()


@contextmanager
def hold_runtime_root(data_root: str | Path) -> Iterator[RuntimeRootLease]:
    lease = RuntimeRootLease(data_root).acquire()
    try:
        yield lease
    finally:
        lease.release()
