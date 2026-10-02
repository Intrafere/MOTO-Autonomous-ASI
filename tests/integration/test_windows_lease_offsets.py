"""Real Windows descriptor regressions; no production runtime roots are used."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import moto_launcher

from backend.shared.runtime_root_lock import RuntimeRootLease, RuntimeRootInUseError
from moto_launcher import backend_lease_is_held, read_backend_lease_owner


@unittest.skipUnless(sys.platform == "win32", "Windows byte-range locks")
class WindowsLeaseOffsetTests(unittest.TestCase):
    def test_legacy_buffered_owner_and_new_lease(self):
        with tempfile.TemporaryDirectory() as root:
            lock = Path(root) / ".moto_backend.lock"
            lock.write_bytes(b"\0" + b" " * 6571)
            script = '''
import json, os, sys, msvcrt
from pathlib import Path
f = (Path(sys.argv[1]) / '.moto_backend.lock').open('a+b')
f.seek(0)
f.read(1)
f.seek(0)
print(os.lseek(f.fileno(), 0, os.SEEK_CUR), flush=True)
msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
f.seek(1)
f.write(json.dumps({'pid': os.getpid(), 'data_root': sys.argv[1]}).encode())
f.truncate()
f.flush()
print('ready', flush=True)
sys.stdin.readline()
f.close()
'''
            child = subprocess.Popen([sys.executable, "-B", "-c", script, root],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "6572")
                self.assertEqual(child.stdout.readline().strip(), "ready")
                self.assertTrue(backend_lease_is_held(root))
                self.assertEqual(read_backend_lease_owner(root), child.pid)
                with self.assertRaises(RuntimeRootInUseError):
                    RuntimeRootLease(root).acquire()
                self.assertEqual(read_backend_lease_owner(root), child.pid)
                # Exercise production recovery with a real legacy lock owner;
                # network/process identity is faked only for this test child.
                with mock.patch.object(moto_launcher, "windows_listening_ports_for_pid", return_value=[8123]), \
                     mock.patch.object(moto_launcher, "backend_health_identity", return_value="default"), \
                     mock.patch.object(moto_launcher, "is_moto_backend_process", return_value=True), \
                     mock.patch.object(moto_launcher, "windows_parent_pid", return_value=999999), \
                     mock.patch.object(moto_launcher, "is_pid_running", side_effect=lambda pid: pid == child.pid), \
                     mock.patch.object(moto_launcher, "backend_byte_zero_is_held", return_value=False), \
                     mock.patch.object(moto_launcher, "cleanup_launcher_state", return_value=[]), \
                     mock.patch.object(moto_launcher, "port_in_use", return_value=False):
                    moto_launcher.assert_runtime_lock_available(root)
                self.assertFalse(backend_lease_is_held(root))
            finally:
                child.communicate("stop\n", timeout=10)
            self.assertFalse(backend_lease_is_held(root))
            with RuntimeRootLease(root):
                self.assertTrue(backend_lease_is_held(root))
                # The modern lease intentionally protects the complete
                # migration range, including its payload; owner recovery uses
                # launcher metadata while a modern backend is active.
                self.assertIsNone(read_backend_lease_owner(root))
            self.assertFalse(backend_lease_is_held(root))
            first_size = lock.stat().st_size
            with RuntimeRootLease(root):
                self.assertTrue(backend_lease_is_held(root))
            self.assertEqual(lock.stat().st_size, first_size)
