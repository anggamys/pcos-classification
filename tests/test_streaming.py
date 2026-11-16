"""Test streaming live run_matrix: keluaran real-time, exit code, isi log.

Jalankan dari root repo:

    python -m unittest tests.test_streaming -v
"""

import contextlib
import io
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import run_matrix  # noqa: E402

# Child menulis baris pertama, menunggu berkas gate, baru menulis baris kedua
# dan keluar dengan kode 3. Gate memungkinkan test membuktikan baris pertama
# sudah sampai terminal saat proses masih berjalan (bukti streaming).
CHILD_SCRIPT = """
import os
import sys
import time

print("CHILD-FIRST", flush=True)
sys.stdout.write("PROGRESS-50%\\r")
sys.stdout.flush()
gate = sys.argv[1]
deadline = time.monotonic() + 20
while not os.path.exists(gate) and time.monotonic() < deadline:
    time.sleep(0.01)
print("CHILD-SECOND", flush=True)
sys.stderr.write("CHILD-ERR\\n")
sys.stderr.flush()
raise SystemExit(3)
"""

INTERRUPT_SCRIPT = """
import time

print("BEFORE-INTERRUPT", flush=True)
time.sleep(20)
"""


class _InterruptingStream:
    """Stream palsu yang memicu KeyboardInterrupt saat menerima penanda."""

    def __init__(self) -> None:
        self.text = ""

    def write(self, data: str) -> int:
        self.text += data

        if "BEFORE-INTERRUPT" in data:
            raise KeyboardInterrupt

        return len(data)

    def flush(self) -> None:
        pass


class StreamSubprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.gate = self.dir / "gate"

    def _command(self) -> list:
        return [sys.executable, "-u", "-c", CHILD_SCRIPT, str(self.gate)]

    def _read_log(self, log_path: Path) -> str:
        with log_path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()

    def test_prints_live_before_child_exits(self) -> None:
        log_path = self.dir / "stdout.log"
        buffer = io.StringIO()
        result = {}

        def run() -> None:
            with contextlib.redirect_stdout(buffer):
                result["value"] = run_matrix.stream_subprocess(
                    self._command(), log_path
                )

        thread = threading.Thread(target=run)
        thread.start()

        try:
            deadline = time.monotonic() + 20

            while (
                "PROGRESS-50%" not in buffer.getvalue()
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)

            live_text = buffer.getvalue()
            still_running = thread.is_alive()
            released = self.gate.exists()
        finally:
            self.gate.write_text("go")
            thread.join(timeout=20)

        # Baris pertama dan pembaruan progress bar (tanpa newline) terlihat
        # saat child masih menunggu gate: keluarannya di-stream ke terminal,
        # bukan ditahan sampai proses selesai.
        self.assertIn("CHILD-FIRST", live_text)
        self.assertIn("PROGRESS-50%", live_text)
        self.assertTrue(still_running)
        self.assertFalse(released)
        self.assertFalse(thread.is_alive())

        returncode, output = result["value"]
        self.assertEqual(returncode, 3)
        self.assertIn("CHILD-FIRST", output)
        self.assertIn("CHILD-SECOND", output)
        self.assertIn("CHILD-ERR", output)

        # stdout.log memuat gabungan stdout+stderr, termasuk run yang gagal.
        self.assertEqual(self._read_log(log_path), output)

    def test_nonzero_exit_still_writes_full_log(self) -> None:
        self.gate.write_text("go")
        log_path = self.dir / "stdout.log"

        with contextlib.redirect_stdout(io.StringIO()):
            returncode, output = run_matrix.stream_subprocess(
                self._command(), log_path
            )

        self.assertEqual(returncode, 3)
        self.assertTrue(log_path.exists())
        self.assertEqual(self._read_log(log_path), output)
        self.assertIn("CHILD-ERR", self._read_log(log_path))

    def test_interrupt_writes_partial_log_and_terminates_child(self) -> None:
        cmd = [sys.executable, "-u", "-c", INTERRUPT_SCRIPT]
        log_path = self.dir / "stdout.log"
        stream = _InterruptingStream()

        with mock.patch.object(sys, "stdout", stream):
            with self.assertRaises(KeyboardInterrupt):
                run_matrix.stream_subprocess(cmd, log_path)

        self.assertIn("BEFORE-INTERRUPT", self._read_log(log_path))

    def test_build_command_forces_unbuffered_child(self) -> None:
        experiment = {
            "id": "E1",
            "desc": "Baseline tanpa denoise",
            "denoise": False,
            "enhance": "none",
            "segment": "none",
            "input_mode": "full",
            "attention": "self_attention",
        }

        cmd = run_matrix.build_command(experiment, self.dir / "E1")

        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1], "-u")
        self.assertTrue(cmd[2].endswith("main.py"))


class RunExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.experiment = {
            "id": "E1",
            "desc": "Baseline tanpa denoise",
            "denoise": False,
            "enhance": "none",
            "segment": "none",
            "input_mode": "full",
            "attention": "self_attention",
        }

    def test_failure_still_records_row_and_log(self) -> None:
        fake_cmd = [
            sys.executable,
            "-u",
            "-c",
            "print('Accuracy: 0.5'); raise SystemExit(2)",
        ]
        out_root = self.dir / "results"

        with (
            mock.patch.object(run_matrix, "build_command", return_value=fake_cmd),
            mock.patch.object(run_matrix, "model_stats", return_value=(123, 4.5)),
            contextlib.redirect_stdout(io.StringIO()) as buffer,
        ):
            row = run_matrix.run_experiment(self.experiment, out_root)

        self.assertEqual(row["returncode"], 2)
        self.assertEqual(row["total_params"], 123)
        self.assertEqual(row["infer_ms_cpu"], 4.5)
        self.assertEqual(row["accuracy"], 0.5)
        self.assertIsNone(row["auc"])

        log_path = out_root / "E1" / "stdout.log"
        self.assertIn("Accuracy: 0.5", log_path.read_text())
        self.assertIn("E1: Baseline tanpa denoise", buffer.getvalue())
        self.assertIn("E1 status: FAIL", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
