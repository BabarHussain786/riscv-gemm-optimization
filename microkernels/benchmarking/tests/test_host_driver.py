"""Optional compiled-C tests. HOST_REFERENCE_ONLY data never enters paper results."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import unittest

SPEC = importlib.util.spec_from_file_location("host_runner", Path(__file__).resolve().parents[1] / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
BINARY = os.environ.get("BENCH_TEST_BINARY")


@unittest.skipUnless(BINARY, "set BENCH_TEST_BINARY to a compiled --host-test executable")
class CompiledHostTests(unittest.TestCase):
    def invoke(self, extra=(), corrupt=False):
        env = dict(os.environ)
        env.pop("BENCH_TEST_CORRUPT", None)
        if corrupt:
            env["BENCH_TEST_CORRUPT"] = "1"
        command = [BINARY, "--implementation", "reference", "--m", "17", "--n", "67", "--k", "69",
                   "--warmups", "1", "--repetitions", "3"] + list(extra)
        return subprocess.run(command, capture_output=True, text=True, env=env, timeout=60)

    def test_end_to_end_matches_runner_schema(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        rows, issues = runner.parse_rows(result.stdout)
        expected = dict(M=17, N=67, K=69, threads=1, cpu_ids=[0], implementation="reference",
                        timing_mode="end_to_end", profiled=False, repetitions=3)
        self.assertEqual(issues + runner.gate_rows(rows, expected), [])
        self.assertEqual(runner.gate_validation(result.stderr, (17, 67, 69)), [])
        self.assertTrue(all(row["kernel_sec"] is None and row["timestamp_utc"] for row in rows))

    def test_prepacked_and_profiled_scopes(self):
        for mode in ("prepacked", "end_to_end"):
            with self.subTest(mode=mode):
                result = self.invoke(["--timing", mode, "--profile", "1"])
                self.assertEqual(result.returncode, 0, result.stderr)
                for row in map(json.loads, result.stdout.splitlines()):
                    self.assertGreater(row["kernel_sec"], 0)
                    self.assertIsNone(row["output_sec"])
                    self.assertIsNone(row["boundary_sec"])
                    if mode == "prepacked":
                        self.assertIsNone(row["packing_sec"])
                    else:
                        self.assertGreaterEqual(row["packing_sec"], 0)

    def test_validation_only_never_emits_timing(self):
        result = self.invoke(["--validate-only", "1"])
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(rows, [{"record_type": "validation", "status": "OK", "validation": "PASS"}])

    def test_corruption_aborts_before_timing(self):
        result = self.invoke(corrupt=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("status=FAILED", result.stderr)
        self.assertIn("mismatches=", result.stderr)

    def test_host_cannot_impersonate_rvv_or_ime(self):
        for backend in ("rvv", "ime"):
            result = self.invoke(["--implementation", backend])
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")

    def test_bad_dimensions_are_rejected(self):
        result = self.invoke(["--m", "0"])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_unavailable_counters_are_null(self):
        result = self.invoke(["--counters", "1"])
        self.assertEqual(result.returncode, 0, result.stderr)
        if os.name == "nt":
            for row in map(json.loads, result.stdout.splitlines()):
                self.assertEqual(row["counters_status"], "UNAVAILABLE_OR_MULTIPLEXED")
                self.assertIsNone(row["instructions"])
                self.assertIsNone(row["ipc"])


if __name__ == "__main__":
    unittest.main()
