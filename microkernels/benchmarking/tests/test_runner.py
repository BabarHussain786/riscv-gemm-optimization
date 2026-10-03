"""Meaningful failure gates: one failed invocation excludes its entire case."""
import copy
import contextlib
import csv
import importlib.util
import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location("benchmark_run", Path(__file__).resolve().parents[1] / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@contextlib.contextmanager
def fixture_directory():
    # Default permissions avoid Windows sandbox mode-0700 temp-directory ACLs.
    parent = Path(__file__).resolve().parent
    directory = parent / ("runner-fixture-" + uuid.uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        assert directory.resolve().parent == parent
        shutil.rmtree(directory)


def arguments(root, host=False):
    args = runner.parser().parse_args(["run", "--m", "16", "--n", "16", "--k", "64", "--repetitions", "2"])
    args.project_root = root
    args.cpus = [0]
    args.host_test = host
    return args


def row(expected, rep=1):
    fields = ("M", "N", "K", "implementation", "threads", "timing_mode", "cpu_ids", "profiled")
    return {**{key: expected[key] for key in fields}, "status": "OK", "validation": "PASS", "rep": rep,
            "total_sec": 0.01, "gops": 2 * expected["M"] * expected["N"] * expected["K"] / 0.01 / 1e9,
            "packing_sec": None, "kernel_sec": None, "output_sec": None, "boundary_sec": None,
            "phase_aggregation": "sum_worker_elapsed"}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.expected = {"M": 16, "N": 16, "K": 64, "implementation": "rvv", "threads": 1,
                         "timing_mode": "end_to_end", "cpu_ids": [0], "profiled": False, "repetitions": 2}
        self.rows = [row(self.expected, rep) for rep in (1, 2)]

    def test_valid_rows(self):
        self.assertEqual(runner.gate_rows(self.rows, self.expected), [])

    def test_bad_second_row_rejects_case(self):
        mutations = ({"validation": "FAIL"}, {"total_sec": float("nan")}, {"gops": float("inf")},
                     {"gops": 4.0}, {"rep": 1}, {"cpu_ids": [1]}, {"implementation": "reference"},
                     {"kernel_sec": -1}, {"phase_aggregation": "wall"})
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                rows = copy.deepcopy(self.rows)
                rows[1].update(mutation)
                self.assertTrue(runner.gate_rows(rows, self.expected))

    def test_truncated_and_extra_repetitions_rejected(self):
        self.assertTrue(runner.gate_rows(self.rows[:1], self.expected))
        self.assertTrue(runner.gate_rows(self.rows + [self.rows[0]], self.expected))

    def test_non_json_stdout_is_rejected_without_synthetic_row(self):
        rows, errors = runner.parse_rows("kernel aborted\n" + json.dumps(self.rows[0]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(errors), 1)

    def test_missing_boundary_validation_is_rejected(self):
        stderr = "VALIDATION shape=16x16x64 status=PASS\nVALIDATION shape=16x16x64 status=PASS\n"
        self.assertTrue(runner.gate_validation(stderr, (16, 16, 64)))


class CaseTests(unittest.TestCase):
    def run_fixture(self, host=False, returncode=0, validation_failure=False, validation_only=False):
        with fixture_directory() as directory:
            campaign = Path(directory)
            args = arguments(campaign, host)
            _, case = runner.case_plan(args)[0]
            calls = []

            def fake_launch(command, target, label, timeout):
                calls.append(label)
                if label == "build":
                    (target / "build").mkdir()
                    (target / "build" / ("bench.exe" if runner.os.name == "nt" else "bench")).touch()
                    runner.write_json(target / "build" / "build.json", {"source_digest": "fixture-only", "host_test": host})
                    return {"returncode": 0, "stdout": "", "stderr": ""}
                shape = [int(command[command.index("--" + key) + 1]) for key in ("m", "n", "k")]
                expected = {**case, **dict(zip(("M", "N", "K"), shape))}
                data = ([row(expected, rep) for rep in (1, 2)] if label == "timing" else
                        [{"record_type": "validation", "status": "OK", "validation": "PASS"}])
                if validation_failure and label.startswith("validation"):
                    data[0]["validation"] = "FAIL"
                stderr = "\n".join(f"VALIDATION shape={m}x{n}x{k} status=PASS" for m, n, k in
                                   ((16, 16, 64), (15, 15, 69), tuple(shape)))
                return {"returncode": returncode if label == "timing" else 0,
                        "stdout": "\n".join(map(json.dumps, data)), "stderr": stderr}

            info = {"native_platform": not host, "system_id": "fixture", "system": "Linux", "machine": "riscv64"}
            with mock.patch.object(runner, "launch", side_effect=fake_launch):
                result = runner.run_case(args, campaign, "end_to_end", case, info, validation_only)
            self.assertEqual(len(list((campaign / "validation").glob("*.json"))), 1)
            return result, calls

    def test_success_accepts_exact_rows_after_mandatory_validations(self):
        (raw, accepted, failed, error), calls = self.run_fixture()
        self.assertEqual(len(raw), 2)
        self.assertEqual(len(accepted), 2)
        self.assertFalse(failed or error)
        self.assertEqual(sum(label.startswith("validation") for label in calls), 1)
        self.assertEqual(calls[-1], "timing")

    def test_process_failure_rejects_even_well_formed_partial_output(self):
        (raw, accepted, failed, error), _ = self.run_fixture(returncode=-4)
        self.assertEqual(len(raw), 2)
        self.assertEqual(accepted, [])
        self.assertEqual(len(failed), 2)
        self.assertTrue(error)

    def test_host_reference_never_enters_paper_data(self):
        (raw, accepted, failed, error), _ = self.run_fixture(host=True)
        self.assertEqual(len(raw), 2)
        self.assertEqual(accepted, [])
        self.assertTrue(all(item["implementation"] == "reference" and not item["publishable"] for item in raw))
        self.assertFalse(error)

    def test_failed_validation_prevents_timing(self):
        (raw, accepted, failed, error), calls = self.run_fixture(validation_failure=True)
        self.assertEqual(raw, [])
        self.assertEqual(accepted, [])
        self.assertNotIn("timing", calls)
        self.assertTrue(error)

    def test_preflight_success_has_no_timing_or_accepted_or_failed_rows(self):
        (raw, accepted, failed, error), calls = self.run_fixture(validation_only=True)
        self.assertEqual((raw, accepted, failed, error), ([], [], [], False))
        self.assertNotIn("timing", calls)

    def test_unsupported_platform_never_starts_subprocess(self):
        with fixture_directory() as directory:
            args = arguments(Path(directory))
            _, case = runner.case_plan(args)[0]
            info = {"native_platform": False, "system_id": "fixture", "system": "Windows", "machine": "AMD64"}
            with mock.patch.object(runner, "launch") as launch:
                raw, accepted, failed, error = runner.run_case(args, Path(directory), "end_to_end", case, info)
            launch.assert_not_called()
            self.assertEqual(raw, [])
            self.assertEqual(accepted, [])
            self.assertTrue(failed[0]["diagnostic_only"])
            self.assertTrue(error)

    def test_campaign_uses_same_rvv_and_disjoint_mixed_roles(self):
        args = arguments(Path("."))
        args.command = "campaign"
        central = [case for group, case in runner.case_plan(args) if group.startswith("eight_core")]
        self.assertEqual(len(central), 3)
        self.assertEqual({case["rvv_kernel"] for case in central}, {args.rvv_kernel})
        self.assertEqual([case["ime_workers"] for case in central], [0, 4, 4])
        self.assertTrue(all(case["cpu_ids"] == list(range(8)) for case in central))

    def test_campaign_profiles_both_backends_by_default(self):
        args = arguments(Path("."))
        args.command = "campaign"
        profiles = [case for _, case in runner.case_plan(args) if case["profiled"]]
        self.assertEqual({case["implementation"] for case in profiles}, {"rvv", "ime"})
        self.assertTrue(all(case["threads"] == 1 and case["timing_mode"] == "end_to_end" for case in profiles))

    def test_ime_preflight_covers_each_intended_cpu_once(self):
        args = arguments(Path("."))
        args.command = "campaign"
        preflight = runner.ime_preflight_plan(args, runner.case_plan(args))
        self.assertEqual([case["cpu_ids"] for case in preflight], [[0], [1], [2], [3]])
        self.assertTrue(all(case["implementation"] == "ime" and case["ime_kernel"] == args.ime_kernel for case in preflight))

    def test_tuning_is_36_rvv_plus_8_independent_ime(self):
        with fixture_directory() as directory:
            root = Path(directory)
            sources = set()
            for tile, lmuls in (("8x4", ("mf8", "mf4", "mf2", "1", "2")), ("8x8", ("mf4", "mf2", "1", "2"))):
                for lmul in lmuls:
                    for unroll in (1, 2, 4, 8):
                        name = f"igemm_kernel_{tile}_zvl256b_lmul{lmul}_unroll{unroll}"
                        source = root / f"GEMM_RVV_FP32_INT8_{tile}_Baseline" / f"RVV_IGEMM_INT8_I8I32_{tile}" / name / (name + "_i8i32.c")
                        sources.add(str(source))
                for unroll in (1, 2, 4, 8):
                    name = f"ime_kernel_{tile}_zvl256b_lmul1_unroll{unroll}"
                    source = root / "IME_NATIVE_KERNELS" / f"IME_GEMM_INT8_I8I32_{tile}_NATIVE" / name / (name + ".c")
                    sources.add(str(source))
            args = arguments(root)
            args.command = "tuning"
            with mock.patch.object(Path, "is_file", lambda path: str(path) in sources):
                plan = runner.case_plan(args)
            rvv = [case for _, case in plan if case["implementation"] == "rvv"]
            ime = [case for _, case in plan if case["implementation"] == "ime"]
            self.assertEqual((len(rvv), len(ime)), (36, 8))
            self.assertTrue(all(case["ime_kernel"] == args.ime_kernel for case in rvv))
            self.assertTrue(all(case["rvv_kernel"] == args.rvv_kernel for case in ime))

    def test_interruption_preserves_completed_csv_rows(self):
        with fixture_directory() as directory:
            info = {"native_platform": False, "system_id": "fixture", "system": "Windows", "machine": "AMD64"}
            row = {"status": "OK", "publishable": False, "implementation": "reference"}
            with mock.patch.object(runner, "system_info", return_value=info), mock.patch.object(
                    runner, "run_case", side_effect=[([row], [], [row], False), KeyboardInterrupt]):
                result = runner.main(["repeatability", "--host-test", "--output", str(directory), "--m", "16", "--n", "16", "--k", "64"])
            campaign = next(Path(directory).glob("repeatability_*"))
            with (campaign / "raw_all_runs.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            self.assertIn("interrupted", (campaign / "error.json").read_text())
            self.assertEqual(result, 1)

    def test_failed_ime_preflight_stops_campaign_before_primary_cases(self):
        with fixture_directory() as directory:
            info = {"native_platform": True, "system_id": "fixture", "system": "Linux", "machine": "riscv64"}
            with mock.patch.object(runner, "system_info", return_value=info), mock.patch.object(
                    runner, "run_case", return_value=([], [], [{"status": "REJECTED"}], True)) as run:
                result = runner.main(["campaign", "--output", str(directory)])
            self.assertEqual(run.call_count, 1)
            self.assertTrue(run.call_args.args[-1])
            self.assertEqual(result, 1)

    def test_output_directories_are_unique(self):
        with fixture_directory() as directory:
            first = runner.unique_dir(Path(directory), "case")
            second = runner.unique_dir(Path(directory), "case")
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
