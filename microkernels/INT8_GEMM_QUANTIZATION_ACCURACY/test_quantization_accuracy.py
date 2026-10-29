"""Check numerical definitions and prevent incomplete/mislabeled plots.

Run: python3 -m unittest discover -s <this directory> -p 'test_*.py' -v
These tests exercise experiment logic; they do not certify native K1 hardware.
"""
import csv
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import quantization_accuracy as qa
import plot_quantization_accuracy as plot


class NumericalTests(unittest.TestCase):
    def test_exactly_representable_product(self):
        a = np.array([[127., 0., -127.], [1., 2., 3.]])
        b = np.array([[127., 0.], [0., 127.], [1., 2.]])
        for scheme in qa.SCHEMES:
            ai, bi, sa, sb = qa.quantize_inputs(a, b, scheme)
            expected, overflow = qa.integer_reference(ai, bi)
            self.assertEqual(overflow, 0)
            # Independent Python integer loops verify NumPy's INT64 reference.
            hand = np.array([[sum(int(ai[i, k])*int(bi[k, j]) for k in range(3))
                              for j in range(2)] for i in range(2)], dtype=np.int64)
            np.testing.assert_array_equal(expected, hand)
            actual = qa.dequantize(expected, sa, sb)
            self.assertTrue(np.isfinite(actual).all())
        ai, bi, sa, sb = qa.quantize_inputs(a, b, "per_tensor")
        np.testing.assert_array_equal(ai, a.astype(np.int8))
        np.testing.assert_array_equal(bi, b.astype(np.int8))
        np.testing.assert_array_equal(qa.dequantize(ai.astype(np.int64) @ bi.astype(np.int64), sa, sb), a @ b)

    def test_scale_axes_and_zero_rows_columns(self):
        a = np.array([[0., 0.], [127., -63.5]])
        b = np.array([[0., 254.], [0., -127.]])
        ai, bi, sa, sb = qa.quantize_inputs(a, b, "per_row_column")
        np.testing.assert_array_equal(sa, [1., 1.])
        np.testing.assert_array_equal(sb, [1., 2.])
        self.assertEqual(int(ai[1, 1]), -64)  # ties-to-even
        self.assertEqual(int(bi[1, 1]), -64)
        self.assertTrue((ai[0] == 0).all())
        self.assertTrue((bi[:, 0] == 0).all())

    def test_symmetric_range_and_ties(self):
        a = np.array([[127., 0.5, 1.5, 2.5, -0.5, -1.5]])
        b = np.full((6, 1), 127.)
        ai, bi, _, _ = qa.quantize_inputs(a, b, "per_tensor")
        np.testing.assert_array_equal(ai[0], [127, 0, 2, 2, 0, -2])
        self.assertGreaterEqual(int(ai.min()), -127)
        self.assertLessEqual(int(bi.max()), 127)

    def test_metrics_known_answer(self):
        ref = np.array([[3., 4.]])
        actual = np.array([[3., 5.]])
        m = qa.error_metrics(ref, actual)
        self.assertAlmostEqual(m["relative_frobenius_error"], 0.2)
        self.assertAlmostEqual(m["rmse"], 1 / math.sqrt(2))
        self.assertAlmostEqual(m["normalized_max_error"], 0.25)
        self.assertAlmostEqual(m["sqnr_db"], -20 * math.log10(0.2))
        self.assertEqual(m["max_absolute_error"], 1.)

    def test_zero_reference_and_exact_result(self):
        zero = np.zeros((2, 2))
        m = qa.error_metrics(zero, zero)
        self.assertEqual(m["relative_frobenius_error"], 0)
        self.assertEqual(m["normalized_max_error"], 0)
        self.assertTrue(math.isinf(m["sqnr_db"]))
        with self.assertRaises(ValueError):
            qa.error_metrics(zero, np.ones((2, 2)))

    def test_safely_widened_integer_differences(self):
        actual = np.array([[np.iinfo(np.int32).min]], dtype=np.int32)
        ref = np.array([[np.iinfo(np.int32).max]], dtype=np.int64)
        self.assertEqual(qa.compare_integer(actual, ref), (1, 4294967295))

    def test_overflow_reference(self):
        # The experiment itself refuses this K; the reference still detects it.
        a = np.full((1, 131072), -128, dtype=np.int8)
        b = np.full((131072, 1), -128, dtype=np.int8)
        result, overflow = qa.integer_reference(a, b)
        self.assertEqual(int(result[0, 0]), 2**31)
        self.assertEqual(overflow, 1)

    def test_reproducible_inputs_and_distinct_seeds(self):
        seeds = []
        for d, name in enumerate(qa.DISTRIBUTIONS):
            for sample in range(3):
                seed = qa.input_seed(42, d, sample)
                seeds.append(seed)
                a, b = qa.generate_inputs(name, seed, 8, 4, 8)
                c, e = qa.generate_inputs(name, seed, 8, 4, 8)
                np.testing.assert_array_equal(a, c)
                np.testing.assert_array_equal(b, e)
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            qa.quantize_inputs(np.array([[np.nan]]), np.ones((1, 1)), "per_tensor")
        with self.assertRaises(ValueError):
            qa.quantize_inputs(np.ones((2, 3)), np.ones((2, 2)), "per_tensor")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "run"
        args = qa.parser().parse_args([
            "--software-verification", "--m", "8", "--n", "4", "--k", "8",
            "--samples", "2", "--no-plot", "--output-dir", str(self.directory),
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            qa.run_experiment(args)
        self.csv_path = self.directory / "quantization_accuracy.csv"
        self.manifest_path = self.directory / "manifest.json"

    def read_rows(self):
        with self.csv_path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def write_rows(self, rows):
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=qa.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        manifest = json.loads(self.manifest_path.read_text())
        manifest["data_sha256"] = qa.sha256(self.csv_path)
        qa.save_json(self.manifest_path, manifest)

    def test_complete_software_pipeline_explicit_label(self):
        rows, manifest = plot.load_and_validate(self.csv_path, True)
        self.assertEqual(len(rows), 12)
        self.assertEqual(manifest["execution_kind"], "software_verification")
        self.assertEqual(set(r["backend"] for r in rows), {"scalar"})
        with self.assertRaises(plot.ValidationError):
            plot.load_and_validate(self.csv_path)

    def test_incomplete_and_duplicate_data_rejected(self):
        rows = self.read_rows()
        self.write_rows(rows[:-1])
        with self.assertRaises(plot.ValidationError):
            plot.load_and_validate(self.csv_path, True)
        self.write_rows(rows + [rows[0]])
        with self.assertRaises(plot.ValidationError):
            plot.load_and_validate(self.csv_path, True)

    def test_nonfinite_errors_rejected(self):
        rows = self.read_rows()
        rows[0]["relative_frobenius_error"] = "nan"
        self.write_rows(rows)
        with self.assertRaises(plot.ValidationError):
            plot.load_and_validate(self.csv_path, True)

    def test_modified_csv_hash_rejected(self):
        with self.csv_path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaises(plot.ValidationError):
            plot.load_and_validate(self.csv_path, True)

    def test_repeated_input_seed_rejected(self):
        rows = self.read_rows()
        first_seed = rows[0]["seed"]
        for row in rows:
            if row["distribution"] == "uniform" and row["sample_index"] == "1":
                row["seed"] = first_seed
        self.write_rows(rows)
        with self.assertRaises(plot.ValidationError):
            plot.load_and_validate(self.csv_path, True)

    def test_native_pair_validation_using_temporary_fixtures_only(self):
        # This validates the reader's pairing rules, not native execution.
        # No fixture is exported or retained as a hardware measurement.
        rows = self.read_rows()
        paired = []
        for row in rows:
            for name in ("rvv", "ime"):
                paired.append(dict(row, execution_kind="native_k1", backend=name))
        manifest = json.loads(self.manifest_path.read_text())
        manifest.update(execution_kind="native_k1", backends=["rvv", "ime"],
                        planned_rows=24, completed_rows=24)
        qa.save_json(self.manifest_path, manifest)
        self.write_rows(paired)
        parsed, _ = plot.load_and_validate(self.csv_path)
        self.assertEqual(len(parsed), 24)
        paired[1]["relative_frobenius_error"] = str(float(paired[1]["relative_frobenius_error"]) + 0.01)
        self.write_rows(paired)
        with self.assertRaises(plot.ValidationError):
            plot.load_and_validate(self.csv_path)

    def test_inconsistent_paired_seed_rejected(self):
        rows = self.read_rows()
        rows[1]["seed"] = str(int(rows[1]["seed"]) + 1)
        self.write_rows(rows)
        with self.assertRaises(plot.ValidationError):
            plot.load_and_validate(self.csv_path, True)

    def test_failed_correctness_rejected(self):
        rows = self.read_rows()
        rows[0]["mismatches"] = "1"
        self.write_rows(rows)
        with self.assertRaises(plot.ValidationError):
            plot.load_and_validate(self.csv_path, True)

    def test_incomplete_manifest_rejected(self):
        manifest = json.loads(self.manifest_path.read_text())
        manifest["status"] = "running"
        qa.save_json(self.manifest_path, manifest)
        with self.assertRaises(plot.ValidationError):
            plot.load_and_validate(self.csv_path, True)

    def test_native_failure_preserves_failed_manifest(self):
        directory = Path(self.temporary.name) / "failed"
        args = qa.parser().parse_args([
            "--m", "8", "--n", "4", "--k", "8", "--samples", "1", "--no-plot",
            "--build-dir", str(Path(self.temporary.name) / "build"), "--output-dir", str(directory),
        ])
        with patch.object(qa, "NativeBackend", side_effect=RuntimeError("no native hardware")):
            with self.assertRaises(RuntimeError):
                qa.run_experiment(args)
        self.assertEqual(json.loads((directory / "manifest.json").read_text())["status"], "failed")

    def test_existing_output_never_overwritten(self):
        args = qa.parser().parse_args(["--software-verification", "--output-dir", str(self.directory)])
        with self.assertRaises(FileExistsError):
            qa.run_experiment(args)


if __name__ == "__main__":
    unittest.main()
