"""Synthetic in-memory software tests, never benchmark/paper measurements."""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
import unittest
import uuid


MODULE_PATH = Path(__file__).resolve().parents[1] / "analyze.py"
SPEC = importlib.util.spec_from_file_location("benchmark_analyze", MODULE_PATH)
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)


def fixture(**changes):
    """Artificial values exercise invariants only, not performance claims."""
    row = dict(M="16", N="32", K="64", implementation="rvv",
               timing_mode="end_to_end", threads="1", rvv_workers="1", ime_workers="0",
               schedule="static", rep="0", seed="7", profiled="false", status="OK",
               validation="PASS", accepted="true", publishable="true",
               execution_backend="native", datatype="int8_i32", system_id="synthetic-test-system",
               source_digest="synthetic-test-source", build_metadata='{"compiler":"test-only"}',
               cpu_ids="[0]", phase_aggregation="sum_worker_elapsed",
               timing_scope="total_elapsed", validation_scope="full",
               rvv_kernel="rvv_test", ime_kernel="ime_test", rvv_lmul="1", ime_lmul="2",
               rvv_tile="4x4", ime_tile="8x8", rvv_unroll="1", ime_unroll="2",
               total_sec="0.001", gops=str(2 * 16 * 32 * 64 / .001 / 1e9),
               packing_sec="", kernel_sec="", output_sec="", boundary_sec="")
    row.update(changes)
    if "total_sec" in changes and "gops" not in changes:
        row["gops"] = str(2 * int(row["M"]) * int(row["N"]) * int(row["K"]) / float(row["total_sec"]) / 1e9)
    return row


def normalized(**changes):
    return analyze.normalize_row(fixture(**changes))


def schema_fixture(**changes):
    """Synthetic values with the observed src/bench.c + run.py column schema."""
    fields = dict(timestamp_utc="2000-01-01T00:00:00Z", campaign_id="test-only-campaign",
                  case_id="test-only-case", group="test-only-group", host_test=False,
                  hardware_platform="synthetic/platform", counters_requested=False,
                  requested_cpu_ids="[0]", packing_in_scope=True, weight="0.5", chunk="1",
                  warmups="1", rvv_output_fused="true", boundary_scope="RVV_fused_IME_separate",
                  counter_scope="sum_worker_measured_regions", counters_status="DISABLED",
                  workers='[{"id":0,"cpu_before":0,"cpu_after":0,"strips":2}]',
                  cycles="", instructions="", cache_references="", cache_misses="", ipc="",
                  ime_output_sec="", ime_boundary_sec="", command='["test-only-executable"]',
                  rejection_reason="")
    fields.update(changes)
    return fixture(**fields)


def build_fixture(directory="/test-only/a", **changes):
    result = dict(schema_version=1, host_test=False, publishable=True, status="OK",
                  project_root="/test-only/source", binary=directory + "/bench",
                  compiler_command=["test-only-compiler"], compiler_version="test-only version",
                  build_flags=["-O3", "-std=c11", "-march=rv64gcv_zvl256b", "-I", "/test-only/source/src",
                               "-include", directory + "/selected_sources.h"],
                  commands=[["test-only-compiler", "-o", directory + "/bench"]],
                  sources={directory + "/kernel.c": "test-hash"}, source_universe={"kernel.c": "test-hash"},
                  source_digest="synthetic-test-source", hardware_target="SYNTHETIC_ONLY",
                  executable_sha256=directory, rvv_kernel="rvv_test", ime_kernel="ime_test",
                  rvv_lmul="1", rvv_tile="4x4", rvv_unroll=1, ime_lmul="2", ime_tile="8x8", ime_unroll=2)
    result.update(changes)
    return result


@contextlib.contextmanager
def fixture_directory():
    # Windows' sandbox cannot reopen tempfile's restrictive mode-0700 dirs.
    # This default-mode directory contains test fixtures only and is removed.
    directory = MODULE_PATH.parent / ("synthetic-analysis-test-" + uuid.uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        assert directory.parent == MODULE_PATH.parent
        shutil.rmtree(directory)


class AnalysisTests(unittest.TestCase):
    def assert_rejected(self, **changes):
        with self.assertRaises(ValueError):
            normalized(**changes)

    def test_rejected_and_reference_rows(self):
        for change in (dict(status="FAIL"), dict(validation="FAIL"), dict(accepted="false"),
                       dict(publishable="false"), dict(execution_backend="reference"),
                       dict(execution_backend="host_reference")):
            with self.subTest(change=change):
                self.assert_rejected(**change)

    def test_missing_grouping_metadata_rejected(self):
        for field in ("source_digest", "system_id", "cpu_ids", "rvv_tile", "build_metadata", "seed"):
            with self.subTest(field=field):
                row = fixture()
                del row[field]
                with self.assertRaisesRegex(ValueError, "missing required"):
                    analyze.normalize_row(row)
        self.assert_rejected(system_id="")
        self.assert_rejected(rvv_kernel="")

    def test_throughput_uses_declared_total_elapsed(self):
        self.assert_rejected(gops="1.23")
        self.assert_rejected(total_sec="nan")
        self.assert_rejected(gops="inf")
        row = normalized()
        self.assertAlmostEqual(row["gops"], 2 * row["M"] * row["N"] * row["K"] / row["total_sec"] / 1e9)

    def test_worker_and_affinity_consistency(self):
        self.assert_rejected(threads="2", cpu_ids="[0,1]")
        self.assert_rejected(cpu_ids="[]")
        self.assert_rejected(threads="2", rvv_workers="2", cpu_ids="[0,0]")
        self.assert_rejected(implementation="mixed")

    def test_grouping_does_not_pool_configurations(self):
        rows = [normalized(), normalized(rep="1"), normalized(seed="8"),
                normalized(cpu_ids="[1]"), normalized(rvv_lmul="2"),
                normalized(system_id="different-test-machine"),
                normalized(profiled="true", packing_sec=".0002", kernel_sec=".0007"),
                normalized(validation_scope="sampled"), normalized(new_hardware_flag="x")]
        summaries, _ = analyze.summarize(rows)
        self.assertEqual(len(summaries), 8)
        self.assertEqual(sorted(s["repetitions"] for s in summaries), [1] * 7 + [2])

    def test_configuration_column_absent_is_distinct_from_blank(self):
        rows = [normalized(), normalized(extra_configuration="")]
        self.assertEqual(len(analyze.summarize(rows)[0]), 2)

    def test_per_run_gops_mean_is_not_gops_of_mean_time(self):
        rows = [normalized(), normalized(rep="1", total_sec=".002")]
        summary = analyze.summarize(rows)[0][0]
        self.assertAlmostEqual(summary["gops_mean"], sum(r["gops"] for r in rows) / 2)
        self.assertNotAlmostEqual(summary["gops_mean"], 2 * 16 * 32 * 64 / summary["total_sec_mean"] / 1e9)

    def test_rvv_fused_fields_stay_null(self):
        rows = [normalized(profiled="true", packing_sec=".0002", kernel_sec=".0007")]
        summaries, repetitions = analyze.summarize(rows)
        table = analyze.section_v_rows(summaries, repetitions)[0]
        self.assertIsNone(table["output_sec_mean"])
        self.assertIsNone(table["boundary_sec_mean"])
        self.assertIsNone(table["output_sec_share_pct_mean"])
        self.assertIn("fused", table["kernel_label"])
        self.assertAlmostEqual(table["kernel_sec_share_pct_mean"], 70)
        self.assert_rejected(output_sec="0")

    def test_multicore_worker_times_have_no_wall_percentages(self):
        row = normalized(profiled="true", threads="8", rvv_workers="8", cpu_ids="[0,1,2,3,4,5,6,7]",
                         packing_sec=".002", kernel_sec=".006")
        shares, scope = analyze.phase_shares(row)
        self.assertTrue(all(value is None for value in shares.values()))
        self.assertIn("not additive wall", scope)

    def test_prepacked_setup_not_counted_in_phase_share(self):
        row = normalized(profiled="true", timing_mode="prepacked", packing_sec="10", kernel_sec=".0009")
        shares, _ = analyze.phase_shares(row)
        self.assertIsNone(shares["packing_sec"])
        self.assertAlmostEqual(shares["kernel_sec"], 90)

    def test_nonadditive_single_thread_has_no_shares(self):
        row = normalized(profiled="true", packing_sec=".002", kernel_sec=".003")
        self.assertTrue(all(value is None for value in analyze.phase_shares(row)[0].values()))

    def test_incompatible_scopes_never_yield_speedup(self):
        rvv = normalized()
        for changes in (dict(timing_mode="prepacked"), dict(datatype="different"),
                        dict(system_id="other"), dict(source_digest="other"),
                        dict(cpu_ids="[1]"), dict(validation_scope="sampled"),
                        dict(timing_scope="other"), dict(seed="8")):
            ime = normalized(implementation="ime", rvv_workers="0", ime_workers="1", **changes)
            summaries, _ = analyze.summarize([rvv, ime])
            self.assertIsNone(analyze.speedup_rows(summaries)[0]["speedup_vs_rvv"])

    def test_ambiguous_rvv_baseline_is_not_best_picked(self):
        rows = [normalized(), normalized(rvv_lmul="2", total_sec=".0005"),
                normalized(implementation="ime", rvv_workers="0", ime_workers="1", total_sec=".0008")]
        result = analyze.speedup_rows(analyze.summarize(rows)[0])[0]
        self.assertIsNone(result["speedup_vs_rvv"])
        self.assertEqual(result["baseline_candidates"], 2)
        self.assertIn("ambiguous", result["speedup_reason"])

    def test_mixed_matches_rvv_tuning_and_allows_independent_ime(self):
        common = dict(threads="8", cpu_ids="[0,1,2,3,4,5,6,7]")
        rows = [normalized(rvv_workers="8", **common),
                normalized(rvv_workers="8", rvv_lmul="4", total_sec=".0001", **common),
                normalized(implementation="mixed", rvv_workers="4", ime_workers="4", ime_lmul="8",
                           ime_kernel="independently_tuned", total_sec=".0005", **common)]
        result = analyze.speedup_rows(analyze.summarize(rows)[0])[0]
        self.assertEqual(result["baseline_candidates"], 1)
        self.assertAlmostEqual(result["speedup_vs_rvv"], 2)

    def test_profiled_runs_do_not_get_speedups(self):
        rows = [normalized(), normalized(implementation="ime", rvv_workers="0", ime_workers="1",
                                        profiled="true", kernel_sec=".0007")]
        result = analyze.speedup_rows(analyze.summarize(rows)[0])[0]
        self.assertIsNone(result["speedup_vs_rvv"])
        self.assertIn("profiled", result["speedup_reason"])

    def test_actual_schema_transient_fields_do_not_split_repetitions(self):
        first = schema_fixture(build_metadata=build_fixture())
        second = schema_fixture(rep="1", case_id="test-only-other-case", group="repeatability",
                                build_metadata=build_fixture("/test-only/b"), timestamp_utc="2000-01-01T00:01:00Z",
                                workers='[{"id":0,"cpu_before":0,"cpu_after":0,"strips":3}]')
        summaries, _ = analyze.summarize([analyze.normalize_row(first), analyze.normalize_row(second)])
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["repetitions"], 2)

    def test_build_normalization_preserves_compiler_and_unknown_flags(self):
        first = normalized(build_metadata=build_fixture())
        for build in (build_fixture(compiler_command=["test-only-compiler", "-O0"]),
                      build_fixture(compiler_version="other-version"),
                      build_fixture(build_flags=["-O0"]), build_fixture(new_codegen_flag="different")):
            with self.subTest(build=build):
                second = normalized(build_metadata=build)
                self.assertEqual(len(analyze.summarize([first, second])[0]), 2)
        for build in (build_fixture(status="FAILED"), build_fixture(publishable=False),
                      build_fixture(source_digest="mismatched-source")):
            self.assert_rejected(build_metadata=build)

    def test_counter_scope_and_hardware_counts_are_configuration(self):
        rows = [normalized(counter_scope="a", core_count="8"),
                normalized(counter_scope="b", core_count="8"),
                normalized(counter_scope="a", core_count="16")]
        self.assertEqual(len(analyze.summarize(rows)[0]), 3)

    def test_counter_values_aggregate_but_are_excluded_from_primary_speedups(self):
        rows = [normalized(counters_requested=True, counters_status="OK", cycles="10", instructions="20", ipc="2"),
                normalized(rep="1", counters_requested=True, counters_status="OK", cycles="20", instructions="30", ipc="1.5")]
        summaries, _ = analyze.summarize(rows)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["cycles_mean"], 15)
        self.assertEqual(summaries[0]["ipc_mean"], 1.75)
        self.assertEqual(summaries[0]["counters_statuses"], ["OK"])
        rows.append(normalized(implementation="ime", rvv_workers="0", ime_workers="1", counters_requested=True))
        result = analyze.speedup_rows(analyze.summarize(rows)[0])[0]
        self.assertIsNone(result["speedup_vs_rvv"])
        self.assertIn("counter-instrumented", result["speedup_reason"])

    def test_mixed_ime_only_phases_are_preserved_without_wall_shares(self):
        row = normalized(implementation="mixed", threads="8", rvv_workers="4", ime_workers="4",
                         cpu_ids="[0,1,2,3,4,5,6,7]", profiled=True,
                         kernel_sec=".003", ime_output_sec=".0002", ime_boundary_sec=".0001")
        summaries, repetitions = analyze.summarize([row])
        table = analyze.section_v_rows(summaries, repetitions)[0]
        self.assertIsNone(table["output_sec_mean"])
        self.assertEqual(table["ime_output_sec_mean"], .0002)
        self.assertTrue(all(table[phase + "_share_pct_mean"] is None for phase in analyze.PHASES))

    def test_central_comparison_requires_matching_four_plus_four(self):
        common = dict(threads="8", cpu_ids="[0,1,2,3,4,5,6,7]")
        rvv = normalized(rvv_workers="8", **common)
        static = normalized(implementation="mixed", rvv_workers="4", ime_workers="4", schedule="static", **common)
        dynamic = normalized(implementation="mixed", rvv_workers="4", ime_workers="4", schedule="dynamic", **common)
        good = analyze.summarize([rvv, static, dynamic])[0]
        self.assertEqual(len(analyze.central_comparisons(good)), 1)
        for altered in (normalized(implementation="mixed", rvv_workers="6", ime_workers="2", schedule="dynamic", **common),
                        normalized(implementation="mixed", rvv_workers="4", ime_workers="4", schedule="dynamic", ime_lmul="8", **common)):
            self.assertEqual(analyze.central_comparisons(analyze.summarize([rvv, static, altered])[0]), [])

    def test_unprofiled_phase_numbers_and_missing_active_kernel_are_rejected(self):
        self.assert_rejected(kernel_sec=".0001")
        self.assert_rejected(rvv_kernel=None)
        self.assert_rejected(rvv_lmul="null")

    def test_repetition_ids_are_local_to_a_runner_case(self):
        with fixture_directory() as campaign:
            cases = [schema_fixture(case_id="synthetic-case-a"), schema_fixture(case_id="synthetic-case-b")]
            with (campaign / "accepted_runs.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=cases[0].keys())
                writer.writeheader()
                writer.writerows(cases + cases[:1])
            rows, rejected = analyze.load_accepted(campaign)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(analyze.summarize(rows)[0][0]["repetitions"], 2)

    def test_plot_smoke_uses_synthetic_fixtures_and_removes_outputs(self):
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("optional matplotlib is not installed")
        rows = []
        for mode in ("prepacked", "end_to_end"):
            for impl in ("rvv", "ime"):
                base = dict(implementation=impl, rvv_workers=str(int(impl == "rvv")),
                            ime_workers=str(int(impl == "ime")), timing_mode=mode)
                rows.extend([normalized(**base), normalized(rep="1", total_sec=".0012", **base)])
                phases = dict(profiled=True, packing_sec=".0001" if mode == "end_to_end" else "", kernel_sec=".0005")
                if impl == "ime":
                    phases.update(output_sec=".0001", boundary_sec=".0001")
                rows.append(normalized(**base, **phases))
        common = dict(threads="8", cpu_ids="[0,1,2,3,4,5,6,7]")
        rows.append(normalized(rvv_workers="8", **common))
        for schedule in ("static", "dynamic"):
            rows.append(normalized(implementation="mixed", rvv_workers="4", ime_workers="4", schedule=schedule, **common))
        summaries, repetitions = analyze.summarize(rows)
        with fixture_directory() as output:
            self.assertEqual(analyze.plot_results(output, summaries, repetitions), [])
            plots = list((output / "plots").glob("*.png"))
            self.assertTrue(plots)
            for prefix in ("singlecore_rvv_ime_prepacked", "singlecore_rvv_ime_end_to_end", "singlecore_components_",
                           "eight_core_three_way_", "repeatability_", "tuning_"):
                self.assertTrue(any(path.name.startswith(prefix) for path in plots), prefix)
            for path in plots:
                self.assertGreater(path.stat().st_size, 1000)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_reader_ignores_legacyraw_and_deduplicates(self):
        with fixture_directory() as temporary:
            campaign = Path(temporary)
            # Explicit test-only temporary data, removed on test completion.
            with (campaign / "accepted_runs.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fixture().keys())
                writer.writeheader()
                writer.writerows([fixture(), fixture(), fixture(status="FAIL", rep="2")])
            (campaign / "legacyraw.csv").write_text("not even valid input\n")
            rows, rejected = analyze.load_accepted(campaign)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rejected), 2)

    def test_empty_campaign_never_produces_plots_or_fake_data(self):
        with fixture_directory() as temporary:
            campaign, output = Path(temporary) / "campaign", Path(temporary) / "analysis"
            campaign.mkdir()
            (campaign / "accepted_runs.csv").write_text(",".join(fixture().keys()) + "\n")
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                self.assertEqual(analyze.main(["--input", str(campaign), "--output", str(output)]), 0)
            self.assertIn("No accepted hardware data", captured.getvalue())
            self.assertFalse((output / "plots").exists())
            report = json.loads((output / "analysis.json").read_text())
            self.assertEqual(report["accepted_repetitions"], 0)
            with (output / "summary.csv").open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(analyze.main(["--input", str(campaign), "--output", str(output)]), 2)

    def test_direct_legacy_file_input_refused(self):
        with fixture_directory() as temporary:
            legacy = Path(temporary) / "legacyraw.csv"
            legacy.write_text("irrelevant\n")
            with self.assertRaises(ValueError):
                analyze.load_accepted(legacy)


if __name__ == "__main__":
    unittest.main()
