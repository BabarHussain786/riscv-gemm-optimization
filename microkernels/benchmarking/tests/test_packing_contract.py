"""Verify timer-free packing reuse without compiling or running a kernel."""
import re
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parent
CLEAN_DISPATCH = HERE / "src" / "rvv_packing_dispatch.h"
LEGACY_DISPATCH = (
    PROJECT_ROOT
    / "HETEROGENEOUS_RVV_IME_OPENMP_GEMM"
    / "src"
    / "openmp_kernel_dispatch.h"
)


def without_comments(source):
    """Keep literals intact while discarding C comments for source comparison."""
    tokens = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*', re.S)
    return tokens.sub(
        lambda match: " " if match[0].startswith(("/*", "//")) else match[0],
        source,
    )


def normalized(source):
    return re.sub(r"\s+", "", without_comments(source))


def braced_construct(source, pattern):
    """Extract a uniquely identified function/loop including nested braces."""
    source = without_comments(source)
    matches = list(re.finditer(pattern, source))
    if len(matches) != 1:
        raise AssertionError(f"Expected one source construct for {pattern!r}, found {len(matches)}")
    start = matches[0].start()
    opening = source.find("{", matches[0].end())
    if opening < 0:
        raise AssertionError(f"Missing opening brace for {pattern!r}")
    depth = 0
    # These selected helpers and loops contain no string or character literals.
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"Missing closing brace for {pattern!r}")


class PackingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = CLEAN_DISPATCH.read_text(encoding="utf-8")
        cls.legacy = LEGACY_DISPATCH.read_text(encoding="utf-8")
        function = r"static\s+int\s+call_packed_rvv_tile_kernel\s*\([^;{]*\)"
        cls.clean_packer = braced_construct(cls.clean, function)
        cls.legacy_packer = braced_construct(cls.legacy, function)

    def test_block_sizes_match_original_for_full_and_boundary_tiles(self):
        helper = r"static\s+BLASLONG\s+packed_block_size\s*\([^;{]*\)"
        self.assertEqual(
            normalized(braced_construct(self.clean, helper)),
            normalized(braced_construct(self.legacy, helper)),
        )

    def test_packer_signature_and_buffer_checks_match_original(self):
        # The only addition before the original A loop is its legacy clock read.
        prefix = re.compile(r"while\s*\(\s*m_top\s*<\s*M\s*\)")
        clean = prefix.split(without_comments(self.clean_packer), maxsplit=1)[0]
        legacy = prefix.split(without_comments(self.legacy_packer), maxsplit=1)[0]
        legacy, removed = re.subn(
            r"\bdouble\s+pack_t0\s*=\s*omp_get_wtime\s*\(\s*\)\s*;",
            "",
            legacy,
        )
        self.assertEqual(removed, 1, "Reinspect changes to legacy timing instrumentation")
        self.assertEqual(normalized(clean), normalized(legacy))

    def test_a_and_b_packing_loops_match_original_exactly(self):
        for label, pattern in (
            ("A", r"while\s*\(\s*m_top\s*<\s*M\s*\)"),
            ("B", r"while\s*\(\s*n_top\s*<\s*N\s*\)"),
        ):
            with self.subTest(panel=label):
                self.assertEqual(
                    normalized(braced_construct(self.clean_packer, pattern)),
                    normalized(braced_construct(self.legacy_packer, pattern)),
                )

    def test_reused_header_has_no_legacy_phase_instrumentation(self):
        clean = without_comments(self.clean)
        for pattern in (
            r"\bomp_get_wtime\b",
            r"\bopenmp_(?:input_packing|kernel)_time_sec\b",
            r"#\s*pragma\s+omp\s+atomic\b",
        ):
            with self.subTest(forbidden=pattern):
                self.assertNotRegex(clean, pattern)
        self.assertRegex(without_comments(self.legacy), r"\bomp_get_wtime\b")


if __name__ == "__main__":
    unittest.main()
