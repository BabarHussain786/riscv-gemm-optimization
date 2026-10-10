#!/usr/bin/env bash
# Figure 2 only: fixed 1024^3 INT8 strong scaling on SpaceMiT K1.
# One new launcher; uses the existing, UNMODIFIED benchmarking C adapters.
# Requires Bash, Python 3.8+, and a native GCC with RVV intrinsics/OpenMP.
# No pip, jq, sudo, governor changes, source patching, or dataset overwrites.
set -Eeuo pipefail
TASK_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python3}" - "$TASK_SCRIPT_DIR" "$@" <<'PY'
"""Orchestration lives inside this .sh; no additional script is installed."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shlex
import signal
import statistics
import subprocess
import sys
from datetime import datetime, timezone
import uuid

ROOT = Path(os.environ.get('PROJECT_ROOT', sys.argv[1])).resolve()
BENCH = ROOT / 'benchmarking'
N = 1024                       # Intentionally not an environment override.
LMULS = {'8x4': ('mf8', 'mf4', 'mf2', '1', '2')}
LMUL_VALUE = {'mf8': .125, 'mf4': .25, 'mf2': .5, '1': 1, '2': 2}
METRICS = ('total_sec', 'gops', 'packing_sec', 'kernel_sec', 'output_sec',
           'boundary_sec', 'cycles', 'instructions', 'ipc',
           'cache_references', 'cache_misses')
SCOPE = '''FIGURE 2: NEW, INDEPENDENT STRONG-SCALING CAMPAIGN
Run from the project root on K1:
  bash run_fig02_strong_scaling_complete.sh --check
  bash run_fig02_strong_scaling_complete.sh
  UNROLLS=1 bash run_fig02_strong_scaling_complete.sh
  RVV_LMULS=all bash run_fig02_strong_scaling_complete.sh
Optional environment settings: RUNS=7 WARMUPS=2 SEED=42 CC=gcc
TIMEOUT_SEC=1800 COLLECT_COUNTERS=1 RVV_CPUS=0,1,2,3,4,5,6,7 IME_CPUS=0,1,2,3.
PROJECT_ROOT locates the existing source tree; RESULT_ROOT selects a result
parent directory. Every execution creates a new campaign below that parent.
measurements.csv contains accepted numerical samples; summary.csv contains
per-point statistics. raw_all_runs.csv preserves parsed rejected rows too;
failures.csv/attempts.csv and per-case logs explain every missing point.

M=N=K=1024; signed INT8 inputs, INT32 output, C <- C + A*B; alpha=1.
Default: software tile 8x4, LMUL1, U1/U2/U4/U8, RVV 1/2/4/8 cores,
IME 1/2/4 cores. Each unroll/LMUL is a SEPARATE fixed-kernel scaling series.
Use UNROLLS=1 for only the configuration used in the current Figure 2.
RVV_LMULS=all enables all five supported RVV LMULs, not just the default 1.
Only 8x4 is used: the current Figure 2 does not contain an 8x8 series.
IME LMUL is always 1; unsupported experimental mf2 variants are excluded.

PRIMARY pass: unprofiled total_sec is elapsed GEMM wall time. It includes
assignment, input packing, compute, applicable output handling and timed
barriers. Allocation, random initialization, C reset, reference calculation,
warmups, validation and cleanup are OUTSIDE that timer. All samples are
validated against an independent INT64 reference. GOPS=2*M*N*K/time/1e9.

PROFILE pass: separate invocation, NOT primary performance evidence.
packing_sec/kernel_sec/output_sec/boundary_sec are sums of worker elapsed
times, NOT a decomposition of wall time. Timer overhead is not subtracted.
RVV packing_sec includes A+B packing; kernel_sec includes computation,
loading/updating C and output stores. RVV output_sec and boundary_sec are
NA/FUSED_WITH_KERNEL, never zero and never estimated by subtraction.
IME packing_sec includes compact-B preparation AND A/B native panel packing.
IME kernel_sec times native_accumulate including its temporary tile stores;
output_sec times scatter_output to C; boundary_sec times scalar cleanup.
For aligned 1024^3 and the supported unrolls, IME boundary_sec can truly be 0.
Neither backend exposes pure-arithmetic-only time; compute_only_sec stays NA.
Input-A vs input-B packing, allocation, initialization, synchronization,
reference and validation durations are not individually instrumented: NA.
The adapter preallocates workspaces for both paths; these timings are NOT
the legacy public IME entry point with allocation inside the kernel timer.

COUNTER pass: independent invocation. Counters cover each pinned worker's
assignment+packing+compute+output region, not whole-process initialization
or validation and not final waiting. IPC=sum instructions/sum cycles.
The existing driver opens cycles/instructions/cache references/cache misses
as one group; unavailable or multiplexed events make all four values NA.
No privileged settings are changed. Counter-pass times are never used for
speedup. CACHE/IPC unavailability does not discard valid primary timings.

All valid repetitions (including slow ones) are kept. Summary SD is sample
SD (n-1); quartiles use linear interpolation. Each point's summary separates
primary/profile/counter roles. Speedup uses the SAME kernel's complete
single-core PRIMARY mean; efficiency=speedup/cores. Missing points are NA.
No automatic best-kernel selection, mixed-backend execution, or weak scaling.
Do not splice this new campaign into older plots with different timer scopes.

Existing project files are only read. Each run creates a unique result folder.
Failed builds, timeouts, validation failures and unavailable CPUs are logged;
remaining points still run. Ctrl-C/termination stops the campaign intentionally.
Exit 0=all requested measurements available; 1=incomplete/failed/changed source;
2=configuration/platform error; 3=timings complete but counters unavailable;
130=user interruption. Inspect completion.json before using the data.
'''


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                   allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def csv_write(path, rows, initial=()):
    fields = list(initial) + sorted({k for r in rows for k in r} - set(initial))
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: 'NA' if v is None else json.dumps(v, sort_keys=True)
                             if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def words(name, default, choices):
    result = os.environ.get(name, default).replace(',', ' ').split()
    if not result or len(set(result)) != len(result) or any(x not in choices for x in result):
        raise ValueError('%s must contain distinct values from %s' % (name, choices))
    return result


def integer(name, default, minimum=1, maximum=100000):
    value = int(os.environ.get(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError('%s must be in [%s,%s]' % (name, minimum, maximum))
    return value


def cpu_list(name, default, required):
    values = [int(x) for x in os.environ.get(name, default).split(',')]
    if len(values) < required or len(values) != len(set(values)) or any(x < 0 or x >= 1024 for x in values):
        raise ValueError('%s needs at least %d distinct CPU IDs (0..1023)' % (name, required))
    return values


def configuration():
    for dimension in ('M', 'N', 'K'):
        if dimension in os.environ and os.environ[dimension] != '1024':
            raise ValueError('Figure 2 requires %s=1024; workload changes are not allowed' % dimension)
    c = dict(tiles=words('TILES', '8x4', LMULS),
             unrolls=[int(x) for x in words('UNROLLS', '1 2 4 8', ('1', '2', '4', '8'))],
             lmuls=words('RVV_LMULS', '1', tuple(LMUL_VALUE) + ('all',)),
             repetitions=integer('RUNS', 7), warmups=integer('WARMUPS', 2, 0),
             seed=integer('SEED', 42, 1, 4294967295),
             counters=bool(integer('COLLECT_COUNTERS', 1, 0, 1)),
             timeout=integer('TIMEOUT_SEC', 1800),
             cc=os.environ.get('CC', 'gcc'),
             rvv_cpus=cpu_list('RVV_CPUS', '0,1,2,3,4,5,6,7', 8),
             ime_cpus=cpu_list('IME_CPUS', '0,1,2,3', 4))
    if 'all' in c['lmuls'] and c['lmuls'] != ['all']:
        raise ValueError('RVV_LMULS=all must not be combined with other values')
    if c['lmuls'] != ['all']:
        for tile in c['tiles']:
            if any(x not in LMULS[tile] for x in c['lmuls']):
                raise ValueError('Unsupported LMUL for ' + tile)
    if not shlex.split(c['cc']):
        raise ValueError('CC must not be empty')
    return c


def plan(c):
    points = []
    for tile in c['tiles']:
        for u in c['unrolls']:
            # IME points are not duplicated when RVV sweeps several LMULs.
            for backend in ('rvv', 'ime'):
                lmuls = (LMULS[tile] if c['lmuls'] == ['all'] else c['lmuls']) if backend == 'rvv' else ['1']
                for lmul in lmuls:
                    active = '%s_kernel_%s_zvl256b_lmul%s_unroll%d' % ('igemm' if backend == 'rvv' else 'ime', tile, lmul, u)
                    companion_lmul = (LMULS[tile][0] if c['lmuls'] == ['all'] else c['lmuls'][0])
                    rvv = active if backend == 'rvv' else 'igemm_kernel_%s_zvl256b_lmul%s_unroll%d' % (tile, companion_lmul, u)
                    ime = active if backend == 'ime' else 'ime_kernel_%s_zvl256b_lmul1_unroll%d' % (tile, u)
                    series = '%s_%s_lmul%s_u%d' % (backend, tile, lmul, u)
                    for cores in ((1, 2, 4, 8) if backend == 'rvv' else (1, 2, 4)):
                        points.append(dict(point_id=series + '_p%d' % cores,
                                           series_id=series, backend=backend, tile=tile,
                                           lmul=lmul, lmul_numeric=LMUL_VALUE[lmul], unroll=u,
                                           cores=cores, cpus=c[backend + '_cpus'][:cores],
                                           rvv_kernel=rvv, ime_kernel=ime, active_kernel=active))
    return points


def numeric(value, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def phase_gate(rows, role, backend):
    issues = []
    for row in rows:
        for key in ('packing_sec', 'kernel_sec', 'output_sec', 'boundary_sec'):
            value = row.get(key)
            measured = role == 'profile' and (backend == 'ime' or key in ('packing_sec', 'kernel_sec'))
            if measured and not numeric(value):
                issues.append('missing/invalid measured ' + key)
            if not measured and value is not None:
                issues.append('unmeasured/fused ' + key + ' must be null')
        if row.get('rvv_output_fused') != (backend == 'rvv'):
            issues.append('incorrect output fusion flag')
        status = row.get('counters_status')
        if role != 'counters' and status != 'DISABLED':
            issues.append('counters active in primary/profile pass')
        if role == 'counters':
            if status == 'OK':
                if not all(numeric(row.get(k)) for k in METRICS[6:]):
                    issues.append('counter status OK without numerical counters')
                elif row['cycles'] <= 0 or not math.isclose(row['ipc'], row['instructions'] / row['cycles'], rel_tol=1e-8):
                    issues.append('invalid region IPC')
            elif status != 'UNAVAILABLE_OR_MULTIPLEXED' or any(row.get(k) is not None for k in METRICS[6:]):
                issues.append('unsupported counters must be explicitly null')
    return issues


def worker_gate(rows, point, c):
    issues = []
    expected_ime = point['cores'] if point['backend'] == 'ime' else 0
    for row in rows:
        if row.get('seed') != c['seed'] or row.get('schedule') != 'static':
            issues.append('seed/schedule mismatch')
        if row.get('ime_workers') != expected_ime or row.get('rvv_workers') != point['cores'] - expected_ime:
            issues.append('worker backend counts mismatch')
        workers = row.get('workers', [])
        if not isinstance(workers, list) or len(workers) != point['cores']:
            issues.append('missing worker records')
            continue
        for i, worker in enumerate(workers):
            if not isinstance(worker, dict) or worker.get('id') != i or any(
                    worker.get(key) != point['cpus'][i] for key in ('cpu_before', 'cpu_after')):
                issues.append('worker affinity/identity mismatch')
            elif worker.get('strips') != 32 // point['cores']:
                issues.append('static output-strip assignment mismatch')
    return issues


def stats(values):
    values = sorted(x for x in values if numeric(x))
    def percentile(q):
        position = (len(values) - 1) * q
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)
    if not values:
        return dict(n=0, mean=None, median=None, sd=None, min=None, max=None, q25=None, q75=None)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                sd=statistics.stdev(values) if len(values) > 1 else None,
                min=values[0], max=values[-1], q25=percentile(.25), q75=percentile(.75))


def summaries(points, accepted, c, provenance_ok):
    output = []
    for point in points:
        row = dict(point, M=N, N=N, K=N, expected_samples_per_role=c['repetitions'],
                   phase_aggregation='sum_worker_elapsed', source_unchanged=provenance_ok)
        for role in ('primary', 'profile', 'counters'):
            records = [r for r in accepted if r['point_id'] == point['point_id'] and r['role'] == role]
            row[role + '_samples'] = len(records)
            relevant = {'primary': METRICS[:2], 'profile': ('total_sec',) + METRICS[2:6],
                        'counters': METRICS[6:]}[role]
            for metric in relevant:
                for name, value in stats([r.get(metric) for r in records]).items():
                    row[role + '_' + metric + '_' + name] = value
        row['primary_complete'] = row['primary_samples'] == c['repetitions'] and provenance_ok
        row['profile_complete'] = row['profile_samples'] == c['repetitions'] and provenance_ok
        row['ipc_complete'] = row['counters_ipc_n'] == c['repetitions'] and provenance_ok
        output.append(row)
    for row in output:
        baseline = next(x for x in output if x['series_id'] == row['series_id'] and x['cores'] == 1)
        ready = row['primary_complete'] and baseline['primary_complete']
        row['speedup'] = baseline['primary_total_sec_mean'] / row['primary_total_sec_mean'] if ready else None
        row['parallel_efficiency'] = row['speedup'] / row['cores'] if ready else None
        row['figure_ready'] = row['primary_complete'] and row['profile_complete'] and (row['ipc_complete'] or not c['counters'])
    return output


def launch(argv, folder, timeout):
    json_write(folder / 'command.json', dict(argv=argv, started_utc=stamp()))
    # New process group allows timeout/interrupt to stop the compiler or all
    # OpenMP workers, not leave a benchmark consuming cores in the background.
    with (folder / 'stdout.log').open('w', encoding='utf-8') as out, (folder / 'stderr.log').open('w', encoding='utf-8') as err:
        process = subprocess.Popen(argv, cwd=folder, stdout=out, stderr=err,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            code = 124
    json_write(folder / 'process.json', dict(return_code=code, finished_utc=stamp(), timed_out=code == 124))
    return code


def self_test():
    # Synthetic values are assertions only; never saved as benchmark data.
    fake = dict(packing_sec=1., kernel_sec=2., output_sec=None, boundary_sec=None,
                rvv_output_fused=True, counters_status='DISABLED')
    assert not phase_gate([fake], 'profile', 'rvv')
    assert phase_gate([{**fake, 'output_sec': 0.}], 'profile', 'rvv')
    assert phase_gate([fake], 'primary', 'rvv')
    assert phase_gate([{**fake, 'kernel_sec': float('nan')}], 'profile', 'rvv')
    ime = {**fake, 'output_sec': .1, 'boundary_sec': 0., 'rvv_output_fused': False}
    assert not phase_gate([ime], 'profile', 'ime')
    assert stats([])['mean'] is None and stats([1])['sd'] is None
    assert stats([1, 3])['mean'] == 2 and math.isclose(stats([1, 3])['sd'], math.sqrt(2))
    print('Self-test PASS: fused/missing phase guards, finite values, IME boundary zero, sample statistics.')


def main():
    cli = argparse.ArgumentParser(description='One-file Figure 2 launcher; existing sources stay unchanged.',
                                  epilog=SCOPE, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = cli.add_mutually_exclusive_group()
    actions.add_argument('--check', action='store_true', help='Read-only source/configuration check; no K1 required')
    actions.add_argument('--self-test', action='store_true', help='Offline parser/statistics tests; no measurements')
    args = cli.parse_args(sys.argv[2:])
    if args.self_test:
        self_test()
        return 0
    c = configuration()
    points = plan(c)
    for name in ('build.py', 'run.py', 'src/bench.c', 'src/bench.h',
                 'src/rvv_adapter.c', 'src/ime_adapter.c', 'src/counters.h'):
        if not (BENCH / name).is_file():
            raise ValueError('Required existing driver file missing: ' + str(BENCH / name))
    build_api = load_module('fig02_existing_build', BENCH / 'build.py')
    runner = load_module('fig02_existing_run', BENCH / 'run.py')
    source_files = set(BENCH.glob('*.py')) | set((BENCH / 'src').glob('*.[ch]'))
    source_files.add(ROOT / 'HETEROGENEOUS_RVV_IME_OPENMP_GEMM/src/openmp_kernel_dispatch.h')
    source_issues = {}
    for point in points:
        try:
            for backend in ('rvv', 'ime'):
                path, _ = build_api.resolve(ROOT, point[backend + '_kernel'], backend)
                source_files.add(path)
                if backend == 'ime':
                    source_files.add(path.parent / 'rvv_fallback.c')
        except (ValueError, OSError) as exc:
            source_issues[point['point_id']] = str(exc)
    before = {str(p): sha(p) for p in sorted(source_files) if p.is_file()}
    roles = ['primary', 'profile'] + (['counters'] if c['counters'] else [])
    print('Fixed 1024^3 | %d scaling points | %d measured repetitions/role | roles=%s' %
          (len(points), c['repetitions'], ','.join(roles)), flush=True)
    if args.check:
        for point in points:
            print('%-28s cores=%d cpus=%s %s' % (point['active_kernel'], point['cores'],
                  ','.join(map(str, point['cpus'])), source_issues.get(point['point_id'], 'SOURCE_OK')))
        print('Existing files only read. RVV separate output=NA (fused); hardware execution not tested.')
        return 1 if source_issues else 0
    if platform.system() != 'Linux' or not platform.machine().lower().startswith('riscv'):
        raise ValueError('Run measurements on native RISC-V Linux/K1; use --check on this computer.')
    # Keep background thread pools out of this OpenMP experiment; driver pins
    # each worker itself. Inherited affinity restrictions are still respected.
    os.environ.update(OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    allowed = set(os.sched_getaffinity(0))
    result_base = Path(os.environ.get('RESULT_ROOT', str(ROOT / 'datasets/fig02_strong_scaling_complete'))).resolve()
    result = result_base / ('k1_1024_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    result.mkdir(parents=True, exist_ok=False)
    (result / 'measurement_scope.txt').write_text(SCOPE, encoding='utf-8')
    json_write(result / 'configuration.json', dict(c, project_root=str(ROOT), points=points, roles=roles,
               M=N, N=N, K=N, started_utc=stamp(), source_hashes=before))
    info = runner.system_info(ROOT)
    json_write(result / 'system.json', info)
    accepted, raw, failures, attempts = [], [], [], []
    builds = {}
    interrupted = False

    def checkpoint(provenance=True):
        initial = ('point_id', 'role', 'rep', 'accepted', 'backend', 'tile', 'lmul', 'unroll', 'cores')
        csv_write(result / 'measurements.csv', accepted, initial)
        csv_write(result / 'raw_all_runs.csv', raw, initial)
        csv_write(result / 'failures.csv', failures, ('point_id', 'role', 'reason'))
        csv_write(result / 'attempts.csv', attempts, ('point_id', 'role', 'status'))
        summary = summaries(points, accepted, c, provenance)
        csv_write(result / 'summary.csv', summary, ('point_id', 'backend', 'tile', 'lmul', 'unroll', 'cores'))
        return summary

    try:
        # Fixed, logged shuffle reduces always-running-large-points-last bias.
        # All repetitions for a role share one process, matching the existing
        # driver; the order is reproducible from SEED, not selectively retried.
        jobs = [(p, role) for p in points for role in roles]
        random.Random(c['seed']).shuffle(jobs)
        json_write(result / 'execution_order.json', [[p['point_id'], role] for p, role in jobs])
        checkpoint()
        for index, (point, role) in enumerate(jobs, 1):
            folder = result / 'cases' / point['point_id'] / role
            folder.mkdir(parents=True, exist_ok=False)
            print('[%d/%d] %s %s' % (index, len(jobs), point['point_id'], role), flush=True)
            attempt = dict(point_id=point['point_id'], role=role, status='FAILED', log_dir=str(folder), started_utc=stamp())
            try:
                if point['point_id'] in source_issues:
                    raise ValueError(source_issues[point['point_id']])
                if not set(point['cpus']) <= allowed:
                    raise ValueError('Requested CPUs outside permitted affinity: ' + str(point['cpus']))
                pair = (point['rvv_kernel'], point['ime_kernel'])
                if pair not in builds:
                    build_dir = result / 'builds' / ('pair_' + hashlib.sha256('|'.join(pair).encode()).hexdigest()[:12])
                    build_dir.mkdir(parents=True, exist_ok=False)
                    command = [sys.executable, '-B', str(BENCH / 'build.py'), '--project-root', str(ROOT),
                               '--output', str(build_dir), '--rvv-kernel', pair[0], '--ime-kernel', pair[1], '--cc', c['cc']]
                    code = launch(command, build_dir, c['timeout'])
                    meta_path = build_dir / 'build.json'
                    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
                    binary = build_dir / 'bench'
                    valid = code == 0 and meta.get('status') == 'OK' and not meta.get('host_test') and binary.is_file()
                    builds[pair] = (binary, meta, None if valid else 'Build failed; see ' + str(build_dir))
                binary, meta, error = builds[pair]
                if error:
                    raise ValueError(error)
                if sha(binary) != meta['executable_sha256']:
                    raise ValueError('Binary changed after build')
                if any(not Path(p).is_file() or sha(p) != h for p, h in before.items()):
                    raise ValueError('Selected sources/driver changed during campaign')
                case = dict(implementation=point['backend'], timing_mode='end_to_end', threads=point['cores'],
                            cpu_ids=point['cpus'], ime_workers=point['cores'] if point['backend'] == 'ime' else 0,
                            schedule='static', profiled=role == 'profile', counters=role == 'counters')
                command = [str(binary), '--m', str(N), '--n', str(N), '--k', str(N),
                           '--implementation', case['implementation'], '--timing', 'end_to_end',
                           '--threads', str(point['cores']), '--ime-workers', str(case['ime_workers']),
                           '--cpus', ','.join(map(str, point['cpus'])), '--schedule', 'static',
                           '--weight', '4', '--chunk', '1', '--warmups', str(c['warmups']),
                           '--repetitions', str(c['repetitions']), '--seed', str(c['seed']),
                           '--profile', str(int(case['profiled'])), '--counters', str(int(case['counters'])),
                           '--validate-only', '0']
                # Frequency/governor snapshots are read-only explanatory evidence.
                json_write(folder / 'environment_before.json', runner.system_info())
                code = launch(command, folder, c['timeout'])
                stdout = (folder / 'stdout.log').read_text(errors='replace')
                stderr = (folder / 'stderr.log').read_text(errors='replace')
                rows, issues = runner.parse_rows(stdout)
                expected = dict(case, M=N, N=N, K=N, repetitions=c['repetitions'])
                issues += runner.gate_rows(rows, expected)
                issues += runner.gate_validation(stderr, (N, N, N))
                issues += phase_gate(rows, role, point['backend'])
                issues += worker_gate(rows, point, c)
                if code != 0:
                    issues.append('Process return code %s (124=timeout); entire invocation rejected' % code)
                if any(not Path(p).is_file() or sha(p) != h for p, h in before.items()):
                    issues.append('Source changed while measurements were running')
                reason = '; '.join(dict.fromkeys(issues))
                for observed in rows:
                    record = dict(observed, **point, role=role, accepted=not issues, rejection_reason=reason,
                                  campaign_id=result.name, input_a_bytes=N*N, input_b_bytes=N*N,
                                  output_c_bytes=4*N*N, operations=2*N**3, macs=N**3,
                                  alpha=1, lda=N, ldb=N, ldc=N, output_strip_columns=32,
                                  input_a_layout='A[k*M+row]', input_b_layout='B[k*N+col]',
                                  output_c_layout='C[col*M+row]', warmups=c['warmups'], input_seed=c['seed'],
                                  input_rng='xorshift32_shared_A_then_B_then_C', input_min=-7, input_max=7,
                                  initial_c_min=-3, initial_c_max=3,
                                  kernel_unroll_scope='compiler_GCC_pragma_request' if point['backend'] == 'rvv' else 'explicit_IME_K_loop',
                                  vlen_required_bits=256, phase_time_unit='seconds',
                                  packing_scope='A_B_packing' if point['backend'] == 'rvv' else 'compact_B_and_A_B_native_packing',
                                  kernel_scope='RVV_compute_C_update_and_output_stores' if point['backend'] == 'rvv' else 'IME_native_accumulate_including_temporary_tile_stores',
                                  output_scope='FUSED_WITH_KERNEL' if point['backend'] == 'rvv' else 'scatter_output_to_C',
                                  compute_only_sec=None, input_a_pack_sec=None, input_b_pack_sec=None,
                                  allocation_sec=None, initialization_sec=None, synchronization_sec=None,
                                  reference_sec=None, validation_sec=None,
                                  unmeasured_reason='NOT_SEPARATELY_INSTRUMENTED',
                                  source_sha256=sha(build_api.resolve(ROOT, point['active_kernel'], point['backend'])[0]),
                                  binary_sha256=meta['executable_sha256'], compiler_flags=meta['build_flags'],
                                  build_metadata=str(binary.parent / 'build.json'), log_dir=str(folder),
                                  validation_reference='independent_INT64_full_output', system_id=info['system_id'])
                    raw.append(record)
                    if not issues:
                        accepted.append(record)
                attempt['return_code'] = code
                attempt['accepted_samples'] = len(rows) if not issues else 0
                json_write(folder / 'checks.json', dict(issues=issues, accepted_samples=attempt['accepted_samples']))
                if issues:
                    raise ValueError(reason)
                attempt['status'] = 'OK' if role != 'counters' or all(r.get('counters_status') == 'OK' for r in rows) else 'COUNTERS_UNAVAILABLE'
            except (ValueError, OSError, KeyError, TypeError) as exc:
                attempt['reason'] = str(exc)
                failures.append(dict(point_id=point['point_id'], role=role, reason=str(exc), log_dir=str(folder)))
                print('  FAILED; continuing: ' + str(exc), flush=True)
            attempts.append(attempt)
            checkpoint()
    except KeyboardInterrupt:
        interrupted = True
        failures.append(dict(point_id='CAMPAIGN', role='interrupt', reason='User interrupted; completed data retained'))
    after = {p: sha(p) if Path(p).is_file() else None for p in before}
    unchanged = before == after
    json_write(result / 'source_preservation.json', dict(unchanged=unchanged, before=before, after=after))
    summary = checkpoint(unchanged)
    timings_complete = all(x['primary_complete'] and x['profile_complete'] for x in summary)
    counters_complete = not c['counters'] or all(x['ipc_complete'] for x in summary)
    code = 130 if interrupted else 1 if failures or not unchanged or not timings_complete else 3 if not counters_complete else 0
    json_write(result / 'completion.json', dict(finished_utc=stamp(), exit_code=code,
               planned_points=len(points), completed_attempts=len(attempts), failed_attempts=len(failures),
               accepted_samples=len(accepted), timings_complete=timings_complete,
               counters_complete=counters_complete, source_unchanged=unchanged,
               hardware_execution='native_riscv', separate_RVV_output_available=False))
    print('Results: ' + str(result), flush=True)
    print('Exit %d | timings complete=%s | counters complete=%s | source unchanged=%s' %
          (code, timings_complete, counters_complete, unchanged), flush=True)
    return code


try:
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    sys.exit(main())
except (ValueError, OSError) as error:
    print('FIGURE 2 ERROR: ' + str(error), file=sys.stderr)
    sys.exit(2)
PY

