#!/usr/bin/env python3
"""Plan and run isolated K1 paper measurements. Standard library only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
PLOTS = {
    1: 'fig01_strong_scaling', 2: 'fig02_weak_scaling',
    3: 'fig03_static_vs_dynamic', 4: 'fig04_rvv_int8_tuning',
    5: 'fig05_rvv_vs_ime_int8', 6: 'fig06_rvv_fp32_fp64',
    7: 'fig07_rvv_multicore_vs_heterogeneous', 8: 'fig08_correctness',
}
LMULS = ('mf8', 'mf4', 'mf2', '1', '2', '4', '8')
INPUTS = ('bounded_uniform', 'full_range_uniform', 'mixed_magnitude', 'cancellation_stress')


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root, experimental=False):
    """Canonical paths only; never interpret a missing variant as a measurement."""
    found, coverage = [], []
    for kind, prefix in [('INT8_RVV', 'igemm'), ('FP32_RVV', 'sgemm'),
                         ('FP64_RVV', 'dgemm'), ('INT8_IME', 'ime')]:
        for tile in ('8x4', '8x8'):
            if kind == 'INT8_IME':
                family = root / 'IME_NATIVE_KERNELS' / f'IME_GEMM_INT8_I8I32_{tile}_NATIVE'
            else:
                fp = 'FP64' if kind == 'FP64_RVV' else 'FP32'
                sub = {'INT8_RVV': f'RVV_IGEMM_INT8_I8I32_{tile}',
                       'FP32_RVV': f'RVV_SGEMM_FP32_{tile}',
                       'FP64_RVV': f'RVV_DGEMM_FP64_{tile}'}[kind]
                family = root / f'GEMM_RVV_{fp}_INT8_{tile}_Baseline' / sub
            for lmul in LMULS:
                for unroll in (1, 2, 4, 8):
                    name = f'{prefix}_kernel_{tile}_zvl256b_lmul{lmul}_unroll{unroll}'
                    directory = family / name
                    symbol = name + ('_i8i32' if kind == 'INT8_RVV' else '')
                    source = directory / (symbol + '.c')
                    reason = ''
                    if not source.is_file() or not (directory / 'Makefile').is_file():
                        reason = 'missing_source_or_makefile'
                    elif kind == 'INT8_RVV' and lmul in ('4', '8'):
                        reason = 'excluded_by_existing_widening_runner'
                    elif kind == 'FP32_RVV' and tile == '8x8' and lmul == 'mf2':
                        reason = 'excluded_by_existing_lane_capacity_check'
                    elif kind == 'FP64_RVV' and tile == '8x4' and lmul in ('mf8', 'mf4', 'mf2', '1'):
                        reason = 'excluded_by_existing_lane_capacity_check'
                    elif kind == 'INT8_IME' and lmul != '1' and not experimental:
                        reason = 'experimental_opt_in_required'
                    item = dict(kind=kind, tile=tile, lmul=lmul, unroll=unroll,
                                kernel=name, symbol=symbol, source=str(source.relative_to(root)),
                                experimental=(kind == 'INT8_IME' and lmul != '1'))
                    coverage.append(dict(item, status=reason or 'available_not_yet_validated'))
                    if not reason:
                        found.append(item)
    return found, coverage


def plan_cases(kernels, args):
    cases = []
    def add(fig, k, runner, size, threads=1, policy='static', weight='4:1', chunk=1,
            input_class='', mode=None):
        if fig not in args.figures:
            return
        mode = mode or ('k1-ime' if k['kind'] == 'INT8_IME' else 'k1-rvv')
        case = dict(k, figure=PLOTS[fig], runner=runner, dimensions=list(size),
                    threads=threads, mode=mode, policy=policy, weight=weight, chunk=chunk,
                    tile_n=args.tile_n, input_class=input_class,
                    runs=args.accuracy_runs if fig == 8 else args.runs,
                    timing_scope='standalone_wrapper' if runner == 'standalone' else
                                 ('correctness_only' if fig == 8 else 'openmp_end_to_end'),
                    rvv_cores=args.rvv_cores, ime_cores=args.ime_cores)
        case['id'] = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:16]
        cases.append(case)
    rvv = [k for k in kernels if k['kind'] == 'INT8_RVV']
    ime = [k for k in kernels if k['kind'] == 'INT8_IME']
    pairs = [k for k in ime if any(r['tile'] == k['tile'] and r['lmul'] == k['lmul']
                                 and r['unroll'] == k['unroll'] for r in rvv)]
    for k in rvv + ime:
        counts = (1, 2, 4) if k['kind'] == 'INT8_IME' else (1, 2, 4, 8)
        for size in args.sizes:
            for cores in counts:
                add(1, k, 'openmp', (size,)*3, cores)
            add(5, k, 'standalone', (size,)*3)
            if k['kind'] == 'INT8_RVV':
                add(4, k, 'standalone', (size,)*3)
        for cores, dimension in zip((1, 2, 4, 8), args.weak_dimensions):
            if cores in counts:
                add(2, k, 'openmp', (dimension,)*3, cores)
        for shape in args.accuracy_shapes:
            # This additionally covers the canonical RVV inventory, not only IME fallbacks.
            add(8, k, 'openmp', shape)
            if k['kind'] == 'INT8_IME':
                for input_class in INPUTS:
                    add(8, k, 'accuracy', shape, input_class=input_class)
    for k in pairs:
        for size in args.sizes:
            for weight in args.weights:
                add(3, k, 'openmp', (size,)*3, 8, weight=weight, mode='k1-mixed-rvv-ime')
            for chunk in args.chunks:
                add(3, k, 'openmp', (size,)*3, 8, policy='dynamic', chunk=chunk,
                    mode='k1-mixed-rvv-ime')
            for policy in ('static', 'dynamic'):
                add(7, k, 'openmp', (size,)*3, 8, policy=policy, mode='k1-mixed-rvv-ime')
    for k in rvv:
        for size in args.sizes:
            add(7, k, 'openmp', (size,)*3, 8)
    for k in kernels:
        if k['kind'] in ('FP32_RVV', 'FP64_RVV'):
            for size in args.sizes:
                add(6, k, 'standalone', (size,)*3)
    return cases


def command_for(case, root, directory, args):
    # Whitelist inherited environment so an old KERNEL_FILTER, OUT_DIR, compiler
    # flag, or affinity variable cannot silently change this recorded campaign.
    keep = ('PATH', 'HOME', 'USER', 'LOGNAME', 'SHELL', 'TMPDIR', 'LD_LIBRARY_PATH')
    env = {k: os.environ[k] for k in keep if k in os.environ}
    m, n, k = case['dimensions']
    env.update(PAPER_PROJECT_ROOT=str(root), PAPER_CASE_DIR=str(directory),
               PAPER_KERNEL_FILTER=case['kernel'], M=str(m), N=str(n), K=str(k),
               RUNS=str(case['runs']), OUT_DIR=str(directory), CC=args.cc,
               RVV_CORES=case['rvv_cores'], IME_CORES=case['ime_cores'],
               OMP_NUM_THREADS=str(case['threads']), OMP_DYNAMIC='false',
               OMP_PROC_BIND='close', OMP_PLACES='cores',
               KERNEL_FILTER=case['symbol'], KIND_FILTER=case['kind'],
               GEMM_TILE_SCHEDULE=case['policy'], GEMM_DYNAMIC_CHUNK=str(case['chunk']),
               MIXED_IME_TILE_WEIGHT=case['weight'].split(':')[0],
               MIXED_RVV_TILE_WEIGHT=case['weight'].split(':')[1],
               ENABLE_MF2=str(int(case['experimental'])),
               ENABLE_EXPERIMENTAL_MF2=str(int(case['experimental'])),
               GEMM_VALIDATE='1', VALIDATE_EACH_RUN='1', GEMM_WARMUP=str(args.warmups),
               PERF_STAT=str(int(not args.no_perf)), PERF_EVENTS='cycles,instructions',
               INPUT_CLASS=case['input_class'] or 'full_range_uniform', BASE_SEED=str(args.seed),
               VALIDATE_M='15', VALIDATE_N='15', VALIDATE_K='69', LC_ALL='C')
    if case['mode'] == 'k1-mixed-rvv-ime':
        env['KIND_FILTER'] = 'INT8_MIXED'
    cmd = ['bash', str(HERE / ('runner_' + case['runner'] + '.sh'))]
    if case['runner'] == 'openmp':
        cmd += [case['mode'], str(m), str(n), str(k), str(case['tile_n']), str(case['runs'])]
    elif case['runner'] == 'accuracy':
        cmd += ['k1', str(case['runs'])]
    return cmd, env


def read_rows(directory, runner):
    pattern = {'openmp': 'openmp_raw_latest_*.csv',
               'standalone': 'k1_01_rvv_ime_raw_latest.csv',
               'accuracy': 'int8_ime_vs_rvv_accuracy_once_summary_k1_latest.csv'}[runner]
    rows = []
    for path in directory.glob(pattern):
        with path.open(newline='', encoding='utf-8') as stream:
            rows.extend(dict(row, source_csv=str(path)) for row in csv.DictReader(stream))
    return rows


def classify(rc, rows, expected=None):
    if rc != 0:
        return 'failed'
    if not rows:
        return 'missing_data'
    if not all(row.get('status') == 'OK' for row in rows):
        return 'failed_validation_or_run'
    if expected is not None and len(rows) != expected:
        return 'incomplete_sample_count'
    return 'complete'


def expected_rows(case):
    if case['runner'] == 'openmp':
        cores = 1
    elif case['runner'] == 'accuracy':
        cores = len(case['ime_cores'].split()) + len(case['rvv_cores'].split())
    else:
        cores = len(case['ime_cores' if case['kind']=='INT8_IME' else 'rvv_cores'].split())
    return case['runs'] * cores


def aggregate(output, cases):
    for figure in sorted({c['figure'] for c in cases}):
        all_rows, statuses, summaries = [], [], []
        for case in (c for c in cases if c['figure'] == figure):
            base = output / figure / 'cases' / case['id']
            status = json.loads((base / 'status.json').read_text()) if (base / 'status.json').exists() else {'status': 'not_run'}
            statuses.append(dict(case_id=case['id'], **status))
            # Only consume the attempt explicitly recorded in status.json, never stale aliases.
            if status.get('attempt'):
                rows = read_rows(base / status['attempt'], case['runner'])
                for row in rows:
                    all_rows.append(dict(row, case_id=case['id'], case_status=status['status'],
                                         plan_kernel=case['kernel'], plan_lmul=case['lmul'],
                                         plan_unroll=case['unroll'], plan_tile=case['tile'],
                                         plan_timing_scope=case['timing_scope']))
                groups = {}
                for row in rows:
                    if row.get('status') != 'OK':
                        continue
                    try:
                        duration = float(row['time_sec'])
                        if not (0 < duration < float('inf')):
                            continue
                    except (KeyError, ValueError, TypeError):
                        continue
                    key = row.get('core', row.get('core_group', 'NA'))
                    groups.setdefault(key, []).append(duration)
                for core, times in groups.items():
                    summaries.append(dict(case_id=case['id'], kernel=case['kernel'],
                        tile=case['tile'], lmul=case['lmul'], unroll=case['unroll'],
                        mode=case['mode'], threads=case['threads'], policy=case['policy'],
                        dimensions='x'.join(map(str, case['dimensions'])), core=core,
                        timing_scope=case['timing_scope'], case_status=status['status'],
                        successful_samples=len(times), mean_time_sec=statistics.mean(times),
                        median_time_sec=statistics.median(times),
                        sample_sd_time_sec=statistics.stdev(times) if len(times)>1 else '',
                        min_time_sec=min(times), max_time_sec=max(times)))
        folder = output / figure
        save_json(folder / 'case_statuses.json', statuses)
        fields = sorted({key for row in all_rows for key in row}) or ['case_id', 'status']
        with (folder / 'raw_data.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        if summaries:
            with (folder / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=HERE.parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--figures', default='1,2,3,4,5,6,7,8')
    parser.add_argument('--sizes', default='1024', help='comma-separated cubic sizes')
    parser.add_argument('--weak-dimensions', default='512,672,832,1024', help='dimensions for 1,2,4,8 cores; paper uses approximate weak scaling')
    parser.add_argument('--accuracy-shapes', default='15x15x69,1024x1024x1024')
    parser.add_argument('--runs', type=positive, default=6)
    parser.add_argument('--accuracy-runs', type=positive, default=3)
    parser.add_argument('--warmups', type=positive, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weights', default='4:1', help='comma-separated static IME:RVV weights')
    parser.add_argument('--chunks', default='1', help='comma-separated dynamic chunk sizes')
    parser.add_argument('--rvv-cores', default='4', help='space-separated standalone RVV cores')
    parser.add_argument('--ime-cores', default='0', help='space-separated standalone IME cores')
    parser.add_argument('--experimental-ime', action='store_true')
    parser.add_argument('--no-perf', action='store_true')
    parser.add_argument('--cc', default='gcc')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='write complete plan; do not build or measure')
    args = parser.parse_args(argv)
    try:
        args.figures = sorted(set(map(int, args.figures.split(','))))
        if not args.figures or any(f not in PLOTS for f in args.figures):
            raise ValueError('figures must be in 1..8')
        args.sizes = [positive(s) for s in args.sizes.split(',')]
        args.weak_dimensions = [positive(s) for s in args.weak_dimensions.split(',')]
        if len(args.weak_dimensions) != 4:
            raise ValueError('four weak dimensions required')
        args.accuracy_shapes = [tuple(positive(n) for n in s.split('x')) for s in args.accuracy_shapes.split(',')]
        if any(len(s) != 3 for s in args.accuracy_shapes):
            raise ValueError('accuracy shapes must be MxNxK')
        args.weights = args.weights.split(',')
        if any(not re.fullmatch(r'[1-9][0-9]*:[1-9][0-9]*', w) for w in args.weights):
            raise ValueError('weights must be positive IME:RVV pairs')
        args.chunks = [positive(s) for s in args.chunks.split(',')]
        for cores, allowed in ((args.rvv_cores, range(8)), (args.ime_cores, range(4))):
            if not cores.split() or any(int(c) not in allowed for c in cores.split()):
                raise ValueError('invalid K1 core list')
        if any(s % 8 for s in args.sizes + args.weak_dimensions):
            raise ValueError('performance dimensions must be multiples of eight; use accuracy shapes for tails')
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    args.tile_n = 32
    return args


def main(argv=None):
    args = arguments(argv)
    root = args.project_root.resolve()
    if not (root / 'IME_NATIVE_KERNELS').is_dir():
        raise SystemExit('Invalid project root: missing IME_NATIVE_KERNELS')
    kernels, coverage = inventory(root, args.experimental_ime)
    cases = plan_cases(kernels, args)
    if not cases:
        raise SystemExit('No cases found; inspect source inventory')
    output = (args.output or root / 'paper_results' / datetime.now().strftime('campaign_%Y%m%d_%H%M%S_%f')).resolve()
    # Include all source, headers, Makefiles and suite files so resume cannot mix revisions.
    source_hashes = {str(p.relative_to(root)): sha(p) for p in root.rglob('*')
                     if p.is_file() and (p.suffix in ('.c', '.h') or p.name == 'Makefile')}
    source_hashes.update({'paper_scripts/' + p.name: sha(p) for p in HERE.iterdir()
                          if p.suffix in ('.py', '.sh')})
    config = {k: v for k, v in vars(args).items() if k not in ('output', 'resume', 'dry_run', 'project_root')}
    manifest = json.loads(json.dumps(dict(version=1, configuration=config, sources=source_hashes, cases=cases)))
    previous = output / 'manifest.json'
    if output.exists() and any(output.iterdir()):
        if not args.resume or not previous.exists():
            raise SystemExit('Output is nonempty; choose a new folder or use --resume with its unchanged configuration')
        if json.loads(previous.read_text()) != manifest:
            raise SystemExit('Resume refused: configuration or source hashes changed')
    output.mkdir(parents=True, exist_ok=True)
    save_json(previous, manifest)
    save_json(output / 'coverage.json', coverage)
    for fig in args.figures:
        save_json(output / PLOTS[fig] / 'plan.json', [c for c in cases if c['figure'] == PLOTS[fig]])
    for fig in args.figures:
        print(f'{PLOTS[fig]}: {sum(c["figure"] == PLOTS[fig] for c in cases)} cases', flush=True)
    print(f'Total: {len(cases)} cases; data: {output}', flush=True)
    if args.dry_run:
        print('PLAN ONLY: no kernels built or executed.')
        return 0
    if platform.system() != 'Linux' or platform.machine().lower() != 'riscv64':
        raise SystemExit('Execution requires RISC-V Linux on K1; use --dry-run on this machine')
    for tool in ('bash', 'make', 'taskset', args.cc):
        if not shutil.which(tool):
            raise SystemExit(f'Missing required tool: {tool}')
    # Standalone make clean/build uses source directories: serialize even across output folders.
    lock = root / '.paper_campaign.lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise SystemExit('Campaign lock exists. Do not run concurrent campaigns; see README for stale-lock recovery.')
    failures = 0
    try:
        save_json(lock / 'owner.json', dict(pid=os.getpid(), host=platform.node(), output=str(output)))
        provenance = dict(utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                          cpuinfo=Path('/proc/cpuinfo').read_text(), environment={k: os.environ.get(k) for k in ('PATH','LD_LIBRARY_PATH')})
        provenance['frequency_policy'] = {
            str(p): p.read_text().strip()
            for p in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_governor')
            if os.access(p, os.R_OK)}
        if Path('/etc/os-release').exists():
            provenance['os_release'] = Path('/etc/os-release').read_text()
        for name, cmd in [('compiler', [args.cc, '--version']), ('affinity', ['taskset', '-pc', str(os.getpid())])]:
            provenance[name] = subprocess.run(cmd, capture_output=True, text=True).stdout
        save_json(output / ('host_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'), provenance)
        for index, case in enumerate(cases, 1):
            base = output / case['figure'] / 'cases' / case['id']
            status_path = base / 'status.json'
            if status_path.exists():
                prior = json.loads(status_path.read_text())
                if prior.get('status') == 'complete' and prior.get('attempt') and read_rows(base / prior['attempt'], case['runner']):
                    continue
            attempt = 'attempt_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            directory = base / attempt
            directory.mkdir(parents=True)
            cmd, env = command_for(case, root, directory, args)
            save_json(directory / 'command.json', dict(command=cmd, environment=env, configuration=case))
            save_json(status_path, dict(status='running', attempt=attempt))
            print(f'[{index}/{len(cases)}] {case["figure"]} {case["kernel"]} {case["dimensions"]} {case["policy"]}', flush=True)
            started = time.monotonic()
            try:
                with (directory / 'console.log').open('w') as stream:
                    process = subprocess.Popen(cmd, env=env, cwd=root, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        rc = process.wait()
                    except KeyboardInterrupt:
                        import signal
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise
                state = classify(rc, read_rows(directory, case['runner']), expected_rows(case))
            except KeyboardInterrupt:
                save_json(status_path, dict(status='interrupted', attempt=attempt))
                raise
            except OSError as error:
                rc, state = -1, 'execution_error'
                save_json(directory / 'error.json', str(error))
            save_json(status_path, dict(status=state, attempt=attempt, return_code=rc,
                                        elapsed_seconds=time.monotonic()-started))
            failures += state != 'complete'
            print(f'  {state}', flush=True)
    finally:
        try:
            aggregate(output, cases)
        finally:
            (lock / 'owner.json').unlink(missing_ok=True)
            lock.rmdir()
    print(f'Finished: {failures} unsuccessful cases. Inspect case_statuses.json and raw_data.csv.')
    return int(failures > 0)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted; completed cases preserved. Resume with the same arguments and --resume.', file=sys.stderr)
        sys.exit(130)
