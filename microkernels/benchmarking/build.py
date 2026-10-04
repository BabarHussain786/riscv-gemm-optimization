#!/usr/bin/env python3
"""Out-of-tree builds using untouched canonical RVV/IME sources."""
import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATTERN = re.compile(r'(i?gemm|ime)_kernel_(8x[48])_zvl256b_lmul(mf8|mf4|mf2|1|2)_unroll(1|2|4|8)$')

def resolve(root, name, backend):
    match = PATTERN.fullmatch(name)
    if not match or (backend == 'rvv' and match[1] != 'igemm') or (backend == 'ime' and match[1] != 'ime'):
        raise ValueError(f'Invalid {backend} kernel: {name}')
    tile, lmul, unroll = match.group(2, 3, 4)
    if backend == 'rvv':
        directory = root / f'GEMM_RVV_FP32_INT8_{tile}_Baseline' / f'RVV_IGEMM_INT8_I8I32_{tile}' / name
        source = directory / (name + '_i8i32.c')
    else:
        if lmul != '1':
            raise ValueError('IME experimental LMUL variants are not accepted by the primary K1 layer')
        directory = root / 'IME_NATIVE_KERNELS' / f'IME_GEMM_INT8_I8I32_{tile}_NATIVE' / name
        source = directory / (name + '.c')
    if not source.is_file():
        raise ValueError(f'Missing selected kernel: {source}')
    return source, dict(kernel=name, tile=tile, lmul=lmul, unroll=int(unroll))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build(args):
    root, out = args.project_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'build.json').exists() or (out / 'bench').exists() or (out / 'bench.exe').exists():
        raise ValueError('Build output already exists; choose a new directory')
    rvv, rm = resolve(root, args.rvv_kernel, 'rvv')
    ime, im = resolve(root, args.ime_kernel, 'ime')
    compiler = shlex.split(args.cc, posix=True)
    if not compiler:
        raise ValueError('Empty compiler command')
    dispatch = root / 'HETEROGENEOUS_RVV_IME_OPENMP_GEMM/src/openmp_kernel_dispatch.h'
    # The IME translation unit calls the matching fallback symbol when native
    # IME execution is unavailable, so the sibling fallback object is part of
    # the mixed benchmark and must remain linked.
    inputs = list((HERE / 'src').glob('*.[ch]')) + [Path(__file__), rvv, ime, dispatch, ime.parent/'rvv_fallback.c']
    hashes = {str(p): sha(p) for p in inputs}
    # A common source-universe fingerprint, independent of selected tuning.
    universe = {str(p.relative_to(root)).replace('\\','/'): sha(p)
                for p in root.rglob('*') if p.is_file() and p.suffix in ('.c','.h')
                and 'benchmarking' not in p.relative_to(root).parts}
    universe.update({'benchmarking/'+str(p.relative_to(HERE)).replace('\\','/'):sha(p)
                     for p in list((HERE/'src').glob('*.[ch]'))+list(HERE.glob('*.py'))})
    digest = hashlib.sha256(json.dumps(universe, sort_keys=True).encode()).hexdigest()
    flags = ['-O3', '-std=c11', '-Wall', '-Wextra', '-Wno-unused-function', '-Wno-unknown-pragmas']
    if args.host_test:
        flags += ['-DBENCH_HOST_TEST=1']
    else:
        flags += ['-march=rv64gcv_zvl256b', '-mabi=lp64d', '-fopenmp', '-fno-pie']
    config = out/'selected_sources.h'
    config.write_text('#define BENCH_RVV_NR '+rm['tile'][-1]+'\n'+
                      '#define BENCH_DISPATCH_SOURCE '+json.dumps(dispatch.as_posix())+'\n'+
                      '#define BENCH_IME_SOURCE '+json.dumps(ime.as_posix())+'\n', encoding='utf-8')
    flags += ['-I', str(HERE/'src'), '-include', str(config)]
    sources = [(HERE/'src/bench.c', [])]
    if args.host_test:
        sources += [(HERE/'src/reference_adapter.c', [])]
    else:
        # Most canonical RVV kernels honor CNAME, while the 8x8 sources also
        # contain a literal function name.  Rename both forms in this
        # out-of-tree translation unit so the adapter exports one stable
        # callback.  No source file in the kernel tree is edited.  The
        # selected RVV source is renamed, while the IME fallback retains its
        # canonical symbol because the IME adapter calls it when required.
        rvv_symbol = rm['kernel']
        sources += [(HERE/'src/rvv_adapter.c', []), (HERE/'src/ime_adapter.c', []),
                    (rvv, ['-DCNAME=bench_rvv_entry',
                           f'-D{rvv_symbol}=bench_rvv_entry',
                           f'-D{rvv_symbol}_i8i32=bench_rvv_entry']),
                    (ime.parent/'rvv_fallback.c', [])]
    objects, commands = [], []
    for index, (source, extra) in enumerate(sources):
        obj = out/f'unit_{index}.o'
        commands.append(compiler+flags+extra+['-c', str(source), '-o', str(obj)])
        objects.append(str(obj))
    binary = out/('bench.exe' if args.host_test and os.name == 'nt' else 'bench')
    commands.append(compiler+objects+(['-fopenmp', '-no-pie'] if not args.host_test else [])+['-o', str(binary)])
    metadata = {'schema_version': 1, 'host_test': args.host_test, 'publishable': not args.host_test,
                'project_root': str(root), 'binary': str(binary), 'compiler_command': compiler,
                'build_flags': flags, 'commands': commands, 'sources': hashes,
                'source_digest': digest, 'source_universe': universe, 'hardware_target': 'HOST_REFERENCE_ONLY' if args.host_test else 'K1_RVV256_IME_A60',
                'rvv_kernel': rm['kernel'], 'ime_kernel': im['kernel'],
                **{'rvv_'+k:v for k,v in rm.items() if k!='kernel'},
                **{'ime_'+k:v for k,v in im.items() if k!='kernel'}}
    try:
        metadata['compiler_version'] = subprocess.run(compiler+['--version'], capture_output=True, text=True, check=True).stdout
        with (out/'build.log').open('w', encoding='utf-8') as log:
            for cmd in commands:
                log.write(shlex.join(cmd)+'\n');log.flush()
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        metadata['executable_sha256'] = sha(binary)
        metadata['status'] = 'OK'
    except (OSError, subprocess.CalledProcessError) as exc:
        metadata['status'] = 'FAILED';metadata['error'] = str(exc)
        (out/'build.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        raise
    (out/'build.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(binary)
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=HERE.parent)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rvv-kernel',default='igemm_kernel_8x4_zvl256b_lmulmf8_unroll2')
    p.add_argument('--ime-kernel',default='ime_kernel_8x4_zvl256b_lmul1_unroll1')
    p.add_argument('--host-test',action='store_true')
    p.add_argument('--cc',default=os.environ.get('CC','gcc'))
    args=p.parse_args()
    try:return build(args)
    except (ValueError,OSError,subprocess.CalledProcessError) as exc:
        p.exit(1,f'BUILD FAILED: {exc}\n')

if __name__=='__main__':raise SystemExit(main())
