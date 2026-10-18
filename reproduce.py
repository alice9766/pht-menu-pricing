#!/usr/bin/env python3
"""Source-only entry point. Data, results and plots are generated locally."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
BASE = ROOT / '.work/experiments'
GENERATED = ROOT / 'generated'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def command(args, cwd=None):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd or ROOT, check=True)


def prepare():
    # Preserve original module-relative paths and source checksums while keeping
    # all subsequently generated data outside the published source directories.
    for folder in ('src', 'tests', 'protocol'):
        shutil.copytree(ROOT/'experiments'/folder, BASE/folder, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source = BASE/'data/source'
    source.mkdir(parents=True, exist_ok=True)
    for name in ('audit_windows.py', 'legacy_cost_reference.json'):
        shutil.copyfile(ROOT/'experiments/source'/name, source/name)
    return BASE


def source_parameters():
    return json.loads((ROOT/'experiments/protocol/data_protocol.json').read_text())['source']


def raw_data(archive=None):
    prepare()
    params = source_parameters()
    target = BASE/'data/source/electricityloaddiagrams20112014.zip'
    if archive:
        archive = Path(archive).resolve()
        if sha(archive) != params['archive_sha256']:
            raise ValueError('The supplied UCI archive does not match the protocol checksum.')
        if archive != target:
            shutil.copyfile(archive, target)
    elif not target.exists():
        partial = target.with_suffix('.part')
        print('Downloading the official UCI archive (approximately 249 MiB).', flush=True)
        with urllib.request.urlopen(params['url'], timeout=90) as inp, partial.open('wb') as out:
            shutil.copyfileobj(inp, out)
        if sha(partial) != params['archive_sha256']:
            raise ValueError('Downloaded archive checksum mismatch; incomplete copy retained as .part.')
        partial.replace(target)
    if sha(target) != params['archive_sha256']:
        raise ValueError('UCI archive checksum mismatch.')
    write_json(target.parent/'source_manifest.json',
               {'source_url': params['url'], 'size_bytes': target.stat().st_size,
                'sha256': params['archive_sha256']})
    command([sys.executable, target.parent/'audit_windows.py'])
    command([sys.executable, BASE/'src/build_instances.py', '--verify-raw', target])
    for key in ('audit_csv', 'finiteness_audit'):
        if sha(BASE/params[key+'_path']) != params[key+'_sha256']:
            raise ValueError('Source measurements differ from the frozen protocol: ' + key)
    print('Source measurements built and checked.')


def inputs():
    prepare()
    params = source_parameters()
    for key in ('audit_csv', 'finiteness_audit'):
        path = BASE/params[key+'_path']
        if not path.exists():
            raise FileNotFoundError('Run python reproduce.py raw-data before inputs.')
    command([sys.executable, BASE/'src/build_instances.py', '--all', '--output-dir', BASE/'data/all'])
    command([sys.executable, BASE/'src/build_instances.py', '--output-dir', BASE/'data/pilot'])
    command([sys.executable, BASE/'src/build_final.py'])


def tests(with_data=False):
    prepare()
    if with_data:
        # These fixtures are generated from the same protocol lineage. This
        # command only constructs inputs and does not run additional benchmarks.
        if not (BASE/'data/final/jobs.jsonl').exists():
            inputs()
        command([sys.executable, BASE/'src/build_formal.py'])
        command([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=BASE)
    else:
        names = [p.stem for p in sorted((BASE/'tests').glob('test_*.py'))
                 if p.name not in ('test_builder.py', 'test_formal_builder.py')]
        # unittest accepts module names through the tests directory on sys.path.
        command([sys.executable, '-m', 'unittest', '-v', *names], cwd=BASE/'tests')


def run(mode, workers):
    prepare()
    if not (BASE/'data/final/jobs.jsonl').exists():
        raise FileNotFoundError('Run python reproduce.py inputs before run.')
    command([sys.executable, BASE/'src/run_formal.py', '--package-root', BASE,
             '--protocol', BASE/'protocol/final_protocol.json', '--jobs', BASE/'data/final/jobs.jsonl',
             '--mode', mode, '--workers', workers, '--output-dir', BASE/f'results/final_{mode}'])


def summarize():
    prepare()
    output = GENERATED/'summary'
    if (output/'summary.json').exists():
        raise FileExistsError('Move or remove generated/summary before regenerating it.')
    command([sys.executable, BASE/'src/summarize_final.py', '--results',
             BASE/'results/final_bulk/results.jsonl', BASE/'results/final_isolated/results.jsonl',
             '--inputs', BASE/'data/all/instances.jsonl', BASE/'data/final/comparison_instances.jsonl',
             BASE/'data/final/mapping_instances.jsonl', BASE/'data/final/scaling_instances.jsonl',
             '--jobs', BASE/'data/final/jobs.jsonl', '--output-dir', output])


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def figures():
    prepare()
    summary = GENERATED/'summary'
    if not (summary/'summary.json').exists():
        raise FileNotFoundError('Run python reproduce.py summarize before figures.')
    analysis = GENERATED/'analysis_figures'
    command([sys.executable, BASE/'src/plot_final.py', '--package', BASE,
             '--summary-dir', summary, '--output-dir', analysis])
    command([sys.executable, ROOT/'scripts/extract_menu_example.py', '--base', BASE,
             '--output', GENERATED/'menu_example'])
    plot = load_module('publication_plots', ROOT/'scripts/plot_publication.py')
    data = json.loads((analysis/'figure_values.json').read_text())
    plot.style()
    output = GENERATED/'figures'
    plot.efficiency(data, output)
    plot.scaling(data, output)
    plot.capacity(data, output)
    plot.menu(json.loads((GENERATED/'menu_example/menu_example.json').read_text()), output)
    command([sys.executable, ROOT/'scripts/export_paper_tables.py', '--base', BASE,
             '--summary', summary, '--output', GENERATED/'tables'])
    print('Experiment figures and tables generated from local run records.')


def proofs():
    output = GENERATED/'proofs'
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT/'proofs', output, dirs_exist_ok=True)
    command(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
             'complete_proofs.tex'], cwd=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    raw = sub.add_parser('raw-data', help='Download UCI data and construct quality measurements')
    raw.add_argument('--archive', type=Path, help='Use an already downloaded official UCI ZIP')
    sub.add_parser('inputs', help='Construct the prescribed final experiment inputs and jobs')
    test = sub.add_parser('tests', help='Run unit checks without source data')
    test.add_argument('--with-data', action='store_true', help='Also run input-construction checks')
    bench = sub.add_parser('run', help='Execute the fixed experiment protocol')
    bench.add_argument('--mode', choices=('bulk', 'isolated'), required=True)
    bench.add_argument('--workers', type=int, choices=range(1,5), default=4)
    sub.add_parser('summarize', help='Aggregate all completed runs and retained terminal statuses')
    sub.add_parser('figures', help='Generate experimental figures and numerical tables')
    sub.add_parser('proofs', help='Compile complete proofs with latexmk')
    args = parser.parse_args()
    if args.action == 'raw-data': raw_data(args.archive)
    elif args.action == 'tests': tests(args.with_data)
    elif args.action == 'run': run(args.mode, args.workers)
    else: globals()[args.action]()


if __name__ == '__main__':
    main()
