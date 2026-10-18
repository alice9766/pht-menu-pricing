#!/usr/bin/env python3
"""Isolated end-to-end pilot; records all jobs and checks exact accounting.

This run checks the pipeline. Its timings are not a formal performance study.
Each method receives a fresh Python process and a private oracle cache.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('joint', 'G0', 'G1', 'G2', 'capacity', 'fixed', 'contiguous')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def worker():
    from pht_experiment import solve_instance
    request = json.load(sys.stdin)
    instance, method = request['instance'], request['method']
    affinity = None
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        if allowed:
            os.sched_setaffinity(0, {allowed[0]})
            affinity = [allowed[0]]
    options = {}
    if method == 'fixed':
        options['fixed_products'] = [p['q'] for p in instance['products']]
    if method == 'capacity':
        options['capacity'] = len(instance['products'])
    start = time.perf_counter()
    result = solve_instance(instance, method=method, **options)
    seconds = time.perf_counter() - start
    record = {
        'status': 'ok', 'result': result,
        'measurement': {
            'solve_validate_replay_seconds': seconds,
            'worker_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'cpu_affinity': affinity,
            'time_scope': 'solve_instance validation, optimization, recovery and independent replay; excludes import, JSON parsing, process startup and serialization',
            'memory_scope': 'Linux whole-worker peak RSS, including Python interpreter and imports',
        },
    }
    print(json.dumps(record, ensure_ascii=False, allow_nan=False))


def exact_cross_checks(instances, records):
    from pht_experiment import replay_menu
    by_id = {p['instance_id']: p for p in instances}
    jobs = {}
    checks, violations = Counter(), []
    for record in records:
        if record['status'] != 'ok':
            violations.append({'instance_id': record['instance_id'],
                               'method': record['method'], 'reason': record['status']})
            continue
        r = record['result']
        payload = by_id[r['instance_id']]
        checks['returned_menus'] += 1
        replay = replay_menu(payload, r['menu']['products'], r['menu']['prices'])
        if replay['accounting'] != r['accounting'] or replay['buyer_choices'] != r['buyer_choices']:
            violations.append({'instance_id': r['instance_id'], 'method': r['method'],
                               'reason': 'independent_parent_replay_mismatch'})
        else:
            checks['parent_replay_passed'] += 1
        if r['method'] == 'fixed' and r['menu']['products'] != list(range(1, len(payload['products']) + 1)):
            violations.append({'instance_id': r['instance_id'], 'reason': 'fixed_catalog_changed'})
        if r['method'] == 'capacity':
            frontier = r['frontier']
            values = [Fraction(f['net_profit']) for f in frontier]
            if values != sorted(values) or len(frontier) != len(payload['products']) + 1 or values[0] != 0:
                violations.append({'instance_id': r['instance_id'], 'reason': 'invalid_capacity_frontier'})
            for f in frontier:
                checks['capacity_frontier_menus'] += 1
                rr = replay_menu(payload, f['menu']['products'], f['menu']['prices'])
                if len(f['menu']['products']) > f['capacity'] or rr['accounting']['net_profit'] != f['net_profit']:
                    violations.append({'instance_id': r['instance_id'], 'reason': 'capacity_menu_mismatch', 'capacity': f['capacity']})
                else:
                    checks['capacity_frontier_replay_passed'] += 1
        jobs.setdefault(r['instance_id'], {})[r['method']] = r
    for instance in instances:
        key = instance['instance_id']
        methods = jobs.get(key, {})
        if set(methods) != set(METHODS):
            violations.append({'instance_id': key, 'reason': 'incomplete_method_set'})
            continue
        profit = {m: Fraction(r['accounting']['net_profit']) for m, r in methods.items()}
        checks['cross_method_instances'] += 1
        passed = (profit['joint'] >= profit['G2'] >= profit['G1'] >= profit['G0'] >= 0
                  and profit['fixed'] <= profit['joint']
                  and profit['contiguous'] <= profit['joint']
                  and profit['capacity'] == profit['joint'])
        if not passed:
            violations.append({'instance_id': key, 'reason': 'cross_method_objective_relation',
                               'profits': {m: str(p) for m, p in profit.items()}})
        else:
            checks['cross_method_passed'] += 1
    return {'status': 'pass' if not violations else 'fail', 'counts': dict(checks),
            'violations': violations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='JSONL constructed instances')
    parser.add_argument('--output-dir', type=Path, required=True, help='A new directory')
    parser.add_argument('--limit', type=int, help='First n input instances, for smoke checking only')
    parser.add_argument('--timeout', type=float, default=30.0, help='Seconds per isolated job')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory already exists. Choose a new path.')
    if args.timeout <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error('timeout and limit must be positive.')
    instances = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()]
    if args.limit:
        instances = instances[:args.limit]
    from pht_experiment import validate_instance
    for instance in instances:
        validate_instance(instance)
    if len({p['instance_id'] for p in instances}) != len(instances):
        raise ValueError('Duplicate instance identifiers.')
    args.output_dir.mkdir(parents=True)
    dump(args.output_dir / 'run_manifest.json', {
        'run_schema': 'pht-pilot-run-v1', 'purpose': 'pipeline verification',
        'input_file': args.input.name, 'input_sha256': digest(args.input),
        'instance_count': len(instances), 'methods': METHODS,
        'timeout_seconds_per_job': args.timeout, 'python': sys.version,
        'platform': platform.platform(), 'processor': platform.processor(),
        'cpu_count': os.cpu_count(), 'runner_sha256': digest(__file__),
        'interface_sha256': digest(Path(__file__).with_name('pht_experiment.py')),
        'measurement_policy': 'sequential isolated processes; fresh per-call oracle caches; each worker pinned to first available CPU when supported',
    })
    records = []
    started = time.perf_counter()
    with (args.output_dir / 'results.jsonl').open('x', encoding='utf-8') as stream:
        for i, instance in enumerate(instances, 1):
            for method in METHODS:
                base = {'instance_id': instance['instance_id'], 'method': method}
                try:
                    process = subprocess.run([sys.executable, __file__, '--worker'],
                        input=json.dumps({'instance': instance, 'method': method}, allow_nan=False),
                        capture_output=True, text=True, timeout=args.timeout,
                        env={**os.environ, 'PYTHONHASHSEED': '0'})
                    if process.returncode:
                        record = {**base, 'status': 'error', 'exit_code': process.returncode,
                                  'stderr': process.stderr, 'stdout': process.stdout}
                    else:
                        record = {**base, **json.loads(process.stdout)}
                except subprocess.TimeoutExpired as exc:
                    record = {**base, 'status': 'timeout', 'timeout_seconds': args.timeout}
                except Exception:
                    record = {**base, 'status': 'runner_error', 'traceback': traceback.format_exc()}
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
            if i % 24 == 0 or i == len(instances):
                print(f'Completed {i}/{len(instances)} instances; {len(records)} jobs.', flush=True)
    checks = exact_cross_checks(instances, records)
    dump(args.output_dir / 'checks.json', checks)
    counts = Counter(r['status'] for r in records)
    dump(args.output_dir / 'summary.json', {
        'purpose': 'pipeline verification, not a formal timing benchmark',
        'instances': len(instances), 'jobs': len(records), 'status_counts': dict(counts),
        'wall_seconds_including_process_startup': time.perf_counter() - started,
        'checks': checks, 'results_sha256': digest(args.output_dir / 'results.jsonl'),
    })
    print(json.dumps({'status_counts': dict(counts), 'checks': checks}, ensure_ascii=False), flush=True)
    return 0 if checks['status'] == 'pass' else 1


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        raise SystemExit(main())
