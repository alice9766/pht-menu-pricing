#!/usr/bin/env python3
"""Resumable formal evaluation with isolated benchmarks and bulk validation.

Solver imports are warmed before timing. Each isolated job gets a fresh
process. Bulk jobs reuse pinned workers and label RSS as cumulative. Records
are append-only and keyed by job, protocol, input file and implementation hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'pht-formal-result-v1'
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def cpu_for_instance(instance_id, cpus):
    value = int.from_bytes(hashlib.sha256(instance_id.encode('utf-8')).digest()[:8], 'big')
    return cpus[value % len(cpus)]


def peak_rss_mib():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 ** 2 if sys.platform == 'darwin' else 1024)


def current_rss_mib(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return 0.0


def completed_resource_status(wall_seconds, peak_rss, job):
    """Classify a returned result using worker measurements as well as monitoring."""
    if peak_rss > job['memory_limit_mib']:
        return 'memory_limit_completed'
    if wall_seconds > job['wall_budget_seconds']:
        return 'over_budget_completed'
    return 'completed'


def _worker(connection, persistent, warm_method, cpu):
    """No numerical library is imported before single-thread/affinity setup."""
    for key in THREAD_KEYS:
        os.environ[key] = '1'
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {cpu})
    try:
        if warm_method.upper() == 'MILP':
            from milp_baseline import solve_milp_instance
            api = solve_milp_instance
        else:
            from benchmark_api import benchmark
            api = benchmark
        connection.send({'kind': 'ready', 'pid': os.getpid(),
                         'rss_mib': peak_rss_mib(), 'at': utc_now()})
        while True:
            command = connection.recv()
            if command['kind'] == 'stop':
                break
            job, payload = command['job'], command['payload']
            start = time.perf_counter()
            try:
                options = dict(job.get('options', {}))
                if job['method'].upper() == 'MILP':
                    result = api(payload, time_limit=job['optimization_budget_seconds'],
                                 threads=1, **options)
                else:
                    result = api(payload, method=job['method'],
                                 budget_seconds=job['optimization_budget_seconds'], **options)
                connection.send({'kind': 'result', 'result': result,
                                 'wall_seconds': time.perf_counter() - start,
                                 'rss_mib': peak_rss_mib()})
            except BaseException as exc:
                connection.send({'kind': 'error', 'error_type': type(exc).__name__,
                                 'error': str(exc), 'traceback': traceback.format_exc(),
                                 'wall_seconds': time.perf_counter() - start,
                                 'rss_mib': peak_rss_mib()})
            if not persistent:
                break
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send({'kind': 'startup_error', 'error_type': type(exc).__name__,
                             'error': str(exc), 'traceback': traceback.format_exc()})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


class InputStore:
    """Indexed JSONL access; one file scan and at most 256 parsed rows cached."""
    def __init__(self, root, cache_size=256):
        self.root, self.cache_size = Path(root), cache_size
        self.offsets, self.hashes, self.cache = {}, {}, OrderedDict()

    def file_hash(self, relative):
        if relative not in self.hashes:
            self.hashes[relative] = digest_file(self.root / relative)
        return self.hashes[relative]

    def load(self, relative, line_index):
        key = (relative, line_index)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        path = self.root / relative
        if relative not in self.offsets:
            offsets = []
            with path.open('rb') as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    offsets.append(offset)
            self.offsets[relative] = offsets
        offsets = self.offsets[relative]
        if type(line_index) is not int or not 0 <= line_index < len(offsets):
            raise ValueError(f'Invalid input line index {relative}:{line_index}')
        with path.open('rb') as stream:
            stream.seek(offsets[line_index])
            payload = json.loads(stream.readline())
        self.cache[key] = payload
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return payload


def load_jobs(path, mode, protocol_hash, store, implementation_hash):
    wanted = 'bulk_validation' if mode == 'bulk' else 'isolated_benchmark'
    jobs, seen = [], set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        job = json.loads(line)
        if job['job_id'] in seen:
            raise ValueError(f"Duplicate job_id {job['job_id']}")
        seen.add(job['job_id'])
        if job['formal_protocol_sha256'] != protocol_hash:
            raise ValueError(f"Protocol mismatch for {job['job_id']}")
        if job['timing_mode'] != wanted:
            continue
        if job['wall_budget_seconds'] <= 0 or job['memory_limit_mib'] <= 0:
            raise ValueError('Wall/memory budgets must be positive')
        job = dict(job)
        job['input_file_sha256'] = store.file_hash(job['input_path'])
        job['job_signature'] = canonical_digest({'job': job, 'implementation_sha256': implementation_hash})
        jobs.append(job)
    return jobs


def existing_records(path):
    """Read complete records; preserve only an interrupted final write separately."""
    path = Path(path)
    if not path.exists():
        return {}, Counter()
    latest, attempts = {}, Counter()
    good_end = 0
    with path.open('rb') as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                remainder = stream.read()
                if remainder:
                    raise ValueError('Malformed record before end of result log; no records were removed')
                backup = path.with_name(path.name + '.partial_tail.' + str(time.time_ns()))
                backup.write_bytes(line)
                with path.open('r+b') as trunc:
                    trunc.truncate(good_end)
                break
            latest[record['job_id']] = record
            attempts[record['job_id']] += 1
            good_end = stream.tell()
            if not line.endswith(b'\n'):
                with path.open('ab') as repair:
                    repair.write(b'\n')
                break
    return latest, attempts


def implementation_fingerprint(root):
    """Fingerprint executed solver/runner code; analysis/report scripts are separate."""
    root = Path(root)
    src = root / 'src'
    names = ('run_formal.py', 'benchmark_api.py', 'pht_experiment.py', 'milp_baseline.py')
    paths = [src / name for name in names if (src / name).is_file()]
    for directory in ('timed_solver', 'solver_snapshot'):
        paths.extend((src / directory).rglob('*.py'))
    entries = {str(path.relative_to(root)): digest_file(path) for path in sorted(set(paths))}
    return canonical_digest(entries), entries


def environment_record(cpus):
    versions = {}
    for package in ('numpy', 'scipy', 'pandas'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cpu_model = platform.processor()
    try:
        cpu_model = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                         if line.startswith('model name'))
    except (OSError, StopIteration):
        pass
    return {'python': sys.version, 'platform': platform.platform(), 'cpu_model': cpu_model,
            'library_versions': versions, 'assigned_cpus': cpus,
            'available_cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'threads': {key: '1' for key in THREAD_KEYS},
            'process_start_method': 'fork' if 'fork' in mp.get_all_start_methods() else 'spawn'}


def run(args):
    root = Path(args.package_root).resolve()
    protocol = root / args.protocol
    protocol_hash = digest_file(protocol)
    store = InputStore(root)
    protocol_data = json.loads(protocol.read_text())
    for source in protocol_data.get('source_snapshots', []):
        if store.file_hash(source['path']) != source['sha256']:
            raise ValueError('Frozen source snapshot hash mismatch: ' + source['path'])
    registry_path = root / protocol_data.get('input_registry_path', 'data/formal/input_registry.json')
    if registry_path.exists():
        for source in json.loads(registry_path.read_text()):
            if store.file_hash(source['input_path']) != source['sha256']:
                raise ValueError('Input registry hash mismatch: ' + source['input_path'])
    implementation_hash, implementation_files = implementation_fingerprint(root)
    jobs = load_jobs(root / args.jobs, args.mode, protocol_hash, store, implementation_hash)
    cpus_allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    cpus = [int(v) for v in args.cpus.split(',')] if args.cpus else cpus_allowed[:args.workers]
    if not cpus or len(cpus) > 4 or len(set(cpus)) != len(cpus) or not set(cpus) <= set(cpus_allowed):
        raise ValueError('Use one to four distinct allowed CPUs')
    output = Path(args.output_dir) if args.output_dir else root / 'results' / ('formal_' + args.mode)
    output.mkdir(parents=True, exist_ok=True)
    record_path, manifest_path = output / 'results.jsonl', output / 'run_manifest.json'
    identity = {'formal_protocol_sha256': protocol_hash, 'jobs_sha256': digest_file(root / args.jobs),
                'implementation_sha256': implementation_hash, 'mode': args.mode, 'cpu_ids': cpus}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['run_identity'] != identity:
            raise ValueError('Resume identity differs; use a new output directory')
    else:
        manifest = {'schema_version': 'pht-formal-run-v1', 'created_at': utc_now(), 'run_identity': identity,
                    'environment': environment_record(cpus), 'implementation_files': implementation_files,
                    'input_files': store.hashes, 'planned_mode_jobs': len(jobs),
                    'timing': {'start': 'after imports/CPU affinity, immediately before benchmark API call',
                               'wall_seconds': 'API parsing, optimization, recovery and replay; process startup excluded',
                               'optimization': 'result.phase_seconds.preprocess_optimization; cooperative limit stored per job in optimization_budget_seconds',
                               'hard_wall': 'job.wall_budget_seconds from command dispatch after warmed ready',
                               'rss': 'fresh process high-water mark for isolated; persistent cumulative high-water mark for bulk',
                               'bulk_use': 'correctness/value coverage, not per-job performance benchmarking',
                               'cpu_assignment': 'SHA256(instance_id) modulo fixed cpu_ids, identical across methods/repeats'},
                    'results': 'All statuses retained; hard kills without reported incumbent have result=null.'}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    latest, attempts = existing_records(record_path)
    todo = []
    for job in jobs:
        old = latest.get(job['job_id'])
        if old:
            if old.get('job_signature') != job['job_signature']:
                raise ValueError(f"Resume source/protocol/code mismatch for {job['job_id']}")
            if old['runner_status'] != 'interrupted':
                continue
        todo.append(job)
    if args.limit is not None:
        todo = todo[:args.limit]
    queues = {cpu: deque() for cpu in cpus}
    for job in todo:
        queues[cpu_for_instance(job['instance_id'], cpus)].append(job)
    context = mp.get_context('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')
    slots = {}
    completed, counters, dirty = 0, Counter(), 0
    begin, last_print = time.monotonic(), time.monotonic()
    stream = record_path.open('a', encoding='utf-8')

    def append(record):
        nonlocal completed, dirty
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        dirty += 1
        if dirty >= args.checkpoint_every or record['runner_status'] != 'completed':
            os.fsync(stream.fileno())
            dirty = 0
        completed += 1
        counters[record['runner_status'] + ':' + str(record.get('worker_status'))] += 1

    def close_slot(cpu, kill=False):
        slot = slots.pop(cpu)
        process = slot['process']
        if process.is_alive():
            if kill:
                process.terminate()
            else:
                try:
                    slot['pipe'].send({'kind': 'stop'})
                except (BrokenPipeError, EOFError, OSError):
                    pass
            process.join(timeout=.5)
            if process.is_alive():
                process.kill()
        process.join(timeout=.5)
        slot['pipe'].close()

    def dispatch(cpu):
        slot = slots[cpu]
        job = queues[cpu].popleft()
        payload = store.load(job['input_path'], job['input_line_index'])
        if payload['instance_id'] != job['instance_id']:
            raise ValueError(f"Manifest input identity mismatch: {job['job_id']}")
        attempts[job['job_id']] += 1
        base = {k: job[k] for k in ('job_id', 'suite', 'instance_id', 'method', 'repeat', 'timing_mode',
                                    'formal_protocol_sha256', 'input_file_sha256', 'job_signature')}
        base.update(schema_version=SCHEMA, input_sha256=canonical_digest(payload),
                    source_sha256=payload.get('metadata', {}).get('source_sha256'),
                    source_protocol_sha256=payload.get('metadata', {}).get('protocol_sha256'),
                    cpu_id=cpu, attempt=attempts[job['job_id']], worker_pid=slot['process'].pid,
                    started_at=utc_now(), optimization_budget_seconds=job['optimization_budget_seconds'],
                    wall_budget_seconds=job['wall_budget_seconds'], memory_limit_mib=job['memory_limit_mib'],
                    options=job.get('options', {}),
                    rss_scope='persistent_worker_cumulative' if args.mode == 'bulk' else 'fresh_process')
        slot.update(job=job, base=base, clock=time.monotonic(), stage='running', peak_rss=slot.get('peak_rss', 0))
        slot['pipe'].send({'kind': 'run', 'job': job, 'payload': payload})

    def finalize(cpu, status, message=None):
        slot = slots[cpu]
        base = dict(slot['base'])
        message = message or {}
        result = message.get('result')
        base.update(runner_status=status, worker_status=result.get('status') if isinstance(result, dict) else None,
                    result=result, wall_seconds=message.get('wall_seconds', time.monotonic() - slot['clock']),
                    process_peak_rss_mib=max(slot['peak_rss'], message.get('rss_mib', 0)), finished_at=utc_now())
        for name in ('error_type', 'error', 'traceback'):
            if name in message:
                base[name] = message[name]
        if status != 'completed' and result is None:
            base['incumbent_status'] = 'no_incumbent_reported_before_termination'
        if result is not None and result.get('input_sha256') not in (None, base['input_sha256']):
            base['runner_status'] = 'error'
            base['error'] = 'Solver input SHA differs from scheduled input'
        status = base['runner_status']
        if status == 'completed':
            status = base['runner_status'] = completed_resource_status(
                base['wall_seconds'], base['process_peak_rss_mib'], slot['job'])
        append(base)
        if args.mode == 'bulk' and status == 'completed' and queues[cpu]:
            dispatch(cpu)
        else:
            close_slot(cpu, kill=status != 'completed')

    def stop_handler(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    old_term = signal.signal(signal.SIGTERM, stop_handler)
    try:
        while any(queues.values()) or slots:
            for cpu in cpus:
                if cpu in slots or not queues[cpu]:
                    continue
                job = queues[cpu][0]
                parent, child = context.Pipe()
                process = context.Process(target=_worker,
                                          args=(child, args.mode == 'bulk', job['method'], cpu))
                process.start()
                child.close()
                slots[cpu] = {'process': process, 'pipe': parent, 'stage': 'starting',
                              'clock': time.monotonic(), 'peak_rss': 0, 'job': job}
            for cpu in list(slots):
                slot = slots[cpu]
                slot['peak_rss'] = max(slot['peak_rss'], current_rss_mib(slot['process'].pid))
                message = None
                try:
                    if slot['pipe'].poll():
                        message = slot['pipe'].recv()
                except (EOFError, OSError):
                    pass
                if message:
                    if message['kind'] == 'ready':
                        slot['peak_rss'] = max(slot['peak_rss'], message.get('rss_mib', 0))
                        dispatch(cpu)
                    elif message['kind'] in ('result', 'error'):
                        finalize(cpu, 'completed' if message['kind'] == 'result' else 'error', message)
                        continue
                    elif message['kind'] == 'startup_error':
                        # Attribute startup failure to the scheduled job without dropping it.
                        job = queues[cpu].popleft()
                        payload = store.load(job['input_path'], job['input_line_index'])
                        attempts[job['job_id']] += 1
                        slot['base'] = {**job, 'schema_version': SCHEMA, 'input_sha256': canonical_digest(payload),
                                        'cpu_id': cpu, 'attempt': attempts[job['job_id']],
                                        'rss_scope': 'fresh_process' if args.mode == 'isolated' else 'persistent_worker_cumulative',
                                        'started_at': utc_now(), 'worker_pid': slot['process'].pid}
                        finalize(cpu, 'error', message)
                        continue
                if cpu not in slots:
                    continue
                slot = slots[cpu]
                if slot['stage'] == 'starting':
                    if time.monotonic() - slot['clock'] > args.startup_timeout or not slot['process'].is_alive():
                        job = queues[cpu].popleft()
                        payload = store.load(job['input_path'], job['input_line_index'])
                        attempts[job['job_id']] += 1
                        slot['base'] = {**job, 'schema_version': SCHEMA, 'input_sha256': canonical_digest(payload),
                                        'cpu_id': cpu, 'attempt': attempts[job['job_id']],
                                        'rss_scope': 'fresh_process' if args.mode == 'isolated' else 'persistent_worker_cumulative',
                                        'started_at': utc_now(), 'worker_pid': slot['process'].pid}
                        finalize(cpu, 'startup_timeout', {'error': 'Worker did not complete import warmup'})
                elif slot['peak_rss'] > slot['job']['memory_limit_mib']:
                    finalize(cpu, 'memory_limit')
                elif time.monotonic() - slot['clock'] > slot['job']['wall_budget_seconds']:
                    finalize(cpu, 'hard_timeout')
                elif not slot['process'].is_alive() and not slot['pipe'].poll():
                    finalize(cpu, 'error', {'error': 'Worker exited without a result'})
            if time.monotonic() - last_print >= args.progress_seconds:
                print(json.dumps({'mode': args.mode, 'completed_this_invocation': completed,
                                  'scheduled_this_invocation': len(todo), 'elapsed_seconds': round(time.monotonic() - begin, 2),
                                  'status_counts': dict(counters)}, ensure_ascii=False), flush=True)
                last_print = time.monotonic()
            if slots:
                time.sleep(.01)
    except KeyboardInterrupt:
        for cpu in list(slots):
            if slots[cpu]['stage'] == 'running':
                finalize(cpu, 'interrupted')
            else:
                close_slot(cpu, kill=True)
        raise
    finally:
        for cpu in list(slots):
            close_slot(cpu, kill=True)
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        signal.signal(signal.SIGTERM, old_term)
    summary = {'mode': args.mode, 'mode_planned_jobs': len(jobs),
               'resumed_completed_jobs': len(jobs) - len(todo) if args.limit is None else sum(
                   j['job_id'] in latest and latest[j['job_id']]['runner_status'] != 'interrupted' for j in jobs),
               'completed_this_invocation': completed, 'elapsed_seconds': time.monotonic() - begin,
               'status_counts': dict(counters), 'results_path': str(record_path)}
    (output / 'last_invocation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=ROOT)
    parser.add_argument('--protocol', default='protocol/formal_protocol.json')
    parser.add_argument('--jobs', default='data/formal/jobs.jsonl')
    parser.add_argument('--mode', choices=('bulk', 'isolated'), required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--cpus', help='Comma separated allowed CPU IDs, fixed for comparable repetitions')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--limit', type=int, help='Run at most this many pending jobs for a smoke check')
    parser.add_argument('--checkpoint-every', type=int, default=25)
    parser.add_argument('--progress-seconds', type=float, default=15)
    parser.add_argument('--startup-timeout', type=float, default=60)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error('--limit must be nonnegative')
    if args.checkpoint_every < 1 or args.progress_seconds <= 0:
        parser.error('Checkpoint and progress intervals must be positive')
    main_path = str((Path(args.package_root) / 'src').resolve())
    if main_path not in sys.path:
        sys.path.insert(0, main_path)
    run(args)


if __name__ == '__main__':
    main()
