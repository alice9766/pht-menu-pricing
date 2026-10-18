#!/usr/bin/env python3
"""Build the final prespecified experiment matrix without executing a solver.

Source trace/scenario construction and the synthetic generator are unchanged.
Selection reads only eligibility, source IDs and pre-existing selection hashes.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import build_formal as base

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'protocol/final_protocol.json'


def select_final_windows(ledger, pilot, protocol):
    rule = protocol['comparison']['selection']
    months = rule['evaluation_months']
    old_keys = {(r['metadata']['source_client'], r['metadata']['evaluation_month']) for r in pilot}
    used = {client for client, month in old_keys}
    if len(old_keys) != 24 or len(used) != 24:
        raise ValueError('Original 24 windows must have distinct meters')
    lookup = {(r['client'], r['evaluation_month']): r for r in ledger}
    chosen, monthly = [], []
    for month in months:
        retained = [lookup[key] for key in old_keys if key[1] == month]
        if len(retained) != rule['retained_per_month']:
            raise ValueError('Retained per-month count does not match protocol')
        if any(r['model_eligible'].lower() != 'true' for r in retained):
            raise ValueError('A retained window is no longer model eligible')
        eligible = sorted((r for r in ledger if r['evaluation_month'] == month and
                           r['model_eligible'].lower() == 'true'),
                          key=lambda r: (r['selection_hash'], r['client']))
        selected = sorted(retained, key=lambda r: (r['selection_hash'], r['client']))
        rejected_used = []
        for row in eligible:
            if len(selected) == rule['windows_per_month']:
                break
            if row['client'] in used:
                rejected_used.append(row['client'])
                continue
            selected.append(row)
            used.add(row['client'])
        if len(selected) != rule['windows_per_month']:
            raise ValueError('Insufficient unused eligible source meters')
        chosen.extend(selected)
        monthly.append({'month': month, 'eligible_candidates': len(eligible),
                        'retained_count': len(retained), 'added_count': len(selected)-len(retained),
                        'selected_count': len(selected), 'skipped_already_selected_meters': rejected_used,
                        'selected': [{'client': r['client'], 'selection_hash': r['selection_hash'],
                                      'retained_original': (r['client'], month) in old_keys} for r in selected]})
    if len(chosen) != 96 or len({r['client'] for r in chosen}) != 96:
        raise ValueError('Final comparison requires 96 windows from 96 distinct meters')
    return chosen, monthly


def final_coordinates(protocol):
    coords = dict(base.scaling_configurations(protocol))
    for item in protocol['scaling']['additional_coordinates']:
        key = tuple(item[d] for d in ('Q', 'B', 'T'))
        if key in coords:
            raise ValueError('Additional scale coordinate duplicates an existing coordinate')
        coords[key] = [item['axis']]
    return sorted(coords.items())


def method_order(methods, suite, iid, repeat, protocol):
    """Portable deterministic permutation; no platform random-generator dependence."""
    seed = protocol['execution']['method_order_seed']
    namespace = protocol['execution']['method_order_namespace']
    return sorted(methods, key=lambda method: (
        hashlib.sha256(f'{namespace}|{seed}|{suite}|{iid}|{repeat}|{method}'.encode()).hexdigest(), method))


def jobs_for(protocol, protocol_hash, all_real, comparison, mapping, scaling):
    jobs = []
    def append(suite, instance, input_path, line, method, repeat, mode, options=None):
        limits = protocol['budget_sensitivity'] if suite == 'budget_sensitivity' else protocol['execution']
        jobs.append({'job_id': f"{suite}:{instance['instance_id']}:{method}:rep{repeat}",
                     'suite': suite, 'instance_id': instance['instance_id'],
                     'input_path': input_path, 'input_line_index': line,
                     'method': method, 'repeat': repeat, 'timing_mode': mode,
                     'optimization_budget_seconds': limits['optimization_budget_seconds'],
                     'wall_budget_seconds': limits['wall_budget_seconds'],
                     'memory_limit_mib': protocol['execution']['memory_limit_mib'],
                     'options': options or {}, 'formal_protocol_sha256': protocol_hash})
    for line, instance in enumerate(all_real):
        append('real_full', instance, 'data/all/instances.jsonl', line, 'joint', 1, 'bulk_validation')
    for line, instance in enumerate(comparison):
        for repeat in range(1, protocol['comparison']['repetitions'] + 1):
            for method in method_order(protocol['comparison']['methods'], 'real_comparison',
                                       instance['instance_id'], repeat, protocol):
                append('real_comparison', instance, 'data/final/comparison_instances.jsonl', line,
                       method, repeat, 'isolated_benchmark')
        append('real_capacity', instance, 'data/final/comparison_instances.jsonl', line,
               'capacity', 1, 'isolated_benchmark', {'capacity': 6})
    for line, instance in enumerate(mapping):
        append('mapping_sensitivity', instance, 'data/final/mapping_instances.jsonl', line,
               'joint', 1, 'bulk_validation')
    for line, instance in enumerate(scaling):
        q = instance['metadata']['dimensions']['Q']
        methods = protocol['comparison']['methods'] if q <= 32 else protocol['scaling']['large_q_methods']
        for repeat in range(1, protocol['scaling']['timing_repetitions'] + 1):
            for method in method_order(methods, 'synthetic_scaling', instance['instance_id'], repeat, protocol):
                append('synthetic_scaling', instance, 'data/final/scaling_instances.jsonl', line,
                       method, repeat, 'isolated_benchmark')
        if all(instance['metadata']['dimensions'][d] == protocol['scaling']['base_dimensions'][d]
               for d in ('Q', 'B', 'T')):
            for repeat in range(1, protocol['budget_sensitivity']['repetitions'] + 1):
                for method in method_order(protocol['comparison']['methods'], 'budget_sensitivity',
                                           instance['instance_id'], repeat, protocol):
                    append('budget_sensitivity', instance, 'data/final/scaling_instances.jsonl', line,
                           method, repeat, 'isolated_benchmark')
    expected = protocol['expected_counts']
    actual = Counter(j['suite'] for j in jobs)
    if dict(actual) != expected['jobs_by_suite'] or len(jobs) != expected['total_jobs']:
        raise ValueError(f'Job count mismatch: {actual}')
    if len({j['job_id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate job IDs')
    return jobs


def validate_readiness(protocol, protocol_hash, all_real, pilot, comparison, mapping, scaling, jobs, mapping_audit):
    """Check the complete input matrix and pair bindings with exact fractions."""
    failures = []
    def require(condition, message):
        if not condition:
            failures.append(message)
    paths = {'data/all/instances.jsonl': all_real,
             'data/final/comparison_instances.jsonl': comparison,
             'data/final/mapping_instances.jsonl': mapping,
             'data/final/scaling_instances.jsonl': scaling}
    whole = {i['instance_id']: i for i in all_real}
    compare = {i['instance_id']: i for i in comparison}
    require(len(all_real) == 20232 and len(whole) == 20232, 'Fullpool count/unique IDs')
    require(len(comparison) == 864 and len(compare) == 864, 'Comparison count/unique IDs')
    require(all(i == whole[i['instance_id']] for i in comparison), 'Comparison must copy source payloads unchanged')
    require(all(i == compare[i['instance_id']] for i in pilot), 'All old pilot inputs retained unchanged')
    windows = {(i['metadata']['source_client'], i['metadata']['evaluation_month']) for i in comparison}
    require(len(windows) == 96 and len({m for m, _ in windows}) == 96, '96 distinct sources/windows')
    require(set(Counter(month for _, month in windows).values()) == {12}, 'Twelve windows per month')
    require(len(mapping_audit) == 288 and all(x['strictly_increasing'] for x in mapping_audit), 'All gamma vectors ordered')
    seen_mapping = Counter()
    for ins in mapping:
        original = compare[ins['metadata']['base_instance_id']]
        seen_mapping[(original['instance_id'], ins['metadata']['quality_mapping_gamma'])] += 1
        require(ins['buyers'] == original['buyers'], 'Mapping changed buyer parameters')
        require(all((p['q'], p['activation_cost'], p['unit_service_cost']) ==
                    (o['q'], o['activation_cost'], o['unit_service_cost'])
                    for p, o in zip(ins['products'], original['products'])), 'Mapping changed costs or IDs')
    require(len(seen_mapping) == 1728 and set(seen_mapping.values()) == {1}, 'Mapping pair completeness')
    coords, seeds = Counter(), set()
    for ins in scaling:
        meta = ins['metadata']; q, b, t = [meta['dimensions'][d] for d in ('Q', 'B', 'T')]
        coords[(q, b, t)] += 1; seeds.add(meta['seed'])
        require(len(ins['products']) == q and len(ins['buyers']) == b*(t-1), 'Synthetic Q/K dimensions')
        require(len({x['access_cap'] for x in ins['buyers']}) == b, 'Synthetic B dimension')
        require(len({Fraction(x['theta']) for x in ins['buyers']}) == t-1, 'Synthetic T dimension')
        require(sum(x['count'] for x in ins['buyers']) == 512, 'Synthetic population')
        require(all(type(x['count']) is int and x['count'] > 0 for x in ins['buyers']), 'Positive integer counts')
        group_count, theta_count = Counter(), Counter()
        for buyer in ins['buyers']:
            group_count[buyer['access_cap']] += buyer['count']
            theta_count[Fraction(buyer['theta'])] += buyer['count']
        require(set(group_count.values()) == {512//b} and set(theta_count.values()) == {512//(t-1)}, 'Uniform buyer margins')
    require(len(coords) == 17 and set(coords.values()) == {30}, '17 coordinates times 10 seeds times 3 costs')
    require(seeds == set(protocol['scaling']['seeds']), 'Seed completeness')
    for relative, inputs in paths.items():
        for ins in inputs:
            alphas = [Fraction(p['alpha']) for p in ins['products']]
            require(all(x > 0 for x in alphas) and all(x < y for x, y in zip(alphas, alphas[1:])), 'Quality order '+ins['instance_id'])
            require(all(Fraction(p['activation_cost']) >= 0 and Fraction(p['unit_service_cost']) >= 0
                        for p in ins['products']), 'Nonnegative costs')
    assigned = defaultdict(list)
    for job in jobs:
        require(paths[job['input_path']][job['input_line_index']]['instance_id'] == job['instance_id'], 'Job input binding')
        require(job['formal_protocol_sha256'] == protocol_hash, 'Job protocol binding')
        require(job['memory_limit_mib'] == 2048, 'Memory budget')
        expected_budget = (30, 45) if job['suite'] == 'budget_sensitivity' else (5, 10)
        require((job['optimization_budget_seconds'], job['wall_budget_seconds']) == expected_budget, 'Time budgets')
        assigned[(job['suite'], job['instance_id'], job['repeat'])].append(job['method'])
    for (suite, iid, repeat), methods in assigned.items():
        require(methods == method_order(methods, suite, iid, repeat, protocol), 'Frozen randomized method order')
    require(Counter(j['timing_mode'] for j in jobs) == {'bulk_validation': 21960, 'isolated_benchmark': 21384}, 'Timing mode totals')
    result = {'all_checks_passed': not failures, 'failures': failures,
              'solver_executed': False, 'solver_outcomes_used_in_selection': False,
              'fullpool_instances_checked': len(all_real), 'comparison_instances_checked': len(comparison),
              'comparison_distinct_meters': len({m for m, _ in windows}),
              'retained_original_instances_checked': len(pilot), 'mapping_vectors_checked': len(mapping_audit),
              'mapping_pairs_checked': len(mapping), 'synthetic_instances_checked': len(scaling),
              'job_bindings_checked': len(jobs), 'protocol_sha256': protocol_hash}
    if failures:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return result


def build(protocol_path=PROTOCOL):
    protocol, protocol_hash = base.load_protocol(protocol_path)
    base.verify_sources(protocol)
    all_real = base.read_jsonl(ROOT / 'data/all/instances.jsonl')
    pilot = base.read_jsonl(ROOT / 'data/pilot/instances.jsonl')
    with (ROOT / 'data/all/window_eligibility.csv').open() as f:
        ledger = list(csv.DictReader(f))
    with (ROOT / 'data/source/window_quality_audit.csv').open() as f:
        audit = list(csv.DictReader(f))
    chosen, monthly = select_final_windows(ledger, pilot, protocol)
    rank = {(r['client'], r['evaluation_month']): index for index, r in enumerate(chosen)}
    comparison = sorted((copy.deepcopy(i) for i in all_real
                         if (i['metadata']['source_client'], i['metadata']['evaluation_month']) in rank),
                        key=lambda i: (rank[(i['metadata']['source_client'], i['metadata']['evaluation_month'])], i['instance_id']))
    mapping, mapping_audit = base.build_mapping(comparison, audit, protocol, protocol_hash)
    scaling = [base.make_scaling_instance(q, b, t, seed, cost, axes, protocol, protocol_hash)
               for (q, b, t), axes in final_coordinates(protocol)
               for seed in protocol['scaling']['seeds'] for cost in protocol['scaling']['cost_profiles']]
    jobs = jobs_for(protocol, protocol_hash, all_real, comparison, mapping, scaling)
    readiness = validate_readiness(protocol, protocol_hash, all_real, pilot, comparison, mapping, scaling, jobs, mapping_audit)
    output = ROOT / 'data/final'
    output.mkdir(parents=True, exist_ok=True)
    base.write_json(output / 'selection_manifest.json', {
        'protocol_sha256': protocol_hash, 'selection_independent_of_solver_outputs': True,
        'method': protocol['comparison']['selection'], 'per_month': monthly})
    for name, rows in [('comparison_instances', comparison), ('mapping_instances', mapping),
                       ('scaling_instances', scaling), ('jobs', jobs)]:
        base.write_jsonl(output / (name + '.jsonl'), rows)
    base.write_json(output / 'mapping_quality_audit.json', mapping_audit)
    registry = [{'input_path': relative, 'instances': len(rows), 'sha256': base.digest_file(ROOT / relative)}
                for relative, rows in [('data/all/instances.jsonl', all_real),
                                       ('data/final/comparison_instances.jsonl', comparison),
                                       ('data/final/mapping_instances.jsonl', mapping),
                                       ('data/final/scaling_instances.jsonl', scaling)]]
    base.write_json(output / 'input_registry.json', registry)
    base.write_json(output / 'readiness_audit.json', readiness)
    summary = {'protocol_id': protocol['protocol_id'], 'formal_protocol_sha256': protocol_hash,
               'protocol_frozen_before_final_execution': True, 'solver_executed_by_builder': False,
               'all_real_instances': len(all_real), 'comparison_instances': len(comparison),
               'comparison_windows': len(chosen), 'comparison_distinct_meters': len({r['client'] for r in chosen}),
               'additional_mapping_instances': len(mapping), 'mapping_quality_vectors_checked': len(mapping_audit),
               'scaling_instances': len(scaling), 'unique_scaling_coordinates': len(final_coordinates(protocol)),
               'jobs': len(jobs), 'jobs_by_suite': dict(Counter(j['suite'] for j in jobs)),
               'jobs_by_method': dict(Counter(j['method'] for j in jobs)),
               'jobs_by_timing_mode': dict(Counter(j['timing_mode'] for j in jobs)),
               'job_manifest_sha256': base.digest_file(output / 'jobs.jsonl'),
               'builder_sha256': base.digest_file(Path(__file__)),
               'readiness_all_checks_passed': readiness['all_checks_passed'], 'input_registry': registry}
    base.write_json(output / 'build_summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    args = parser.parse_args()
    print(json.dumps(build(args.protocol), ensure_ascii=False, indent=2))
