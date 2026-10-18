#!/usr/bin/env python3
"""Build predeclared formal inputs and jobs without loading solver results."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
from collections import Counter
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'protocol/formal_protocol.json'


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_protocol(path=PROTOCOL):
    return json.loads(Path(path).read_text()), digest_file(path)


def verify_sources(protocol):
    for item in protocol['source_snapshots']:
        if digest_file(ROOT / item['path']) != item['sha256']:
            raise ValueError('Predeclared source changed: ' + item['path'])


def gamma_quality(audit_row, gamma, decimal_places=10):
    """Recompute from retained normalized MAE; quantize only the final value."""
    delays = [240, 120, 60, 30, 15, 0]
    result = []
    with localcontext() as context:
        context.prec = 50
        for delay in delays:
            score = Decimal(1) / (Decimal(1) + Decimal(audit_row[f'nmae_delay_{delay}']))
            if gamma == '0.5':
                transformed = score.sqrt()
            elif gamma == '1':
                transformed = score
            elif gamma == '2':
                transformed = score * score
            else:
                raise ValueError('Gamma is not predeclared')
            result.append(format(transformed.quantize(Decimal(1).scaleb(-decimal_places),
                                rounding=ROUND_HALF_EVEN), f'.{decimal_places}f'))
    return result


def build_mapping(pilot, audit_rows, protocol, protocol_hash):
    audit = {(r['client'], r['evaluation_month']): r for r in audit_rows}
    output, audit_output = [], []
    window_quality = {}
    for base in pilot:
        meta = base['metadata']
        key = (meta['source_client'], meta['evaluation_month'])
        if key not in window_quality:
            window_quality[key] = {}
            for gamma in protocol['mapping_sensitivity']['gamma_values']:
                alphas = gamma_quality(audit[key], gamma)
                exact = list(map(Fraction, alphas))
                valid = all(x > 0 for x in exact) and all(x < y for x, y in zip(exact, exact[1:]))
                window_quality[key][gamma] = alphas
                audit_output.append({'client': key[0], 'month': key[1], 'gamma': gamma,
                                     'alpha': alphas, 'strictly_increasing': valid})
                if not valid:
                    raise ValueError('Predeclared gamma quality order failed: ' + repr(key))
            if window_quality[key]['1'] != [p['alpha'] for p in base['products']]:
                raise ValueError('Gamma one does not reproduce original quantized alpha')
        for gamma in protocol['mapping_sensitivity']['additional_gamma_values']:
            instance = copy.deepcopy(base)
            tag = gamma.replace('.', 'p')
            instance['instance_id'] += '-gamma' + tag
            for product, alpha in zip(instance['products'], window_quality[key][gamma]):
                product['alpha'] = alpha
            instance['metadata'].update({
                'base_instance_id': base['instance_id'], 'quality_mapping_gamma': gamma,
                'formal_protocol_sha256': protocol_hash,
                'quality_mapping_source': 'retained source audit normalized MAE; Decimal precision 50; final HALF_EVEN quantization to 10 decimal places',
                'quality_mapping_base_protocol_sha256': meta['protocol_sha256']})
            instance['metadata']['parameter_sources']['alpha_q'] = {
                'source': 'UCI raw-trace normalized MAE through gamma score and final 10-place quantization',
                'calibration': 'declared mapping sensitivity; no observed willingness-to-pay claim'}
            output.append(instance)
    return output, audit_output


def scaling_configurations(protocol):
    grid = protocol['scaling']
    configurations = {}
    for axis, values in grid['axis_values'].items():
        for value in values:
            dim = dict(grid['base_dimensions'])
            dim[axis] = value
            key = (dim['Q'], dim['B'], dim['T'])
            configurations.setdefault(key, []).append(axis)
    return sorted(configurations.items())


def make_scaling_instance(q, b, t, seed, profile, axes, protocol, protocol_hash):
    """A separate controlled generator with scale-normalized cost profiles."""
    if not 1 <= b <= q or t < 2:
        raise ValueError('Invalid scaling dimensions')
    levels = t - 1
    population = protocol['scaling']['population']
    if population % (b * levels):
        raise ValueError('Scaling population must divide evenly into buyer types')
    rng = random.Random(seed)
    increments = [rng.randint(1, 9) for _ in range(q)]
    total, cumulative = sum(increments), 0
    cost_seed_text = f'sec6-scaling-cost-v1|{seed}|{profile}'
    cost_seed = int.from_bytes(hashlib.sha256(cost_seed_text.encode()).digest()[:8], 'big')
    cost_rng = random.Random(cost_seed)
    products = []
    for index, increment in enumerate(increments, 1):
        cumulative += increment
        x = Fraction(index, q)
        y = Fraction(index - 1, q - 1) if q > 1 else Fraction(0)
        if profile == 'balanced':
            activation, service = Fraction(3, 2) + 3*x, Fraction(1, 20) + Fraction(7, 40)*y
        elif profile == 'activation_heavy':
            activation, service = 4 + 8*x, Fraction(1, 50) + Fraction(1, 25)*y
        elif profile == 'service_heavy':
            activation, service = Fraction(1, 5) + Fraction(1, 2)*y, Fraction(1, 10) + Fraction(3, 8)*y
        else:
            raise ValueError('Unknown cost profile')
        activation *= Fraction(cost_rng.randint(950, 1050), 1000)
        service *= Fraction(cost_rng.randint(950, 1050), 1000)
        products.append({'q': index, 'alpha': str(Fraction(cumulative, total)),
                         'activation_cost': str(activation), 'unit_service_cost': str(service)})
    caps = [(j*q+b-1)//b for j in range(1, b+1)]
    theta = [Fraction(1) + Fraction(3*j, levels-1) for j in range(levels)] if levels > 1 else [Fraction(1)]
    count = population // (b * levels)
    buyers = [{'type_id': f'b{group+1}_theta{j+1}', 'access_cap': cap,
               'theta': str(value), 'count': count}
              for group, cap in enumerate(caps) for j, value in enumerate(theta)]
    identity = f'scaling-Q{q}-B{b}-T{t}-seed{seed}-{profile}'
    return {'schema_version': 'pht-experiment-v1', 'instance_id': identity,
            'products': products, 'buyers': buyers,
            'metadata': {'data_kind': 'controlled_synthetic',
                'generator': 'build_formal.make_scaling_instance.v1',
                'formal_protocol_sha256': protocol_hash, 'seed': seed,
                'cost_seed': cost_seed, 'cost_profile': profile,
                'association': 'independent',
                'scaling_axes': axes, 'market_population': population,
                'dimensions': {'Q': q, 'B': b, 'K': b*levels, 'T': t},
                'parameter_sources': {name: {'source': 'predeclared controlled synthetic generator',
                                             'calibration': 'scenario_input'}
                                      for name in ['alpha_q','c_q','s_q','bar_q_k','theta_k','n_k']},
                'quality_scale': 'Strict positive random cumulative increments normalized to alpha_Q=1',
                'cost_scale': 'New normalized-position cost profiles; not a measured cost or extrapolation of trace price',
                'buyer_distribution': 'Uniform independent full permission-theta grid, fixed total integer population',
                'currency_unit': 'one_declared_reference_value_unit', 'empirical_buyer_calibration': False}}


def jobs_for(protocol, protocol_hash, all_real, pilot, mapping, scaling):
    jobs = []
    def append(suite, instance, input_path, line, method, repeat, mode, options=None):
        jobs.append({'job_id': f"{suite}:{instance['instance_id']}:{method}:rep{repeat}",
                     'suite': suite, 'instance_id': instance['instance_id'],
                     'input_path': input_path, 'input_line_index': line,
                     'method': method, 'repeat': repeat, 'timing_mode': mode,
                     'optimization_budget_seconds': protocol['execution']['optimization_budget_seconds'],
                     'wall_budget_seconds': protocol['execution']['wall_budget_seconds'],
                     'memory_limit_mib': protocol['execution']['memory_limit_mib'],
                     'options': options or {}, 'formal_protocol_sha256': protocol_hash})
    for line, instance in enumerate(all_real):
        append('real_full', instance, 'data/all/instances.jsonl', line, 'joint', 1, 'bulk_validation')
    for line, instance in enumerate(pilot):
        for method in protocol['comparison']['methods']:
            append('real_comparison', instance, 'data/pilot/instances.jsonl', line, method, 1, 'isolated_benchmark')
        append('real_capacity', instance, 'data/pilot/instances.jsonl', line, 'capacity', 1,
               'isolated_benchmark', {'capacity': 6})
    for line, instance in enumerate(mapping):
        append('mapping_sensitivity', instance, 'data/formal/mapping_instances.jsonl', line,
               'joint', 1, 'bulk_validation')
    for line, instance in enumerate(scaling):
        q = instance['metadata']['dimensions']['Q']
        methods = protocol['comparison']['methods'] if q <= protocol['scaling']['all_methods_q_max'] else ['joint']
        for repeat in range(1, protocol['scaling']['timing_repetitions'] + 1):
            for method in methods:
                append('synthetic_scaling', instance, 'data/formal/scaling_instances.jsonl', line,
                       method, repeat, 'isolated_benchmark')
    if len(jobs) != protocol['expected_counts']['total_jobs']:
        raise ValueError(f"Unexpected job count {len(jobs)}")
    if len({job['job_id'] for job in jobs}) != len(jobs):
        raise ValueError('Duplicate job identities')
    return jobs


def build(protocol_path=PROTOCOL):
    protocol, protocol_hash = load_protocol(protocol_path)
    verify_sources(protocol)
    all_real = read_jsonl(ROOT / 'data/all/instances.jsonl')
    pilot = read_jsonl(ROOT / 'data/pilot/instances.jsonl')
    with (ROOT / 'data/source/window_quality_audit.csv').open() as f:
        audit = list(csv.DictReader(f))
    mapping, mapping_audit = build_mapping(pilot, audit, protocol, protocol_hash)
    scaling = [make_scaling_instance(q, b, t, seed, cost, axes, protocol, protocol_hash)
               for (q, b, t), axes in scaling_configurations(protocol)
               for seed in protocol['scaling']['seeds']
               for cost in protocol['scaling']['cost_profiles']]
    output = ROOT / 'data/formal'
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'mapping_instances.jsonl', mapping)
    write_jsonl(output / 'scaling_instances.jsonl', scaling)
    write_json(output / 'mapping_quality_audit.json', mapping_audit)
    jobs = jobs_for(protocol, protocol_hash, all_real, pilot, mapping, scaling)
    write_jsonl(output / 'jobs.jsonl', jobs)
    registry = []
    for path, rows in [('data/all/instances.jsonl', all_real), ('data/pilot/instances.jsonl', pilot),
                       ('data/formal/mapping_instances.jsonl', mapping), ('data/formal/scaling_instances.jsonl', scaling)]:
        registry.append({'input_path': path, 'instances': len(rows), 'sha256': digest_file(ROOT / path)})
    write_json(output / 'input_registry.json', registry)
    summary = {'formal_protocol_id': protocol['protocol_id'], 'formal_protocol_sha256': protocol_hash,
               'formal_protocol_frozen_before_solving': True, 'solver_executed_by_builder': False,
               'all_real_instances': len(all_real), 'comparison_instances': len(pilot),
               'additional_mapping_instances': len(mapping), 'scaling_instances': len(scaling),
               'mapping_quality_vectors_checked': len(mapping_audit),
               'mapping_quality_vectors_strictly_ordered': sum(r['strictly_increasing'] for r in mapping_audit),
               'unique_scaling_coordinates': len(scaling_configurations(protocol)),
               'jobs': len(jobs), 'jobs_by_suite': dict(Counter(j['suite'] for j in jobs)),
               'jobs_by_method': dict(Counter(j['method'] for j in jobs)),
               'jobs_by_timing_mode': dict(Counter(j['timing_mode'] for j in jobs)),
               'job_manifest_sha256': digest_file(output / 'jobs.jsonl'),
               'input_registry': registry}
    write_json(output / 'build_summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    args = parser.parse_args()
    print(json.dumps(build(args.protocol), ensure_ascii=False, indent=2))
