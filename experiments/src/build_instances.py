#!/usr/bin/env python3
"""Construct deterministic PHT experiment inputs; never call a pricing solver.

Default input is the retained, hash-checked UCI 321 audit. Raw reconstruction is
provided by data/source/audit_windows.py and --verify-raw. Numeric instance
fields are decimal strings interpreted exactly as rationals by the solver.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from decimal import Decimal, ROUND_HALF_EVEN
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / 'protocol' / 'data_protocol.json'
SCHEMA = 'pht-experiment-v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def as_bool(value):
    return str(value).strip().lower() == 'true'


def quantize_alpha(value, places=10):
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError('Nonfinite quality input')
    return format(number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN), f'.{places}f')


def legacy_costs(profile):
    """Exact copy of original trace/verify_trace.py costs arithmetic and seeds."""
    seeds = {'balanced': 2026092110, 'activation_heavy': 2026092100,
             'service_heavy': 2026092120}
    rng = random.Random(seeds[profile])
    fixed, service = [], []
    for m in range(6):
        if profile == 'balanced':
            h, c = 1.5 + .5 * (m + 1), .05 + .035 * m
        elif profile == 'activation_heavy':
            h, c = 4.0 + 1.2 * (m + 1) ** 1.25, .02 + .008 * m
        else:
            h, c = .2 + .1 * m, .10 + .075 * m
        fixed.append(str(Fraction(str(round(h * rng.uniform(.95, 1.05), 9)))))
        service.append(str(Fraction(str(round(c * rng.uniform(.95, 1.05), 9)))))
    return fixed, service


def verify_raw_finiteness(archive, expected_sha256):
    """Optional official-archive verification, using pandas only in this path."""
    import zipfile
    import numpy as np
    import pandas as pd
    digest = sha256_file(archive)
    if digest != expected_sha256:
        raise ValueError('Official archive hash differs from protocol')
    cells = nonfinite = rows = 0
    with zipfile.ZipFile(archive) as z:
        with z.open('LD2011_2014.txt') as stream:
            for frame in pd.read_csv(stream, sep=';', decimal=',', chunksize=12000):
                values = frame.iloc[:, 1:].to_numpy(dtype=float)
                cells += values.size
                rows += len(frame)
                nonfinite += int((~np.isfinite(values)).sum())
    return {'archive_sha256': digest, 'rows': rows, 'numeric_cells': cells,
            'nonfinite_cells': nonfinite, 'all_raw_values_finite': nonfinite == 0}


def load_protocol(path=DEFAULT_PROTOCOL):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')), sha256_file(path)


def audit_windows(protocol, audit_path, finiteness_path):
    """Return every source row, a model eligibility decision and its reasons."""
    if sha256_file(audit_path) != protocol['source']['audit_csv_sha256']:
        raise ValueError('Source audit CSV hash differs from protocol')
    finite_audit = json.loads(Path(finiteness_path).read_text())
    if sha256_file(finiteness_path) != protocol['source']['finiteness_audit_sha256']:
        raise ValueError('Finiteness audit hash differs from protocol')
    if finite_audit['archive_sha256'] != protocol['source']['archive_sha256']:
        raise ValueError('Finiteness audit refers to another source archive')
    raw_finite = finite_audit['all_raw_values_finite']
    if not raw_finite:
        raise ValueError('A per-window finite-value audit is required for this archive')
    delays = protocol['products']['delay_minutes_in_quality_order']
    with Path(audit_path).open(newline='', encoding='utf-8') as f:
        source_rows = list(csv.DictReader(f))
    results = []
    keys = set()
    for original in source_rows:
        row = dict(original)
        key = (row['client'], row['evaluation_month'])
        if key in keys:
            raise ValueError(f'Duplicate window identity: {key}')
        keys.add(key)
        reasons = []
        if not as_bool(row['complete_window']):
            reasons.append('incomplete_calendar_window')
        reference_mean = float(row['reference_mean_kw'])
        if not math.isfinite(reference_mean) or reference_mean <= 0:
            reasons.append('nonpositive_or_nonfinite_reference_mean')
        if float(row['evaluation_zero_fraction']) == 1:
            reasons.append('all_zero_evaluation_window')
        activation = row['first_positive_timestamp']
        reference_first_day_end = datetime.fromisoformat(row['reference_month'] + '-01') + timedelta(days=1)
        if not activation:
            reasons.append('no_observed_activation')
        elif datetime.fromisoformat(activation) > reference_first_day_end:
            reasons.append('late_activation_in_reference')
        computable = as_bool(row['eligible'])
        if not computable and not reasons:
            reasons.append('failed_original_common_sample_audit')
        alphas = []
        if computable:
            try:
                alphas = [quantize_alpha(row[f'alpha_delay_{d}'],
                                        protocol['products']['alpha_decimal_places']) for d in delays]
            except (ValueError, ArithmeticError):
                reasons.append('nonfinite_quality')
            if alphas:
                exact = list(map(Fraction, alphas))
                if any(x <= 0 for x in exact):
                    reasons.append('nonpositive_quantized_quality')
                if any(x >= y for x, y in zip(exact, exact[1:])):
                    reasons.append('quantized_quality_not_strictly_increasing')
        identity = '|'.join([protocol['selection']['hash_namespace'],
                             protocol['source']['archive_sha256'], row['client'], row['evaluation_month']])
        row.update(raw_values_finite=raw_finite,
                   computable_before_quality_order=computable,
                   model_eligible=not reasons,
                   exclusion_reasons=';'.join(reasons),
                   selection_hash=hashlib.sha256(identity.encode('utf-8')).hexdigest())
        for d, value in zip(delays, alphas):
            row[f'alpha_quantized_delay_{d}'] = value
        results.append(row)
    expected = protocol['source']['expected_window_count']
    if len(results) != expected:
        raise ValueError(f'Expected {expected} audit windows, got {len(results)}')
    return results


def select_windows(ledger, protocol, use_all=False):
    selected, monthly = [], []
    for month in protocol['source']['evaluation_months']:
        candidates = [r for r in ledger if r['evaluation_month'] == month]
        eligible = sorted((r for r in candidates if r['model_eligible']),
                          key=lambda r: (r['selection_hash'], r['client']))
        take = len(eligible) if use_all else protocol['selection']['pilot_windows_per_month']
        if len(eligible) < take:
            raise ValueError(f'Insufficient eligible windows in {month}')
        chosen = eligible[:take]
        selected.extend(chosen)
        monthly.append({'month': month, 'candidate_windows': len(candidates),
                        'model_eligible_windows': len(eligible),
                        'excluded_windows': len(candidates) - len(eligible),
                        'selected_windows': len(chosen),
                        'exclusion_reason_counts': dict(Counter(reason for r in candidates
                            for reason in r['exclusion_reasons'].split(';') if reason)),
                        'selected': [{'client': r['client'], 'selection_hash': r['selection_hash']}
                                     for r in chosen]})
    return selected, monthly


def make_trace_instance(row, association, cost_profile, protocol, protocol_hash):
    fixed, service = legacy_costs(cost_profile)
    expected = protocol['cost_scenarios'][cost_profile]
    if fixed != expected['activation_cost_exact'] or service != expected['unit_service_cost_exact']:
        raise ValueError('Legacy cost generator differs from frozen protocol values')
    alphas = [row[f'alpha_quantized_delay_{d}'] for d in protocol['products']['delay_minutes_in_quality_order']]
    products = [{'q': q, 'alpha': value, 'activation_cost': fixed[q - 1],
                 'unit_service_cost': service[q - 1]} for q, value in enumerate(alphas, 1)]
    buyers = []
    counts = protocol['buyers']['counts_by_association'][association]
    for b, cap in enumerate(protocol['buyers']['access_caps']):
        for j, theta in enumerate(protocol['buyers']['theta_values']):
            buyers.append({'type_id': f'b{b + 1}_theta{j + 1}', 'access_cap': cap,
                           'theta': theta, 'count': counts[b][j]})
    instance_id = f"uci321-{row['client']}-{row['evaluation_month']}-{association}-{cost_profile}"
    metadata = {
        'construction_protocol': protocol['protocol_id'], 'protocol_sha256': protocol_hash,
        'dataset': 'UCI ElectricityLoadDiagrams20112014 (321)',
        'data_kind': 'real_trace_driven_simulation',
        'source_url': protocol['source']['url'],
        'source_sha256': protocol['source']['archive_sha256'],
        'source_audit_sha256': protocol['source']['audit_csv_sha256'],
        'source_client': row['client'], 'evaluation_month': row['evaluation_month'],
        'reference_month': row['reference_month'], 'selection_hash': row['selection_hash'],
        'quality_origin': 'same_meter_same_month_delayed_versions',
        'delay_minutes_in_quality_order': protocol['products']['delay_minutes_in_quality_order'],
        'shared_comparison_rows': int(row['comparison_rows']),
        'reference_mean_kw': row['reference_mean_kw'],
        'reference_active_first_day': as_bool(row['reference_active_first_day']),
        'association': association, 'cost_profile': cost_profile,
        'currency_unit': 'one_declared_reference_value_unit',
        'subscription_period': 'one_evaluation_calendar_month',
        'parameter_sources': protocol['parameter_sources'],
        'dimensions': {'Q': 6, 'B': 3, 'K': 12, 'T': 5},
        'market_population': 120,
    }
    return {'schema_version': SCHEMA, 'instance_id': instance_id,
            'products': products, 'buyers': buyers, 'metadata': metadata}


def build_trace(output_dir, protocol_path=DEFAULT_PROTOCOL, use_all=False):
    protocol, protocol_hash = load_protocol(protocol_path)
    audit_path = ROOT / protocol['source']['audit_csv_path']
    finite_path = ROOT / protocol['source']['finiteness_audit_path']
    ledger = audit_windows(protocol, audit_path, finite_path)
    selected, monthly = select_windows(ledger, protocol, use_all)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chosen_keys = {(r['client'], r['evaluation_month']) for r in selected}
    fields = list(ledger[0])
    for row in ledger:
        for key in row:
            if key not in fields:
                fields.append(key)
        row['selected_for_this_build'] = (row['client'], row['evaluation_month']) in chosen_keys
    fields.append('selected_for_this_build')
    ledger_path = output_dir / 'window_eligibility.csv'
    with ledger_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ledger)
    jsonl_path = output_dir / 'instances.jsonl'
    count = 0
    with jsonl_path.open('w', encoding='utf-8') as f:
        for row in selected:
            for association in protocol['buyers']['association_order']:
                for profile in protocol['cost_profile_order']:
                    instance = make_trace_instance(row, association, profile, protocol, protocol_hash)
                    f.write(json.dumps(instance, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
                    count += 1
    manifest = {'protocol_id': protocol['protocol_id'], 'protocol_sha256': protocol_hash,
                'mode': 'all_model_eligible_windows' if use_all else 'pilot_hash_selected_windows',
                'selection_independent_of_solver_outputs': True, 'per_month': monthly}
    write_json(output_dir / 'selection_manifest.json', manifest)
    summary = {'schema_version': SCHEMA, 'protocol_id': protocol['protocol_id'],
               'protocol_sha256': protocol_hash, 'source_sha256': protocol['source']['archive_sha256'],
               'candidate_windows': len(ledger),
               'quality_computable_windows': sum(r['computable_before_quality_order'] for r in ledger),
               'quantized_strict_quality_windows_before_activation_filter': sum(
                   r['computable_before_quality_order'] and
                   'quantized_quality_not_strictly_increasing' not in r['exclusion_reasons']
                   for r in ledger),
               'strict_quality_windows_excluded_by_activation_filter': sum(
                   r['computable_before_quality_order'] and
                   'quantized_quality_not_strictly_increasing' not in r['exclusion_reasons'] and
                   'late_activation_in_reference' in r['exclusion_reasons'] for r in ledger),
               'model_eligible_windows': sum(r['model_eligible'] for r in ledger),
               'excluded_windows': sum(not r['model_eligible'] for r in ledger),
               'selected_windows': len(selected), 'generated_instances': count,
               'eligible_distinct_sources': len({r['client'] for r in ledger if r['model_eligible']}),
               'selected_distinct_sources': len({r['client'] for r in selected}),
               'exclusion_reason_counts': dict(Counter(reason for r in ledger
                    for reason in r['exclusion_reasons'].split(';') if reason)),
               'instance_dimensions': {'Q': 6, 'B': 3, 'K': 12, 'T': 5},
               'solver_executed': False, 'numeric_representation': 'exact_rational_strings',
               'instances_sha256': sha256_file(jsonl_path),
               'eligibility_sha256': sha256_file(ledger_path),
               'selection_manifest_sha256': sha256_file(output_dir / 'selection_manifest.json')}
    write_json(output_dir / 'build_summary.json', summary)
    return summary


def make_synthetic_instance(q=12, b=3, theta_levels=4, seed=20261018):
    """Integer-count scale-control instance; not fitted to buyer transactions."""
    if q < 1 or not 1 <= b <= q or theta_levels < 1:
        raise ValueError('Require Q >= 1, 1 <= B <= Q, theta_levels >= 1')
    rng = random.Random(seed)
    alpha = Fraction(0)
    products = []
    for index in range(1, q + 1):
        alpha += Fraction(rng.randint(1, 9), 10)
        products.append({'q': index, 'alpha': str(alpha),
                         'activation_cost': str(Fraction(rng.randint(0, 50), 10)),
                         'unit_service_cost': str(Fraction(rng.randint(0, 20), 100))})
    caps = [(j * q + b - 1) // b for j in range(1, b + 1)]
    buyers = [{'type_id': f'b{group + 1}_theta{j}', 'access_cap': cap,
               'theta': str(j), 'count': rng.randint(1, 20)}
              for group, cap in enumerate(caps) for j in range(1, theta_levels + 1)]
    return {'schema_version': SCHEMA,
            'instance_id': f'synthetic-Q{q}-B{b}-T{theta_levels + 1}-seed{seed}',
            'products': products, 'buyers': buyers,
            'metadata': {'data_kind': 'controlled_synthetic', 'seed': seed,
                         'generator': 'build_instances.make_synthetic_instance.v1',
                         'dimensions': {'Q': q, 'B': b, 'K': b * theta_levels, 'T': theta_levels + 1},
                         'all_model_inputs': 'declared_synthetic_scenario',
                         'buyer_counts': 'positive_integers', 'empirical_buyer_calibration': False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'pilot')
    parser.add_argument('--all', action='store_true', help='Generate all model-eligible trace windows')
    parser.add_argument('--verify-raw', type=Path, help='Verify official raw zip hash and all raw values')
    parser.add_argument('--synthetic', action='store_true', help='Generate one controlled synthetic input')
    parser.add_argument('--Q', type=int, default=12)
    parser.add_argument('--B', type=int, default=3)
    parser.add_argument('--theta-levels', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261018)
    args = parser.parse_args()
    if args.verify_raw:
        protocol, _ = load_protocol(args.protocol)
        report = verify_raw_finiteness(args.verify_raw, protocol['source']['archive_sha256'])
        write_json(ROOT / protocol['source']['finiteness_audit_path'], report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.synthetic:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        instance = make_synthetic_instance(args.Q, args.B, args.theta_levels, args.seed)
        (args.output_dir / 'instances.jsonl').write_text(json.dumps(instance, ensure_ascii=False) + '\n')
        print(instance['instance_id'])
    else:
        print(json.dumps(build_trace(args.output_dir, args.protocol, args.all), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
