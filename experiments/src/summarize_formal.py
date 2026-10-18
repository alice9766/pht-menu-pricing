#!/usr/bin/env python3
"""Summarize recorded formal jobs without dropping failures or changing inputs.

All profits used for method comparisons come from canonical menu replay. The
MILP's weak-IC objective and numerical bound are reported separately. Real
scenario contrasts are paired before aggregation; equal-meter means are
provided alongside descriptive equal-instance means. Timing repetitions are
collapsed within each instance before paired speed comparisons.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
import statistics
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f'{path}:{line_number}: expected an object')
            yield value


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(Fraction(str(value)))
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def exact(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return Fraction(str(value))
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def stats(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not values:
        return {'n': 0, 'mean': None, 'median': None, 'p95': None, 'min': None, 'max': None}
    return {'n': len(values), 'mean': statistics.fmean(values),
            'median': statistics.median(values), 'p95': quantile(values, .95),
            'min': min(values), 'max': max(values)}


def metadata_index(paths):
    index, source = {}, []
    for path in paths:
        count = 0
        file_digest = hash_file(path)
        for item in read_jsonl(path):
            iid = item['instance_id']
            products = item.get('products', [])
            metadata = dict(item.get('metadata') or {})
            entry = {'metadata': metadata,
                     'product_count': len(products),
                     'parameters': metadata.get('dimensions') or {},
                     'input_sha256': hashlib.sha256(json.dumps(item,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()}
            if iid in index:
                if any(index[iid][key] != value for key,value in entry.items()):
                    raise ValueError(f'Conflicting input payload or metadata for {iid}')
                index[iid]['source_file_sha256s'].add(file_digest)
            else:
                entry['source_file_sha256s']={file_digest}
                index[iid] = entry
            count += 1
        source.append({'path': str(path), 'sha256': hash_file(path), 'rows': count})
    return index, source


def meta(record, inputs):
    value = dict(inputs.get(record['instance_id'], {}).get('metadata') or {})
    value.update(record.get('input_metadata') or {})
    return value


def result(record):
    return record.get('result') or {}


def profit(record):
    return exact((result(record).get('accounting') or {}).get('net_profit'))


def statuses(record):
    return (record.get('runner_status', record.get('status', 'unknown')),
            record.get('worker_status') or result(record).get('status') or 'not_reported')


def timeout(record):
    text = ' '.join(map(str, statuses(record))).lower()
    api = result(record)
    # HiGHS/SciPy status 1 is the configured solver limit, even when it returns
    # a valid incumbent and uses the short status text 'limit'. A numerical
    # optimization_budget_overrun_seconds value alone does not make an
    # optimal (raw_status=0) return an unfinished solve.
    if str(record.get('method', api.get('method', ''))).upper() == 'MILP':
        worker = str(statuses(record)[1]).lower()
        if api.get('raw_status') == 1 or worker.startswith('limit'):
            return True
    return any(token in text for token in ('timeout', 'time_limit', 'budget_exhausted', 'over_budget')) or bool(api.get('budget_exhausted'))


def is_completed(record):
    return statuses(record)[0] in ('completed', 'ok')


def optimization_overrun(record):
    api = result(record)
    for field in ('optimization_budget_overrun_seconds', 'budget_exceeded_overhead_seconds'):
        value = number(api.get(field))
        if value is not None and value > 0:
            return True
    budget = number(api.get('budget_seconds', api.get('optimization_budget_seconds', record.get('optimization_budget_seconds'))))
    elapsed = number((api.get('phase_seconds') or {}).get('preprocess_optimization'))
    return budget is not None and elapsed is not None and elapsed > budget


def is_successful_timing(record):
    if not is_completed(record) or timeout(record) or optimization_overrun(record) or profit(record) is None:
        return False
    if result(record).get('numeric_recovery_issue') or result(record).get('quality_comparison_eligible') is False:
        return False
    worker = str(statuses(record)[1]).lower()
    return not any(token in worker for token in ('unsupported', 'error', 'invalid', 'infeasible', 'unbounded', 'interrupted', 'memory_limit'))


def is_isolated(record):
    value = str(record.get('timing_mode', '')).lower()
    return 'isolat' in value and 'nonisolat' not in value


def phase(record, name):
    value = result(record).get('phase_seconds') or {}
    if name == 'preprocess_optimization':
        return number(value.get(name))
    if name == 'total':
        return number(value.get(name, record.get('wall_seconds')))
    return number(value.get(name))


def gamma(metadata):
    value = exact(metadata.get('quality_mapping_gamma', '1'))
    return '0.5' if value == Fraction(1, 2) else str(value)


def descriptor(record, inputs):
    m = meta(record, inputs)
    p = result(record).get('parameters') or inputs.get(record['instance_id'], {}).get('parameters') or {}
    return {'suite': record.get('suite', 'unspecified'), 'instance_id': record['instance_id'],
            'source_client': m.get('source_client'), 'evaluation_month': m.get('evaluation_month'),
            'association': m.get('association'), 'cost_profile': m.get('cost_profile'),
            'gamma': gamma(m), 'seed': m.get('seed'), 'parameters': p,
            'scaling_axes': m.get('scaling_axes') or [],
            'base_instance_id': m.get('base_instance_id', record['instance_id'])}


def clustered_summary(rows, field):
    clusters = collections.defaultdict(list)
    windows = collections.defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        client = row.get('source_client')
        if client is not None:
            clusters[str(client)].append(value)
            windows[(str(client), str(row.get('evaluation_month')))].append(value)
    meter_windows = collections.defaultdict(list)
    for (client, month), values in windows.items():
        meter_windows[client].append(statistics.fmean(values))
    return {'instance_weighted': stats(row.get(field) for row in rows),
            'window_equal': stats(statistics.fmean(v) for v in windows.values()),
            'meter_equal': stats(statistics.fmean(v) for v in meter_windows.values()),
            'cluster_unit': 'source meter; scenarios and repeated windows stay together',
            'inference': 'descriptive summaries; no independence assumption or confidence interval'}


def normalize_attempts(paths):
    attempts, raw_sources = [], []
    for path in paths:
        count = 0
        for record in read_jsonl(path):
            if 'instance_id' not in record or 'method' not in record:
                raise ValueError(f'Record without instance_id/method in {path}')
            record['_source_path'] = str(path)
            record['_source_line'] = count + 1
            attempts.append(record)
            count += 1
        raw_sources.append({'path': str(path), 'sha256': hash_file(path), 'rows': count})
    final = {}
    for ordinal, record in enumerate(attempts):
        identity = record.get('job_id') or '|'.join(map(str, [record.get('suite'), record['instance_id'], record['method'], record.get('repeat', 0), record.get('timing_mode')]))
        # A job hash protects against accidentally combining two protocols.
        key = (identity, record.get('formal_protocol_sha256'), record.get('input_sha256'))
        order = (int(record.get('attempt', 1)), ordinal)
        if key not in final or order > final[key][0]:
            final[key] = (order, record)
    return attempts, [x[1] for x in final.values()], raw_sources


def status_tables(attempts, final):
    tables = []
    for population, rows in [('all_attempts', attempts), ('final_jobs', final)]:
        groups = collections.defaultdict(list)
        for row in rows:
            groups[(row.get('suite', 'unspecified'), row['method'], row.get('timing_mode', 'unspecified'))].append(row)
        for (suite, method, mode), group in sorted(groups.items()):
            count = len(group)
            tables.append({'population': population, 'suite': suite, 'method': method, 'timing_mode': mode,
                           'jobs': count, 'instances': len({x['instance_id'] for x in group}),
                           'runner_status_counts': dict(collections.Counter(statuses(x)[0] for x in group)),
                           'worker_status_counts': dict(collections.Counter(statuses(x)[1] for x in group)),
                           'timeout_jobs': sum(timeout(x) for x in group),
                           'timeout_rate': sum(timeout(x) for x in group)/count,
                           'optimization_overrun_jobs': sum(optimization_overrun(x) for x in group),
                           'jobs_with_replayed_profit': sum(profit(x) is not None for x in group)})
    return tables


def representatives(final):
    groups = collections.defaultdict(list)
    for row in final:
        groups[(row.get('suite', 'unspecified'), row['instance_id'], row['method'])].append(row)
    reps, inconsistencies = {}, []
    for key, group in groups.items():
        available = [x for x in group if profit(x) is not None]
        exact_values = {profit(x) for x in available if result(x).get('proof_status') == 'exact'}
        if len(exact_values) > 1:
            inconsistencies.append({'suite': key[0], 'instance_id': key[1], 'method': key[2],
                                    'reason': 'exact profit differs across timing repeats',
                                    'values': sorted(map(str, exact_values))})
        # Quality is from the first completed replay, never the best observed run.
        certified = [x for x in available if result(x).get('proof_status') == 'exact']
        pool = (certified if key[2] in ('joint','capacity','fixed') and certified else available) or group
        reps[key] = sorted(pool, key=lambda x: (int(x.get('repeat', 0)), int(x.get('attempt', 1)), x.get('job_id', '')))[0]
    return reps, groups, inconsistencies


def exact_reference(reps, suite, iid):
    direct = reps.get((suite, iid, 'joint'))
    if direct is not None and profit(direct) is not None and result(direct).get('proof_status') == 'exact':
        return direct
    choices = [r for (s, i, m), r in reps.items() if i == iid and m == 'joint' and profit(r) is not None and result(r).get('proof_status') == 'exact']
    return choices[0] if choices else None


def quality_tables(reps, inputs, abs_tol, rel_tol):
    rows, unavailable = [], []
    for (suite, iid, method), row in sorted(reps.items()):
        if method == 'joint':
            continue
        ref = exact_reference(reps, suite, iid)
        if ref is None or profit(ref) is None or profit(row) is None:
            unavailable.append({'suite': suite, 'instance_id': iid, 'method': method,
                                'reason': 'no_exact_reference_for_this_instance' if ref is None or profit(ref) is None else 'no_replayed_feasible_menu',
                                'runner_status': statuses(row)[0], 'worker_status': statuses(row)[1]})
            continue
        optimum, returned = profit(ref), profit(row)
        gap = optimum-returned
        tol = abs_tol + rel_tol * max(abs(float(optimum)), abs(float(returned)))
        exact_method = result(row).get('proof_status') in ('exact', 'heuristic', 'exact_within_restricted_catalog_family')
        value = descriptor(row, inputs)
        value.update(method=method,optimization_scope=result(row).get('optimization_scope'),eligible_for_algorithm_quality=result(row).get('quality_comparison_eligible',True),numeric_recovery_issue=bool(result(row).get('numeric_recovery_issue')), reference_suite=ref.get('suite'), reference_profit_exact=str(optimum), replay_profit_exact=str(returned),
                     absolute_gap_exact=str(gap), absolute_gap=float(gap),
                     relative_gap=None if optimum <= 0 else float(gap/optimum),
                     reference_zero=optimum == 0, reference_negative=optimum < 0,
                     exact_profit_match=gap == 0,
                     within_numerical_tolerance=abs(float(gap)) <= tol,
                     numerical_tolerance=tol,
                     material_suboptimality=float(gap)>tol,
                     exceeds_exact_reference=float(gap)<-tol,
                     exact_arithmetic_comparison=exact_method,
                     runner_status=statuses(row)[0],worker_status=statuses(row)[1],
                     timeout=timeout(row),
                     weak_ic_objective=number(result(row).get('weak_ic_objective')),
                     weak_ic_upper_bound=number(result(row).get('profit_upper_bound')),
                     weak_ic_objective_minus_replay=number(result(row).get('objective_minus_replay')),
                     weak_ic_choices_match=result(row).get('choices_match'))
        rows.append(value)
    groups = collections.defaultdict(list)
    for row in rows:
        for grouping in ['all', 'cost_association_gamma']:
            key = (row['suite'],row['method'],grouping,
                   row['cost_profile'] if grouping != 'all' else None,
                   row['association'] if grouping != 'all' else None,
                   row['gamma'] if grouping != 'all' else None)
            groups[key].append(row)
    summaries=[]
    for (suite,method,grouping,cost,assoc,gam), group in sorted(groups.items(), key=lambda kv:str(kv[0])):
        eligible=[x for x in group if x['eligible_for_algorithm_quality']]
        summaries.append({'suite':suite,'method':method,'grouping':grouping,'cost_profile':cost,'association':assoc,'gamma':gam,
                          'paired_instances':len(group),'algorithm_quality_eligible_instances':len(eligible),'numeric_or_assignment_exclusions':len(group)-len(eligible),'exact_profit_matches':sum(x['exact_profit_match'] for x in eligible),
                          'within_numerical_tolerance':sum(x['within_numerical_tolerance'] for x in eligible),
                          'material_suboptimality':sum(x['material_suboptimality'] for x in eligible),
                          'material_suboptimality_all_returned_menus':sum(x['material_suboptimality'] for x in group),
                          'exceeds_exact_reference':sum(x['exceeds_exact_reference'] for x in group),
                          'zero_reference_instances':sum(x['reference_zero'] for x in group),
                          'negative_reference_instances':sum(x['reference_negative'] for x in group),
                          'relative_gap':clustered_summary(eligible,'relative_gap'),
                          'absolute_gap':clustered_summary(eligible,'absolute_gap'),
                          'all_returned_menu_relative_gap':clustered_summary(group,'relative_gap'),
                          'all_returned_menu_absolute_gap':clustered_summary(group,'absolute_gap'),
                          'replay_from_timeout_instances':sum(x['timeout'] for x in group),
                          'weak_ic_objective_minus_replay':stats(x['weak_ic_objective_minus_replay'] for x in group),
                          'weak_ic_choices_mismatched':sum(x['weak_ic_choices_match'] is False for x in group)})
    return rows,summaries,unavailable


def timing_tables(final, groups, inputs, planned=None):
    cases=[]
    expected=collections.defaultdict(list)
    for job in planned or []:
        if is_isolated(job):expected[(job.get('suite','unspecified'),job['instance_id'],job['method'])].append(job)
    for key in sorted(set(groups)|set(expected)):
        suite,iid,method=key
        group=groups.get(key,[])
        isolated=[x for x in group if is_isolated(x)]
        scheduled=expected.get(key,isolated)
        if not isolated and not scheduled:
            continue
        successful=[x for x in isolated if is_successful_timing(x)]
        row=descriptor((isolated or scheduled)[0],inputs)
        row.update(method=method,cpu_ids=sorted({x.get('cpu_id') for x in isolated if x.get('cpu_id') is not None}),scheduled_repeats=len(scheduled),recorded_repeats=len(isolated),attempted_repeats=len(isolated),completed_repeats=len(successful),
                   timeout_repeats=sum(timeout(x) for x in isolated),
                   optimization_overrun_repeats=sum(optimization_overrun(x) for x in isolated),
                   all_repeats_completed=len(successful)==len(scheduled) and bool(scheduled),
                   optimization_seconds=stats(phase(x,'preprocess_optimization') for x in successful)['median'],
                   total_seconds=stats(phase(x,'total') for x in successful)['median'],
                   peak_rss_mib=stats(number(x.get('process_peak_rss_mib')) for x in successful)['median'])
        cases.append(row)
    grouped=collections.defaultdict(list)
    for row in cases:
        grouped[(row['suite'],row['method'])].append(row)
    summaries=[]
    for (suite,method), group in sorted(grouped.items()):
        summaries.append({'suite':suite,'method':method,'scheduled_instances':len(group),'attempted_instances':sum(x['recorded_repeats']>0 for x in group),
                          'instances_with_successful_repeat':sum(x['completed_repeats']>0 for x in group),
                          'instances_all_repeats_completed':sum(x['all_repeats_completed'] for x in group),
                          'scheduled_repeats':sum(x['scheduled_repeats'] for x in group),'recorded_repeats':sum(x['recorded_repeats'] for x in group),'attempted_repeats':sum(x['attempted_repeats'] for x in group),
                          'timeout_repeats':sum(x['timeout_repeats'] for x in group),
                          'optimization_overrun_repeats':sum(x['optimization_overrun_repeats'] for x in group),
                          'optimization_seconds':stats(x['optimization_seconds'] for x in group),
                          'total_seconds':stats(x['total_seconds'] for x in group),
                          'peak_rss_mib':stats(x['peak_rss_mib'] for x in group)})
    lookup={(x['suite'],x['instance_id'],x['method']):x for x in cases}
    pairrows=[]
    for row in cases:
        if row['method']=='joint':continue
        ref=lookup.get((row['suite'],row['instance_id'],'joint'))
        if ref is None:continue
        value={k:v for k,v in row.items() if k not in ('optimization_seconds','total_seconds','peak_rss_mib')}
        # Require all attempted timing repetitions on both sides to finish.
        matched=bool(row['all_repeats_completed'] and ref['all_repeats_completed'] and row['cpu_ids']==ref['cpu_ids'])
        value.update(timing_pair_eligible=matched,joint_recorded_repeats=ref['recorded_repeats'],joint_all_repeats_completed=ref['all_repeats_completed'],same_cpu_assignment=row['cpu_ids']==ref['cpu_ids'],
                     speed_ratio_other_over_joint=None,end_to_end_ratio_other_over_joint=None)
        if matched:
            den=ref['optimization_seconds'];num=row['optimization_seconds']
            if den is not None and den>0 and num is not None:value['speed_ratio_other_over_joint']=num/den
            den=ref['total_seconds'];num=row['total_seconds']
            if den is not None and den>0 and num is not None:value['end_to_end_ratio_other_over_joint']=num/den
        pairrows.append(value)
    pairgroups=collections.defaultdict(list)
    for row in pairrows:pairgroups[(row['suite'],row['method'])].append(row)
    pairsummaries=[]
    for (suite,method),group in sorted(pairgroups.items()):
        pairsummaries.append({'suite':suite,'method':method,'instances_with_both_methods_scheduled':len(group),'instances_with_both_methods_attempted':sum(x['recorded_repeats']>0 and x['joint_recorded_repeats']>0 for x in group),
                              'matched_completed_instances':sum(x['timing_pair_eligible'] for x in group),
                              'excluded_due_to_incomplete_repeats':sum(not x['timing_pair_eligible'] for x in group),
                              'speed_ratio_other_over_joint':clustered_summary(group,'speed_ratio_other_over_joint'),
                              'end_to_end_ratio_other_over_joint':clustered_summary(group,'end_to_end_ratio_other_over_joint'),
                              'selection_note':'ratios condition on completed isolated repeats; status tables retain all timeouts and errors'})
    scaling_groups=collections.defaultdict(list)
    for row in cases:
        for axis in row['scaling_axes']:
            params=row['parameters'];value=params.get(axis)
            scaling_groups[(row['suite'],row['method'],axis,value,row['cost_profile'],json.dumps(params,sort_keys=True))].append(row)
    scale=[]
    for (suite,method,axis,value,cost,ptext),group in sorted(scaling_groups.items(),key=lambda kv:str(kv[0])):
        scale.append({'suite':suite,'method':method,'axis':axis,'axis_value':value,'cost_profile':cost,'parameters':json.loads(ptext),
                      'instances':len(group),'successful_instances':sum(x['completed_repeats']>0 for x in group),
                      'optimization_seconds':stats(x['optimization_seconds'] for x in group),
                      'total_seconds':stats(x['total_seconds'] for x in group),
                      'peak_rss_mib':stats(x['peak_rss_mib'] for x in group),
                      'timeout_repeats':sum(x['timeout_repeats'] for x in group),
                      'optimization_overrun_repeats':sum(x['optimization_overrun_repeats'] for x in group),
                      'attempted_repeats':sum(x['attempted_repeats'] for x in group),
                      'validated_completed_repeats':sum(x['completed_repeats'] for x in group)})
    return cases,summaries,pairrows,pairsummaries,scale


def scaling_quality_tables(rows):
    groups=collections.defaultdict(list)
    for row in rows:
        for axis in row['scaling_axes']:
            p=row['parameters']
            groups[(row['suite'],row['method'],axis,p.get(axis),row['cost_profile'],json.dumps(p,sort_keys=True))].append(row)
    output=[]
    for (suite,method,axis,value,cost,ptext),group in sorted(groups.items(),key=lambda kv:str(kv[0])):
        eligible=[x for x in group if x['eligible_for_algorithm_quality']]
        output.append({'suite':suite,'method':method,'axis':axis,'axis_value':value,'cost_profile':cost,
                       'parameters':json.loads(ptext),'paired_instances':len(group),'algorithm_quality_eligible_instances':len(eligible),
                       'numeric_or_assignment_exclusions':len(group)-len(eligible),
                       'exact_profit_matches':sum(x['exact_profit_match'] for x in eligible),
                       'within_numerical_tolerance':sum(x['within_numerical_tolerance'] for x in eligible),
                       'relative_gap':stats(x['relative_gap'] for x in eligible),
                       'absolute_gap':stats(x['absolute_gap'] for x in eligible),
                       'all_returned_menu_relative_gap':stats(x['relative_gap'] for x in group),
                       'timeout_incumbents':sum(x['timeout'] for x in group),
                       'seeds':sorted({x['seed'] for x in group if x['seed'] is not None})})
    return output


def global_joint(reps,inputs):
    groups=collections.defaultdict(list)
    for (suite,iid,method),row in reps.items():
        if method=='joint' and profit(row) is not None and result(row).get('proof_status')=='exact':groups[iid].append(row)
    output=[];mismatches=[]
    for iid,group in sorted(groups.items()):
        values={profit(x) for x in group}
        if len(values)>1:mismatches.append({'instance_id':iid,'reason':'exact joint value differs across suites','values':sorted(map(str,values))})
        row=group[0];m=descriptor(row,inputs);menu=result(row).get('menu') or {};products=menu.get('products') or [];prices=menu.get('prices') or {}
        inversions=[(a,b) for i,a in enumerate(sorted(products)) for b in sorted(products)[i+1:] if exact(prices[str(a)])>exact(prices[str(b)])]
        m.update(profit_exact=str(profit(row)),profit=float(profit(row)),products=products,prices=prices,
                 catalog_size=len(products),nonmonotone=bool(inversions),price_inversions=inversions,
                 subscription_population=sum((result(row).get('subscription_counts') or {}).values()),
                 source_suites=sorted({x.get('suite','unspecified') for x in group}))
        output.append(m)
    return output,mismatches


def structure_tables(joints):
    groups=collections.defaultdict(list)
    for row in joints:
        if row['source_client'] is None:continue
        for grouping in ['all','cost_association_gamma']:
            key=(grouping,row['cost_profile'] if grouping!='all' else None,row['association'] if grouping!='all' else None,row['gamma'])
            groups[key].append(row)
    summaries=[]
    for (grouping,cost,assoc,gam),group in sorted(groups.items(),key=lambda kv:str(kv[0])):
        summaries.append({'grouping':grouping,'cost_profile':cost,'association':assoc,'gamma':gam,
                          'instances':len(group),'source_meters':len({x['source_client'] for x in group}),
                          'source_windows':len({(x['source_client'],x['evaluation_month']) for x in group}),
                          'catalog_size_counts':dict(collections.Counter(x['catalog_size'] for x in group)),
                          'nonmonotone_returned_optimal_menus':sum(x['nonmonotone'] for x in group),
                          'nonmonotone_rate':clustered_summary([dict(x,nonmonotone_numeric=int(x['nonmonotone'])) for x in group],'nonmonotone_numeric'),
                          'profit':clustered_summary(group,'profit'),
                          'product_selection_counts':dict(collections.Counter(q for x in group for q in x['products'])),
                          'interpretation':'nonmonotonicity describes a returned optimal menu, not necessity across all optima'})
    look={(x['source_client'],x['evaluation_month'],x['cost_profile'],x['gamma'],x['association']):x for x in joints if x['source_client'] is not None}
    contrasts=[]
    for (client,month,cost,gam,assoc),left in sorted(look.items()):
        if assoc!='positive':continue
        for other in ('independent','negative'):
            right=look.get((client,month,cost,gam,other))
            if right is None:continue
            delta=exact(right['profit_exact'])-exact(left['profit_exact'])
            contrasts.append({'source_client':client,'evaluation_month':month,'cost_profile':cost,'gamma':gam,
                              'comparison':other+'_minus_positive','positive_instance_id':left['instance_id'],'other_instance_id':right['instance_id'],
                              'profit_difference_exact':str(delta),'profit_difference':float(delta),
                              'catalog_changed':left['products']!=right['products'],
                              'menu_changed':left['products']!=right['products'] or left['prices']!=right['prices']})
    cgroups=collections.defaultdict(list)
    for row in contrasts:cgroups[(row['comparison'],row['cost_profile'],row['gamma'])].append(row)
    csumm=[]
    for (comparison,cost,gam),group in sorted(cgroups.items()):
        csumm.append({'comparison':comparison,'cost_profile':cost,'gamma':gam,'paired_windows':len(group),
                      'profit_difference':clustered_summary(group,'profit_difference'),
                      'positive_difference':sum(x['profit_difference']>0 for x in group),'negative_difference':sum(x['profit_difference']<0 for x in group),
                      'zero_difference':sum(x['profit_difference']==0 for x in group),'catalog_changed':sum(x['catalog_changed'] for x in group),
                      'menu_changed':sum(x['menu_changed'] for x in group)})
    return summaries,contrasts,csumm


def gamma_tables(joints):
    look={(x['base_instance_id'],x['gamma']):x for x in joints if x['source_client'] is not None}
    rows=[]
    for (base,gam),r in sorted(look.items()):
        if gam=='1':continue
        ref=look.get((base,'1'))
        if ref is None:continue
        diff=exact(r['profit_exact'])-exact(ref['profit_exact'])
        row={k:r.get(k) for k in ('instance_id','base_instance_id','source_client','evaluation_month','cost_profile','association','gamma')}
        row.update(reference_instance_id=ref['instance_id'],profit_difference_exact=str(diff),profit_difference=float(diff),
                   catalog_changed=r['products']!=ref['products'],nonmonotone=r['nonmonotone'],reference_nonmonotone=ref['nonmonotone'],
                   catalog_size=r['catalog_size'],reference_catalog_size=ref['catalog_size'])
        rows.append(row)
    groups=collections.defaultdict(list)
    for row in rows:groups[(row['gamma'],row['cost_profile'],row['association'])].append(row)
    summaries=[]
    for (gam,cost,assoc),group in sorted(groups.items()):
        summaries.append({'gamma':gam,'cost_profile':cost,'association':assoc,'paired_instances':len(group),
                          'profit_difference_from_gamma1':clustered_summary(group,'profit_difference'),
                          'catalog_changed':sum(x['catalog_changed'] for x in group),
                          'nonmonotone_returned_optimal_menus':sum(x['nonmonotone'] for x in group)})
    return rows,summaries


def capacity_tables(reps, inputs):
    rows=[]
    for (suite,iid,method),record in sorted(reps.items()):
        frontier=result(record).get('frontier')
        if not frontier:continue
        ref=exact_reference(reps,suite,iid);opt=profit(ref) if ref else None
        values=[]
        for point in frontier:
            value=exact(point.get('net_profit',(point.get('accounting') or {}).get('net_profit')))
            values.append((point.get('capacity'),value))
        nondecreasing=all(a[1]<=b[1] for a,b in zip(values,values[1:]))
        attained=[h for h,v in values if opt is not None and v==opt]
        row=descriptor(record,inputs)
        row.update(method=method,frontier=[{'capacity':h,'net_profit_exact':str(v),'net_profit':float(v)} for h,v in values],
                   nondecreasing=nondecreasing,exact_reference_available=opt is not None,
                   minimal_capacity_attaining_joint_optimum=min(attained) if attained else None)
        rows.append(row)
    groups=collections.defaultdict(list)
    for row in rows:groups[(row['suite'],row['cost_profile'],row['association'],row['gamma'])].append(row)
    summaries=[]
    for (suite,cost,assoc,gam),group in sorted(groups.items(),key=lambda kv:str(kv[0])):
        points=collections.defaultdict(list)
        for row in group:
            for point in row['frontier']:points[point['capacity']].append(dict(row,value=point['net_profit']))
        summaries.append({'suite':suite,'cost_profile':cost,'association':assoc,'gamma':gam,'instances':len(group),
                          'nondecreasing_violations':sum(not x['nondecreasing'] for x in group),
                          'minimal_capacity_counts':dict(collections.Counter(x['minimal_capacity_attaining_joint_optimum'] for x in group if x['minimal_capacity_attaining_joint_optimum'] is not None)),
                          'frontier':{str(h):clustered_summary(v,'value') for h,v in sorted(points.items())}})
    return rows,summaries


def summarize(result_paths,input_paths,output_dir,abs_tol=1e-7,rel_tol=1e-8,jobs_path=None):
    inputs,input_sources=metadata_index(input_paths)
    planned=list(read_jsonl(jobs_path)) if jobs_path is not None else []
    attempts,final,result_sources=normalize_attempts(result_paths)
    reps,groups,issues=representatives(final)
    quality,quality_summary,quality_unavailable=quality_tables(reps,inputs,abs_tol,rel_tol)
    scaling_quality=scaling_quality_tables(quality)
    timing,timing_summary,timing_pairs,timing_pair_summary,scaling=timing_tables(final,groups,inputs,planned)
    joints,jointissues=global_joint(reps,inputs);issues.extend(jointissues)
    structures,association,association_summary=structure_tables(joints)
    gammacases,gammasummary=gamma_tables(joints)
    capacity,capacitysummary=capacity_tables(reps,inputs)
    status=status_tables(attempts,final)
    missing_jobs=[];unplanned_jobs=[];plan_source=None
    if jobs_path is not None:
        expected={(x['job_id'],x.get('formal_protocol_sha256')):x for x in planned}
        actual={(x.get('job_id'),x.get('formal_protocol_sha256')):x for x in final}
        missing_jobs=[{'job_id':x['job_id'],'suite':x.get('suite'),'method':x['method'],'instance_id':x['instance_id']} for k,x in expected.items() if k not in actual]
        unplanned_jobs=[{'job_id':x.get('job_id'),'suite':x.get('suite'),'method':x['method'],'instance_id':x['instance_id']} for k,x in actual.items() if k not in expected]
        plan_source={'path':str(jobs_path),'sha256':hash_file(jobs_path),'planned_jobs':len(planned),'matched_jobs':sum(k in actual for k in expected)}
        pg=collections.defaultdict(list)
        for x in planned:pg[(x.get('suite','unspecified'),x['method'],x.get('timing_mode','unspecified'))].append(x)
        for (suite,method,mode),group in sorted(pg.items()):
            count=len(group);done=[actual[(x['job_id'],x.get('formal_protocol_sha256'))] for x in group if (x['job_id'],x.get('formal_protocol_sha256')) in actual]
            status.append({'population':'planned_jobs','suite':suite,'method':method,'timing_mode':mode,'jobs':count,'recorded_jobs':len(done),'missing_jobs':count-len(done),'instances':len({x['instance_id'] for x in group}),'timeout_jobs':sum(timeout(x) for x in done),'timeout_rate':sum(timeout(x) for x in done)/count,'optimization_overrun_jobs':sum(optimization_overrun(x) for x in done),'runner_status_counts':dict(collections.Counter(statuses(x)[0] for x in done)),'worker_status_counts':dict(collections.Counter(statuses(x)[1] for x in done))})
    missing=sorted({x['instance_id'] for x in final if x['instance_id'] not in inputs})
    hash_mismatches=[]
    for record in final:
        info=inputs.get(record['instance_id'])
        if info is None:continue
        checks={'input_sha256':info['input_sha256']}
        if record.get('input_file_sha256') is not None and record['input_file_sha256'] not in info['source_file_sha256s']:
            hash_mismatches.append({'job_id':record.get('job_id'),'field':'input_file_sha256','recorded':record['input_file_sha256'],'expected':sorted(info['source_file_sha256s'])})
        for field,value in checks.items():
            if record.get(field) is not None and record[field]!=value:
                hash_mismatches.append({'job_id':record.get('job_id'),'field':field,'recorded':record[field],'expected':value})
        if result(record).get('input_sha256') is not None and result(record)['input_sha256']!=info['input_sha256']:
            hash_mismatches.append({'job_id':record.get('job_id'),'field':'result.input_sha256','recorded':result(record)['input_sha256'],'expected':info['input_sha256']})
        source=info['metadata'].get('source_sha256')
        if source is not None and record.get('source_sha256') is not None and source!=record['source_sha256']:
            hash_mismatches.append({'job_id':record.get('job_id'),'field':'source_sha256','recorded':record['source_sha256'],'expected':source})
    execution_complete=bool(jobs_path is not None and len(final)==len(planned) and not missing_jobs and not unplanned_jobs and all(statuses(x)[0] not in ('interrupted','pending') for x in final))
    material_exceeds=[{k:x[k] for k in ('suite','instance_id','method','absolute_gap_exact','numerical_tolerance')} for x in quality if x['exceeds_exact_reference']]
    tables={'status':status,'quality_by_instance':quality,'quality_summary':quality_summary,
            'quality_unavailable':quality_unavailable,'timing_by_instance':timing,'timing_summary':timing_summary,
            'timing_pairs':timing_pairs,'timing_pair_summary':timing_pair_summary,'scaling':scaling,'scaling_quality_summary':scaling_quality,
            'joint_unique_instances':joints,'trace_structure_summary':structures,
            'association_pairs':association,'association_summary':association_summary,
            'gamma_pairs':gammacases,'gamma_summary':gammasummary,
            'capacity_by_instance':capacity,'capacity_summary':capacitysummary,'missing_planned_jobs':missing_jobs,'unplanned_jobs':unplanned_jobs}
    summary={'schema_version':'pht-formal-summary-v1','input_sources':input_sources,'result_sources':result_sources,
             'execution_complete':execution_complete,'planned_job_source':plan_source,'raw_attempts':len(attempts),'final_jobs':len(final),'unique_input_instances_in_results':len({x['instance_id'] for x in final}),
             'suites':sorted({x.get('suite','unspecified') for x in final}),'status':status,
             'quality_summary':quality_summary,'timing_summary':timing_summary,'timing_pair_summary':timing_pair_summary,
             'scaling':scaling,'scaling_quality_summary':scaling_quality,'trace_structure_summary':structures,'association_summary':association_summary,
             'gamma_summary':gammasummary,'capacity_summary':capacitysummary,
             'audit':{'exact_repeat_or_suite_inconsistencies':issues,'material_values_above_exact_reference':material_exceeds,
                      'input_result_hash_mismatches':hash_mismatches,'missing_input_metadata_instances':missing,'unavailable_quality_pairs':len(quality_unavailable),'missing_planned_jobs':len(missing_jobs),'unplanned_jobs':len(unplanned_jobs)},
             'conventions':{'objective':'canonical replay net profit; weak-IC MILP objective is separate',
                            'relative_gap':'(exact joint optimum - replay profit) / exact optimum; null when optimum <= 0',
                            'numerical_diagnostics_tolerance':{'absolute':abs_tol,'relative':rel_tol},
                            'quality_repeat':'first available replay by repeat order; exact reference uses first certified optimum if available; never best-profit selection',
                            'timing':'median successful isolated repeats per case; ratios require all planned repeats on both methods complete on the same CPU assignment',
                            'timing_ratio':'other method optimization time / joint optimization time',
                            'timing_optimization_source':'result.phase_seconds.preprocess_optimization',
                            'timing_total_source':'result.phase_seconds.total; wall_seconds fallback only if total is absent; excludes process startup, imports, input file reads and result serialization',
                            'optimization_overrun':'reported positive optimization-budget overrun or measured optimization phase above its recorded budget; excluded from successful timing even when solve status is optimal; solver-limit status is separately counted as timeout',
                            'failed_jobs':'all attempts retained in source results; latest attempt defines final job status',
                            'trace_replication':'multiple cost, association and gamma scenarios share each source window',
                            'cluster_aggregation':'paired scenarios averaged within window, windows averaged within meter, then equal weight across meters',
                            'joint_full_pool_scope':'joint-only suites support menu/association summaries, not baseline quality comparisons',
                            'nonmonotone':'existence in returned optimum; not necessity among all optimal menus'}}
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    for name in ('summary.json','tables.json'):
        if (output_dir/name).exists():raise FileExistsError(f'Choose a new output directory: {output_dir/name}')
    for name,value in [('summary.json',summary),('tables.json',tables)]:
        (output_dir/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,nargs='+',required=True)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--jobs',type=Path,help='Frozen jobs.jsonl for planned-job coverage denominators')
    parser.add_argument('--absolute-tolerance',type=float,default=1e-7)
    parser.add_argument('--relative-tolerance',type=float,default=1e-8)
    args=parser.parse_args()
    summary=summarize(args.results,args.inputs,args.output_dir,args.absolute_tolerance,args.relative_tolerance,args.jobs)
    print(json.dumps({'raw_attempts':summary['raw_attempts'],'final_jobs':summary['final_jobs'],
                      'suites':summary['suites'],'audit':summary['audit']},ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
