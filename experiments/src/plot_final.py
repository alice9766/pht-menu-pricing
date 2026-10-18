#!/usr/bin/env python3
"""Render final figures from fully audited summaries; no fabricated plot data.

Every plotted number is retained in figure_values.json. Timing distributions
use within-instance medians only when all planned repetitions completed;
completion denominators are retained alongside them. No success filter is
applied to the independent menu-quality comparisons.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from summarize_final import bootstrap_ci

METHODS = ['joint', 'G0', 'G1', 'G2', 'MILP']
LABELS = {'joint': 'PHT-DP', 'G0': 'G0', 'G1': 'G1', 'G2': 'G2', 'MILP': 'MILP'}
COLORS = {'joint': '#176b9f', 'G0': '#bf7b21', 'G1': '#34815b', 'G2': '#8b5797', 'MILP': '#62656a'}
MARKERS = {'joint': 'o', 'G0': 's', 'G1': '^', 'G2': 'D', 'MILP': 'v'}
PROFILES = ['balanced', 'activation_heavy', 'service_heavy']
PROFILE_LABELS = ['Balanced', 'Activation-heavy', 'Service-heavy']
ASSOCIATIONS = ['positive', 'independent', 'negative']


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def table_rows(summary_dir, tables, name):
    value = tables.get(name)
    if isinstance(value, list):
        return value
    entry = tables.get('table_files', {}).get(name)
    if entry is None:
        raise KeyError(f'Missing final table {name}')
    if isinstance(entry, dict):
        entry = entry.get('jsonl', entry.get('path', entry.get('jsonl_path')))
    path = Path(entry)
    if not path.is_absolute():
        path = summary_dir / path
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def avg(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def med(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.median(values) if values else None


def exact_number(x):
    return float(Fraction(str(x)))


def meter_values(rows, key):
    # Average scenarios within a source window, then windows within a meter.
    windows = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None:
            windows[(row['source_client'], row.get('evaluation_month'))].append(float(value))
    meters = defaultdict(list)
    for (meter, _), values in windows.items():
        meters[meter].append(avg(values))
    return {meter: avg(values) for meter, values in sorted(meters.items())}


def meter_mean(rows, key):
    return avg(meter_values(rows, key).values())


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def save_figure(fig, out, name):
    for suffix in ('.png', '.pdf'):
        fig.savefig(out / (name + suffix), bbox_inches='tight', dpi=240)
    plt.close(fig)


def validate_summary_for_plotting(summary):
    """Stop publication figures on integrity errors, not on retained failures.

    Missing-menu/timeout counts are experimental outcomes and stay reportable.
    Older summaries may omit newer audit fields; known fields are checked with
    empty defaults while any reported issue blocks final rendering.
    """
    if not summary.get('execution_complete'):
        raise ValueError('Final plots require all planned jobs to have a retained terminal status')
    audit = summary.get('audit', {})
    if not isinstance(audit, dict):
        raise ValueError('Final summary audit must be an object')
    error_fields = (
        'execution_provenance_issues', 'job_identity_mismatches',
        'cross_mapping_issues', 'exact_repeat_or_suite_inconsistencies',
        'material_values_above_exact_reference', 'input_result_hash_mismatches',
        'missing_input_metadata_instances', 'missing_planned_jobs', 'unplanned_jobs',
    )
    blocked = [name for name in error_fields if audit.get(name)]
    replay = audit.get('independent_menu_replay', {})
    if not isinstance(replay, dict):
        blocked.append('independent_menu_replay: malformed audit')
    elif replay.get('issues'):
        blocked.append('independent_menu_replay.issues')
    capacity_audit = audit.get('capacity_frontier_certification')
    if capacity_audit is not None:
        if not isinstance(capacity_audit, dict):
            blocked.append('capacity_frontier_certification: malformed audit')
        else:
            for name in ('unproven_frontiers', 'missing_frontiers'):
                if capacity_audit.get(name):
                    blocked.append('capacity_frontier_certification.' + name)
            planned = capacity_audit.get('planned_capacity_instances', 0)
            if planned and (capacity_audit.get('all_planned_frontiers_certified') is not True
                            or capacity_audit.get('recorded_frontiers') != planned
                            or capacity_audit.get('certified_optimal_frontiers') != planned):
                blocked.append('capacity_frontier_certification: planned/certified coverage')
    if blocked:
        raise ValueError('Resolve final summary integrity errors before rendering figures: ' + ', '.join(blocked))


def method_timing(timing, out):
    groups = {m: [x for x in timing if x['suite'] == 'real_comparison' and x['method'] == m] for m in METHODS}
    if any(not rows for rows in groups.values()):
        raise ValueError('The final trace comparison must include all five methods')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True, gridspec_kw={'width_ratios': [1.2, 1]})
    records = []
    for i, method in enumerate(METHODS):
        rows = groups[method]
        complete = [x for x in rows if x['all_repeats_completed']]
        for offset, field, color in [(-.17, 'optimization_seconds', '#337da6'), (.17, 'total_seconds', '#b8cddb')]:
            values = [x[field] for x in complete if x.get(field) is not None and x[field] > 0]
            if values:
                axes[0].boxplot([values], positions=[i + offset], widths=.27, patch_artist=True,
                    boxprops={'facecolor': color, 'edgecolor': '#384957'}, medianprops={'color': '#111111'},
                    whiskerprops={'color': '#384957'}, capprops={'color': '#384957'}, showfliers=False)
        n = len(complete)
        axes[1].bar(i, n / len(rows), color=COLORS[method], edgecolor='white', width=.66)
        axes[1].text(i, n / len(rows) + .025, f'{n}\n/{len(rows)}', ha='center', va='bottom', fontsize=8.5)
        records.append({'method': method, 'scheduled_instances': len(rows), 'all_repeats_completed_instances': n,
            'scheduled_repeats': sum(x['scheduled_repeats'] for x in rows),
            'optimization_seconds': [x.get('optimization_seconds') for x in complete],
            'total_seconds': [x.get('total_seconds') for x in complete],
            'median_optimization_seconds': med(x.get('optimization_seconds') for x in complete),
            'median_total_seconds': med(x.get('total_seconds') for x in complete)})
    axes[0].set_xticks(range(len(METHODS)), [LABELS[m] for m in METHODS]); axes[0].set_yscale('log')
    axes[0].set_ylabel('Time per instance (s)'); axes[0].set_title('(a) Completed runs\nOptimization and total API')
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(facecolor='#337da6', label='Optimization'), Patch(facecolor='#b8cddb', label='Total API')], loc='upper left', fontsize=8.5)
    axes[1].set_xticks(range(len(METHODS)), [LABELS[m] for m in METHODS]); axes[1].set_ylim(0, 1.25)
    axes[1].set_ylabel('Fraction of scheduled instances'); axes[1].set_title('(b) Instances with all\ntiming repetitions completed')
    axes[1].set_yticks([0, .25, .5, .75, 1], ['0', '25%', '50%', '75%', '100%'])
    save_figure(fig, out, 'sec6_01_method_timing')
    return records


def scalability(timing, out):
    rows = [x for x in timing if x['suite'] == 'synthetic_scaling']
    coords = defaultdict(list)
    for row in rows:
        d = row['parameters']; coords[(d['Q'], d['B'], d['T'], row['method'])].append(row)
    if not coords:
        raise ValueError('No synthetic scaling results')
    base = (32, 4, 9)
    specs = [('Q', [(q, 4, 9) for q in [8, 16, 32, 64, 128, 256, 512]]),
             ('B', [(32, b, 9) for b in [1, 2, 4, 8, 16]]),
             ('T', [(32, 4, t) for t in [3, 5, 9, 17, 33]]),
             ('joint', [base, (64, 8, 17), (128, 16, 33)])]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9), constrained_layout=True)
    records = []
    for panel, (axis, points) in enumerate(specs):
        ax = axes.flat[panel]
        for method in METHODS:
            xs, ys = [], []
            for j, point in enumerate(points):
                entries = coords.get(point + (method,), [])
                if not entries:
                    continue
                complete = [x for x in entries if x['all_repeats_completed']]
                measured = [x['optimization_seconds'] for x in complete if x.get('optimization_seconds') is not None]
                x = j if axis == 'joint' else point['QBT'.index(axis)]
                y = med(measured)
                records.append({'axis': axis, 'coordinate': point, 'x': x, 'method': method,
                    'scheduled_instances': len(entries), 'all_repeats_completed_instances': len(complete),
                    'median_completed_optimization_seconds': y, 'completed_instance_medians': measured,
                    'timeout_repeats': sum(e.get('timeout_repeats', 0) for e in entries),
                    'optimization_overrun_repeats': sum(e.get('optimization_overrun_repeats', 0) for e in entries)})
                xs.append(x); ys.append(y if y is not None else np.nan)
                if y is not None and len(complete) < len(entries):
                    ax.scatter(x, y, marker='x', s=65, color=COLORS[method], linewidths=1.5, zorder=6)
                if y is None:
                    # Marker is on the budget guide only, never used as a timing value.
                    ax.scatter(x, 5, marker='x', s=60, color=COLORS[method], linewidths=1.6, zorder=6)
            if xs:
                ax.plot(xs, ys, marker=MARKERS[method], color=COLORS[method], label=LABELS[method], markersize=4, linewidth=1.3)
        ax.axhline(5, color='#999999', ls='--', lw=.8)
        ax.set_yscale('log'); ax.set_ylabel('Completed optimization time (s)')
        if axis == 'joint':
            ax.set_xticks(range(len(points)), [f'({q}, {b}, {t})' for q, b, t in points]); ax.set_xlabel('Joint scale (Q, B, T)')
        else:
            values = [x['QBT'.index(axis)] for x in points]
            ax.set_xscale('log', base=2); ax.set_xticks(values, [str(x) for x in values]); ax.set_xlabel({'Q': 'Candidate products Q', 'B': 'Permission groups B', 'T': 'Candidate thresholds T'}[axis])
        ax.set_title(['(a) Product count', '(b) Permission groups', '(c) Candidate thresholds', '(d) Joint growth'][panel])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, ncol=5, loc='outside upper center')
    save_figure(fig, out, 'sec6_02_scalability')
    return records


def capacity_figure(capacity, joints, out):
    ref = {x['instance_id']: exact_number(x['profit_exact']) for x in joints}
    groups = defaultdict(list); minimum = defaultdict(int); unavailable = []; uncertified = []
    for row in capacity:
        if row['suite'] != 'real_capacity':
            continue
        if row.get('certified_optimal_frontier') is not True:
            uncertified.append(row['instance_id']); continue
        minimum[row['minimal_capacity_attaining_joint_optimum']] += 1
        iid = row['instance_id']; optimum = ref.get(iid)
        if optimum is None or optimum <= 0:
            unavailable.append(iid); continue
        for p in row['frontier']:
            groups[(row['cost_profile'], p['capacity'])].append(dict(row, fraction=exact_number(p['net_profit_exact']) / optimum))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True, gridspec_kw={'width_ratios': [1.2, 1]})
    records = []
    ls = sorted({l for _, l in groups})
    for profile, label, marker, linestyle in zip(PROFILES, PROFILE_LABELS, ['o', 's', '^'], ['-', '--', ':']):
        values = [meter_mean(groups[profile, l], 'fraction') for l in ls]
        confidence = [bootstrap_ci(meter_values(groups[profile, l], 'fraction').values()) for l in ls]
        lower = [ci['lower'] if ci else np.nan for ci in confidence]
        upper = [ci['upper'] if ci else np.nan for ci in confidence]
        line, = axes[0].plot(ls, values, marker=marker, linestyle=linestyle, label=label, linewidth=1.5, markersize=4)
        axes[0].fill_between(ls, lower, upper, color=line.get_color(), alpha=.17, linewidth=0)
        records.append({'cost_profile': profile, 'capacity': ls, 'meter_equal_mean_fraction': values,
                        'meter_equal_mean_ci95': confidence})
    axes[0].set_xlabel('Maximum offered products L'); axes[0].set_ylabel('Fraction of unrestricted optimal profit')
    axes[0].set_xticks(ls); axes[0].set_ylim(-.02, 1.04); axes[0].legend(fontsize=8.5)
    axes[0].set_title('(a) Optimal profit\nwith a product limit')
    ks = sorted(k for k in minimum if k is not None)
    axes[1].bar(ks, [minimum[k] for k in ks], color='#176b9f', width=.64)
    for k in ks: axes[1].text(k, minimum[k], str(minimum[k]), ha='center', va='bottom', fontsize=9)
    axes[1].set_xticks(ks); axes[1].set_xlabel('Smallest L attaining\nunrestricted optimum')
    axes[1].set_ylabel('Subscription scenarios'); axes[1].set_title('(b) Products needed\nto attain the optimum')
    axes[1].margins(y=.13)
    save_figure(fig, out, 'sec6_03_capacity')
    return {'curves': records, 'minimum_capacity_counts': dict(minimum),
            'excluded_uncertified_frontiers': uncertified,
            'excluded_zero_or_missing_reference_from_relative_curves': unavailable}


def association_mapping(joints, cross, out):
    full = [x for x in joints if 'real_full' in x.get('source_suites', []) and x.get('gamma') == '1']
    lookup = {(x['source_client'], x['evaluation_month'], x['cost_profile'], x['association']): x for x in full}
    changes = defaultdict(list); rates = defaultdict(list)
    for row in full:
        rates[(row['cost_profile'], row['association'])].append(dict(row, rate=float(row['nonmonotone'])))
        if row['association'] != 'positive': continue
        other = lookup.get((row['source_client'], row['evaluation_month'], row['cost_profile'], 'negative'))
        value = exact_number(row['profit_exact'])
        if other and value > 0:
            changes[row['cost_profile']].append(dict(row, percent_change=100*(exact_number(other['profit_exact'])-value)/value))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), constrained_layout=True, gridspec_kw={'width_ratios': [1, 1.05, 1.1]})
    distributions = [list(meter_values(changes[p], 'percent_change').values()) for p in PROFILES]
    axes[0].boxplot(distributions, tick_labels=['Bal.', 'Act.', 'Svc.'], showfliers=False)
    axes[0].axhline(0, color='#777777', ls='--', lw=.8)
    axes[0].set_ylabel('Net profit change (%)'); axes[0].set_xlabel('Cost scenario'); axes[0].set_title('(a) Negative vs positive\nassociation')
    inversion = []
    for i, assoc in enumerate(ASSOCIATIONS):
        ys = [100 * meter_mean(rates[p, assoc], 'rate') for p in PROFILES]
        axes[1].bar(np.arange(3)+(i-1)*.25, ys, .25, label=assoc)
        inversion.extend({'cost_profile': p, 'association': assoc, 'meter_equal_percent': y} for p, y in zip(PROFILES, ys))
    axes[1].set_xticks(range(3), ['Bal.', 'Act.', 'Svc.'])
    axes[1].set_ylabel('Menus with price inversion (%)'); axes[1].set_xlabel('Cost scenario'); axes[1].set_ylim(0, 105)
    axes[1].set_title('(b) Returned\noptimal menus'); axes[1].legend(fontsize=8.5, loc='lower left')
    gamma_values = ['0.5', '1', '2']; matrix = np.full((3, 3), np.nan); cross_values = []
    for i, source in enumerate(gamma_values):
        for j, target in enumerate(gamma_values):
            rows = [x for x in cross if str(x['source_gamma']) == source and str(x['target_gamma']) == target]
            value = meter_mean(rows, 'relative_regret')
            if value is not None: matrix[i, j] = 100 * value
            cross_values.append({'source_gamma': source, 'target_gamma': target, 'paired_instances': len(rows), 'meter_equal_regret_percent': None if value is None else 100 * value})
    image = axes[2].imshow(matrix, cmap='Blues', vmin=0, aspect='equal')
    limit = np.nanmax(matrix)
    for i in range(3):
        for j in range(3):
            v = matrix[i, j]; text = 'NA' if np.isnan(v) else f'{v:.2f}%'
            axes[2].text(j, i, text, ha='center', va='center', fontsize=8.5, color='white' if np.isfinite(v) and v > .6 * limit else '#111111')
    axes[2].set_xticks(range(3), gamma_values); axes[2].set_yticks(range(3), gamma_values)
    axes[2].set_xlabel(r'Evaluation $\gamma$'); axes[2].set_ylabel(r'Menu-design $\gamma$')
    axes[2].set_title('(c) Cross-mapping\nrelative regret (%)'); axes[2].grid(False)
    # Each cell carries its numerical percentage; a separate color bar would
    # crowd the three-panel conference-width layout.
    save_figure(fig, out, 'sec6_04_association_mapping')
    return {'association_meter_percent_changes': {p: meter_values(changes[p], 'percent_change') for p in PROFILES},
            'price_inversion_rates': inversion, 'cross_mapping': cross_values, 'full_pool_instances': len(full)}


def export_tables(summary, tables, timing, quality, out):
    # Table 1 uses the summary's explicit provenance/coverage populations.
    populations = summary.get('experiment_populations', [])
    if isinstance(populations, dict):
        populations = [dict(population=k, **v) if isinstance(v, dict) else {'population': k, 'value': v} for k, v in populations.items()]
    write_csv(out / 'table1_experiment_coverage.csv', populations)
    groups = defaultdict(list)
    for row in timing: groups[(row['suite'], row['method'])].append(row)
    qgroups = defaultdict(list)
    for row in quality: qgroups[(row['suite'], row['method'])].append(row)
    coverage = {(x['suite'], x['method']): x for x in summary.get('quality_coverage', [])}
    selected = []
    for (suite, method), rows in sorted(groups.items()):
        if method not in METHODS or suite not in ('real_comparison', 'synthetic_scaling') and 'budget' not in suite:
            continue
        valid = qgroups.get((suite, method), [])
        completed = [x for x in rows if x['all_repeats_completed']]
        valid_count = coverage.get((suite, method), {}).get('instances_with_valid_menu')
        selected.append({'suite': suite, 'method': LABELS[method], 'scheduled_instances': len(rows),
            'valid_replayed_menus': valid_count if valid_count is not None else len(valid),
            'profit_matches': sum(x.get('within_numerical_tolerance', False) for x in valid) if method != 'joint' else None,
            'median_relative_gap_percent': None if method == 'joint' else med(100*x['relative_gap'] for x in valid if x.get('relative_gap') is not None),
            'maximum_relative_gap_percent': max((100*x['relative_gap'] for x in valid if x.get('relative_gap') is not None), default=None),
            'all_repeats_completed_instances': len(completed),
            'median_completed_optimization_seconds': med(x['optimization_seconds'] for x in completed),
            'median_completed_total_seconds': med(x['total_seconds'] for x in completed),
            'median_completed_peak_rss_mib': med(x.get('peak_rss_mib') for x in completed),
            'timeout_repeats': sum(x.get('timeout_repeats', 0) for x in rows)})
    write_csv(out / 'table2_method_quality_efficiency.csv', selected)
    return {'coverage': populations, 'quality_efficiency': selected}


def main(root, summary_dir, out):
    summary = load_json(summary_dir / 'summary.json'); tables = load_json(summary_dir / 'tables.json')
    validate_summary_for_plotting(summary)
    timing = table_rows(summary_dir, tables, 'timing_by_instance')
    joints = table_rows(summary_dir, tables, 'joint_unique_instances')
    capacities = table_rows(summary_dir, tables, 'capacity_by_instance')
    cross = table_rows(summary_dir, tables, 'cross_mapping_by_instance')
    quality = table_rows(summary_dir, tables, 'quality_by_instance')
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9.5,
        'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
        'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .18, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    values = {'source_summary_sha256': hashlib.sha256((summary_dir/'summary.json').read_bytes()).hexdigest(),
              'timing': method_timing(timing, out), 'scaling': scalability(timing, out),
              'capacity': capacity_figure(capacities, joints, out),
              'association_mapping': association_mapping(joints, cross, out),
              'tables': export_tables(summary, tables, timing, quality, out)}
    (out/'figure_values.json').write_text(json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    captions = '''图1：同输入比较实例的运行时间与完成比例。左图先对每个实例的重复运行取中位数，再展示全部重复均在预算内完成的实例分布；优化与完整接口分别计时。右图标明全部重复完成的实例数及计划实例总数。MILP完成表示其数值模型完成求解，菜单净利润按统一买方选择规则另行核验。
图2：预定规模网格下的运行时间。每个坐标先对同实例的重复计时取中位数，再对随机种子与成本情景取中位数。实线展示全部重复完成实例的优化时间；叉号表示该坐标存在未完成实例。位于5秒预算线且无实线连接的叉号表示没有全部重复完成的实例，其位置只标识预算。各方法的完整任务分母、超时及数值诊断保留在配套数据表。
图3：在售产品数量限制对最优净利润的影响。左图为相对同输入无数量限制最优净利润的比例，先在时间窗内汇总，再按测点等权平均；透明带为以测点为重采样单元、固定种子进行2000次bootstrap得到的95%置信区间。右图统计达到无约束最优值所需的最小在售产品数量。各条曲线分别对应均衡、启用成本较高和订阅服务成本较高三类成本情景。
图4：完整时序池的许可与估值关联分析，以及比较样本上的质量映射评估。左图先在测点内跨月份平均负关联相对正关联的净利润变化，再展示测点间分布；中图为返回的最优菜单存在价格逆序的测点等权比例；右图保持设计菜单不变，在另一质量映射下重新计算买方选择及净利润，相对该环境最优值计算损失，最后按测点等权平均。横轴为评价映射，纵轴为菜单设计映射。Bal.、Act.、Svc.依次表示均衡、启用成本较高、订阅服务成本较高三类成本情景。
箱线图隐藏须外散点，完整分布、极值与逐实例记录均在复现结果中保留。图内英文标签便于后续英文稿复用，中文图注与本文术语保持一致。
'''
    (out/'图表说明.txt').write_text(captions, encoding='utf-8')
    print(json.dumps({'figures': 4, 'tables': 2, 'output': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--summary-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(); package = args.package.resolve()
    main(package, args.summary_dir or package/'results/final_summary', args.output_dir or package/'figures/final')
