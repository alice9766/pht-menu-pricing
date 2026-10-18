#!/usr/bin/env python3
"""Extract a predetermined stored menu example; no solver or experiment calls.

Window: first selected window in the frozen final selection manifest, preserving
its stored order. Cost: balanced. Association order: positive/independent/negative.
Result: real_comparison, joint, repeat 1; latest recorded attempt for that job.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

ASSOCIATIONS = ('positive', 'independent', 'negative')
CN = {'positive': '正关联', 'independent': '独立', 'negative': '负关联'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def decimal(x, places=10):
    x = Fraction(x)
    with localcontext() as context:
        context.prec = 60
        return format(Decimal(x.numerator)/Decimal(x.denominator), f'.{places}f')


def independent_replay(payload, menu):
    """Only original-model utilities and explicit tie ordering; no PHT code."""
    products = {p['q']: p for p in payload['products']}
    offered = sorted(menu['products'])
    prices = {int(q): Fraction(p) for q, p in menu['prices'].items()}
    if set(prices) != set(offered) or any(p < 0 for p in prices.values()):
        raise ValueError('Invalid menu prices')
    choices, counts = [], Counter()
    revenue = service_cost = Fraction(0)
    group_counts = defaultdict(Counter)
    group_population = Counter()
    for buyer in payload['buyers']:
        cap, theta, count = buyer['access_cap'], Fraction(buyer['theta']), buyer['count']
        options = [(Fraction(0), 0)]
        evaluated = []
        for q in offered:
            if q <= cap:
                utility = theta*Fraction(products[q]['alpha']) - prices[q]
                options.append((utility, q))
                evaluated.append({'q': q, 'utility_exact': str(utility), 'utility_decimal': decimal(utility)})
        utility, q = max(options)  # zero-utility purchase; highest-quality tie.
        counts[q] += count
        group_counts[cap][q] += count
        group_population[cap] += count
        if q:
            revenue += count*prices[q]
            service_cost += count*Fraction(products[q]['unit_service_cost'])
        choices.append({'type_id': buyer['type_id'], 'access_cap': cap,
                        'theta_exact': str(theta), 'count': count, 'product': q,
                        'utility_exact': str(utility), 'utility_decimal': decimal(utility),
                        'evaluated_product_utilities': evaluated})
    activation_cost = sum((Fraction(products[q]['activation_cost']) for q in offered), Fraction(0))
    accounting = {'revenue': revenue, 'service_cost': service_cost,
                  'activation_cost': activation_cost,
                  'net_profit': revenue-service_cost-activation_cost}
    groups = [{'access_cap': cap, 'population': group_population[cap],
               'subscription_counts': {str(q): n for q, n in sorted(group_counts[cap].items())},
               'subscribed_buyers': group_population[cap]-group_counts[cap][0],
               'exit_buyers': group_counts[cap][0]} for cap in sorted(group_counts)]
    return choices, counts, groups, accounting


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract(base, output):
    base, output = Path(base).resolve(), Path(output).resolve()
    manifest_path = base/'data/final/selection_manifest.json'
    input_path = base/'data/final/comparison_instances.jsonl'
    result_path = base/'results/final_isolated/results.jsonl'
    manifest = json.loads(manifest_path.read_text())
    first_month = manifest['per_month'][0]
    first_selected = first_month['selected'][0]
    meter, month = first_selected['client'], first_month['month']
    ids = {association: f'uci321-{meter}-{month}-{association}-balanced' for association in ASSOCIATIONS}
    inputs, input_lines, results, result_lines = {}, {}, {}, {}
    with input_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row['instance_id'] in ids.values():
                inputs[row['instance_id']], input_lines[row['instance_id']] = row, line_number
    wanted_jobs = {f'real_comparison:{iid}:joint:rep1' for iid in ids.values()}
    with result_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row['job_id'] in wanted_jobs:
                results[row['instance_id']], result_lines[row['instance_id']] = row, line_number
    if len(inputs) != 3 or len(results) != 3:
        raise ValueError('The three prespecified stored inputs/results are incomplete; do not substitute another window')
    cases, product_rows, buyer_rows, group_rows = [], [], [], []
    for association in ASSOCIATIONS:
        iid = ids[association]; payload = inputs[iid]; record = results[iid]; result = record['result']
        if record['runner_status'] != 'completed' or result['status'] != 'exact':
            raise ValueError('Prespecified result is not completed exact DP output')
        if canonical_sha(payload) != record['input_sha256']:
            raise ValueError('Stored result input hash mismatch')
        menu = result['menu']
        choices, counts, groups, accounting = independent_replay(payload, menu)
        for name, value in accounting.items():
            if value != Fraction(result['accounting'][name]):
                raise ValueError('Independent accounting differs from stored result')
        stored_choices = {x['type_id']: x for x in result['buyer_choices']}
        for choice in choices:
            stored = stored_choices[choice['type_id']]
            if (choice['product'], choice['count'], Fraction(choice['utility_exact'])) != (
                    stored['product'], stored['count'], Fraction(stored['utility'])):
                raise ValueError('Independent utility choice differs from stored result')
            buyer_rows.append({'association': association, 'association_cn': CN[association],
                               **{k:v for k,v in choice.items() if k != 'evaluated_product_utilities'}})
        case_products = []
        delays = payload['metadata']['delay_minutes_in_quality_order']
        for product, delay in zip(payload['products'], delays):
            q = product['q']; price = menu['prices'].get(str(q))
            row = {'association': association, 'association_cn': CN[association], 'product': q,
                   'delay_minutes': delay,
                   'alpha_exact': str(Fraction(product['alpha'])), 'alpha_decimal': decimal(product['alpha']),
                   'activation_cost_exact': str(Fraction(product['activation_cost'])),
                   'activation_cost_decimal': decimal(product['activation_cost']),
                   'unit_service_cost_exact': str(Fraction(product['unit_service_cost'])),
                   'unit_service_cost_decimal': decimal(product['unit_service_cost']),
                   'offered': q in menu['products'], 'price_exact': price if price is not None else '',
                   'price_decimal': decimal(price) if price is not None else '',
                   'subscribed_buyers': counts[q]}
            product_rows.append(row); case_products.append(row)
        for group in groups:
            for q, count in group['subscription_counts'].items():
                group_rows.append({'association': association, 'association_cn': CN[association],
                                   'access_cap': group['access_cap'], 'group_population': group['population'],
                                   'product': int(q), 'buyer_count': count})
        ordered_prices = [(q, Fraction(menu['prices'][str(q)])) for q in sorted(menu['products'])]
        inversions = [{'low_product': q, 'high_product': r, 'low_price_exact': str(p),
                       'high_price_exact': str(s), 'price_drop_exact': str(p-s)}
                      for (q,p),(r,s) in zip(ordered_prices,ordered_prices[1:]) if p > s]
        cases.append({'association': association, 'association_cn': CN[association], 'instance_id': iid,
                      'input_line_number': input_lines[iid], 'result_line_number': result_lines[iid],
                      'job_id': record['job_id'], 'attempt': record['attempt'],
                      'result_input_sha256': record['input_sha256'], 'job_signature': record['job_signature'],
                      'certificate': result['original_problem_certificate'], 'menu': menu,
                      'products': case_products, 'buyer_choices': choices, 'permission_group_choices': groups,
                      'population': sum(x['count'] for x in payload['buyers']),
                      'subscribed_buyers': sum(n for q,n in counts.items() if q), 'exit_buyers': counts[0],
                      'accounting_exact': {k: str(v) for k,v in accounting.items()},
                      'accounting_decimal': {k: decimal(v) for k,v in accounting.items()},
                      'price_inversions': inversions, 'independent_replay_matches_stored_result': True})
    evidence = {'selection_rule': 'Frozen final selection_manifest.json: first per_month entry, first selected entry; balanced cost; positive/independent/negative in that order; real_comparison joint repeat 1 latest stored attempt. No selection by profit or price pattern.',
                'selection_independent_of_outcomes': True, 'new_experiments_run': False,
                'new_optimizations_run': False, 'independent_utility_replay_performed': True,
                'source_meter': meter, 'evaluation_month': month,
                'reference_month': next(iter(inputs.values()))['metadata']['reference_month'],
                'selection_hash': first_selected['selection_hash'], 'cost_profile': 'balanced',
                'currency_unit': 'one_declared_reference_value_unit',
                'data_interpretation': 'Real electricity traces supply quality parameters; buyer types/counts/permissions and costs are prescribed subscription scenarios, not observed transactions.',
                'formal_protocol_sha256': manifest['protocol_sha256'],
                'sources': [{'path': str(p), 'relative_to_formal_package': str(p.relative_to(base)),
                             'sha256': sha(p)} for p in (manifest_path,input_path,result_path)],
                'cases': cases}
    output.mkdir(parents=True, exist_ok=True)
    (output/'menu_example.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    write_csv(output/'menu_example.csv', product_rows)
    write_csv(output/'menu_example_buyers.csv', buyer_rows)
    write_csv(output/'menu_example_permission_groups.csv', group_rows)
    paragraphs = ['固定订阅菜单实例：来源与逐项回放', '',
                  f'选择规则：冻结96窗口清单原顺序第一项，测点{meter}，评价月份{month}，参考月份{evidence["reference_month"]}。固定balanced成本，按正关联、独立、负关联顺序展示。',
                  '数据来自已存real_comparison/joint/rep1结果；未新增实验，未重跑求解。采用精确有理数独立计算效用，验证购买选择、收入、单位服务成本、固定成本及净利润。',
                  '基准价值来自公开用电时序；买方和成本属于预设订阅情景。全部价格与净利润以统一参考价值单位表示。', '',
                  '候选产品参数：产品号 / 时延分钟 / 基准价值 / 启用固定成本 / 单位订阅服务成本']
    for p in cases[0]['products']:
        paragraphs.append(f'{p["product"]} / {p["delay_minutes"]} / {p["alpha_decimal"]} / {p["activation_cost_decimal"]} / {p["unit_service_cost_decimal"]}')
    for case in cases:
        paragraphs.extend(['', f'{case["association_cn"]}情景',
                           f'在售集合：{case["menu"]["products"]}；订阅人数{case["subscribed_buyers"]}，退出人数{case["exit_buyers"]}；净利润{case["accounting_decimal"]["net_profit"]}。',
                           '在售产品 / 公开价格（精确值） / 公开价格（十位小数） / 订阅人数'])
        for p in case['products']:
            if p['offered']:
                paragraphs.append(f'{p["product"]} / {p["price_exact"]} / {p["price_decimal"]} / {p["subscribed_buyers"]}')
        paragraphs.append('访问许可上限 / 估值系数 / 类型人数 / 选择产品（0表示退出） / 效用')
        for buyer in case['buyer_choices']:
            paragraphs.append(f'{buyer["access_cap"]} / {buyer["theta_exact"]} / {buyer["count"]} / {buyer["product"]} / {buyer["utility_decimal"]}')
        if case['price_inversions']:
            paragraphs.append('本次返回最优菜单存在质量更高而标价更低的相邻在售产品，具体产品对见JSON；该记录展示返回菜单，不声称所有最优菜单必然如此。')
    paragraphs.extend(['', '完整来源、输入散列、结果行号和精确值见menu_example.json；全部候选产品参数见menu_example.csv；逐类型选择见menu_example_buyers.csv；许可群体汇总见menu_example_permission_groups.csv。'])
    (output/'menu_example.txt').write_text('\n'.join(paragraphs)+'\n')
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = extract(args.base, args.output)
    print(json.dumps({'source_meter': evidence['source_meter'], 'evaluation_month': evidence['evaluation_month'],
                      'cases': [{'association': x['association'], 'menu': x['menu'],
                                 'subscribed_buyers': x['subscribed_buyers'],
                                 'net_profit': x['accounting_decimal']['net_profit'],
                                 'inversions': x['price_inversions']} for x in evidence['cases']]},
                     ensure_ascii=False, indent=2))
