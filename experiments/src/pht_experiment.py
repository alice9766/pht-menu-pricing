"""Validated, exact experiment interface for the frozen Section 5 algorithms.

Input schema: pht-experiment-v1. Monetary and valuation fields accept Python
integers or exact rational strings (integer, fraction, or finite decimal).
JSON numeric floats and booleans are rejected in model fields. Every returned
menu is independently replayed against the original, uncompressed buyers.

API: solve_instance(payload, method='joint', capacity=None, fixed_products=None)
CLI: python src/pht_experiment.py --input instance.json --method joint --output result.json
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any

try:
    from .solver_snapshot.pht_grouped_exact import solve_grouped
    from .solver_snapshot.pht_capacity import solve_capacity
    from .solver_snapshot.pht_repricing import solve_fixed
except ImportError:
    from solver_snapshot.pht_grouped_exact import solve_grouped
    from solver_snapshot.pht_capacity import solve_capacity
    from solver_snapshot.pht_repricing import solve_fixed


SCHEMA_VERSION = 'pht-experiment-v1'
RESULT_SCHEMA_VERSION = 'pht-result-v1'
METHODS = ('joint', 'capacity', 'fixed', 'G0', 'G1', 'G2', 'contiguous')
_RATIONAL = re.compile(r'[+-]?(?:\d+(?:/\d+)?|\d+\.\d+|\.\d+)\Z')


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}, not a bool or float.')
    return value


def _rational(value: Any, name: str) -> Fraction:
    if type(value) is int:
        return Fraction(value)
    if type(value) is str and _RATIONAL.fullmatch(value):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError):
            pass
    raise ValueError(f'{name} must be an integer or an exact rational string.')


def _required(obj: dict, names: tuple[str, ...], location: str) -> None:
    missing = set(names) - obj.keys()
    if missing:
        raise ValueError(f'{location} is missing {sorted(missing)}.')


def validate_instance(payload: dict) -> dict:
    """Validate and return a parsed internal representation with Fraction values."""
    if type(payload) is not dict:
        raise ValueError('The instance must be a JSON object.')
    _required(payload, ('schema_version', 'instance_id', 'products', 'buyers', 'metadata'), 'instance')
    if payload['schema_version'] != SCHEMA_VERSION:
        raise ValueError(f'schema_version must be {SCHEMA_VERSION!r}.')
    if type(payload['instance_id']) is not str or not payload['instance_id'].strip():
        raise ValueError('instance_id must be a nonempty string.')
    if type(payload['metadata']) is not dict:
        raise ValueError('metadata must be an object.')
    products = payload['products']
    buyers = payload['buyers']
    if type(products) is not list or not products:
        raise ValueError('products must be a nonempty list, with q = 1,...,Q.')
    if type(buyers) is not list:
        raise ValueError('buyers must be a list.')
    alpha, activation, service = [], [], []
    for q, product in enumerate(products, 1):
        if type(product) is not dict:
            raise ValueError(f'products[{q - 1}] must be an object.')
        _required(product, ('q', 'alpha', 'activation_cost', 'unit_service_cost'), f'product {q}')
        if _integer(product['q'], f'product {q} q', 1) != q:
            raise ValueError('Products must be ordered and numbered consecutively from 1.')
        a = _rational(product['alpha'], f'product {q} alpha')
        c = _rational(product['activation_cost'], f'product {q} activation_cost')
        s = _rational(product['unit_service_cost'], f'product {q} unit_service_cost')
        if a <= 0 or (alpha and a <= alpha[-1]):
            raise ValueError('alpha must be positive and strictly increasing.')
        if c < 0 or s < 0:
            raise ValueError('Both cost fields must be nonnegative.')
        alpha.append(a)
        activation.append(c)
        service.append(s)
    clean, type_ids, seen_pairs, seen_ids = [], [], set(), set()
    for i, buyer in enumerate(buyers):
        if type(buyer) is not dict:
            raise ValueError(f'buyers[{i}] must be an object.')
        _required(buyer, ('type_id', 'access_cap', 'theta', 'count'), f'buyer {i}')
        type_id = buyer['type_id']
        if type(type_id) is not str or not type_id.strip() or type_id in seen_ids:
            raise ValueError('Buyer type_id values must be nonempty unique strings.')
        cap = _integer(buyer['access_cap'], f'buyer {type_id} access_cap')
        count = _integer(buyer['count'], f'buyer {type_id} count', 1)
        theta = _rational(buyer['theta'], f'buyer {type_id} theta')
        if cap > len(products) or theta <= 0:
            raise ValueError('Access caps must be in 0,...,Q and theta must be positive.')
        if (cap, theta) in seen_pairs:
            raise ValueError('Duplicate (access_cap, theta) types must be aggregated before solving.')
        seen_pairs.add((cap, theta))
        seen_ids.add(type_id)
        clean.append((cap, theta, count))
        type_ids.append(type_id)
    # The JSON fingerprint also retains data provenance and scenario settings.
    try:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ValueError('Input must be JSON serializable with finite metadata values.') from exc
    return {'alpha': alpha, 'activation': activation, 'service': service,
            'buyers': clean, 'type_ids': type_ids,
            'instance_id': payload['instance_id'], 'input_sha256': sha256(encoded).hexdigest(),
            'parameters': {'Q': len(products), 'K': len(clean),
                           'B': len({cap for cap, _, _ in clean if cap > 0}),
                           'T': len({Fraction(0), *(t for _, t, _ in clean)})}}


def load_instance(path: str | Path) -> dict:
    """Read and validate an input JSON, returning its original JSON payload."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_instance(payload)
    return payload


def _selection(selected: Any, q: int) -> list[int]:
    if not isinstance(selected, (list, tuple)):
        raise ValueError('Product selection must be a list or tuple of indices.')
    out = [_integer(u, 'selected product', 1) for u in selected]
    if len(set(out)) != len(out) or any(u > q for u in out):
        raise ValueError('Selected indices must be unique and within 1,...,Q.')
    return sorted(out)


def _replay(data: dict, selected: list[int], prices: dict) -> dict:
    """Independent canonical replay; does not call snapshot evaluate_menu."""
    selected = _selection(selected, len(data['alpha']))
    prices = {int(k): Fraction(v) for k, v in prices.items()}
    if set(prices) != set(selected) or any(p < 0 for p in prices.values()):
        raise ValueError('Each offered product must have exactly one nonnegative price.')
    counts = {q: 0 for q in selected}
    choices = []
    revenue = Fraction(0)
    unit_cost = Fraction(0)
    for type_id, (cap, theta, count) in zip(data['type_ids'], data['buyers']):
        # Including exit (utility=0,index=0) makes tie-breaking explicit:
        # maximize utility first, then the product index (= quality order).
        options = [(Fraction(0), 0)]
        for q in selected:
            if q <= cap:
                options.append((theta * data['alpha'][q - 1] - prices[q], q))
        utility, chosen = max(options)
        choices.append({'type_id': type_id, 'product': chosen,
                        'count': count, 'utility': str(utility)})
        if chosen:
            counts[chosen] += count
            revenue += count * prices[chosen]
            unit_cost += count * data['service'][chosen - 1]
    activation = sum((data['activation'][q - 1] for q in selected), Fraction(0))
    return {'buyer_choices': choices,
            'subscription_counts': {str(q): counts[q] for q in selected},
            'accounting': {'revenue': str(revenue), 'activation_cost': str(activation),
                           'service_cost': str(unit_cost),
                           'net_profit': str(revenue - unit_cost - activation)},
            'replay_verified': True}


def replay_menu(payload: dict, selected: list[int], prices: dict) -> dict:
    """Replay any menu under the original instance and frozen buyer rules.

    Public price values follow the input exact-number grammar; product keys may
    be integer indices or their JSON string representation.
    """
    data = validate_instance(payload)
    if type(prices) is not dict:
        raise ValueError('prices must be an object.')
    clean_prices = {}
    for key, value in prices.items():
        if type(key) is int:
            q = key
        elif type(key) is str and key.isdigit():
            q = int(key)
        else:
            raise ValueError('Price keys must be product indices.')
        if q in clean_prices:
            raise ValueError('Price keys must identify distinct products.')
        clean_prices[q] = _rational(value, f'price {q}')
    return _replay(data, selected, clean_prices)


def _menu(selected: list[int], prices: dict) -> dict:
    return {'products': sorted(selected),
            'prices': {str(q): str(prices[q]) for q in sorted(selected)}}


class _FixedOracle:
    """One invocation's fixed-catalog oracle; no cache shared across methods."""
    def __init__(self, data: dict):
        self.data = data
        self.cache = {}
        self.requests = 0
        self.hits = 0

    def __call__(self, selected) -> dict:
        key = tuple(sorted(selected))
        self.requests += 1
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        d = self.data
        result = solve_fixed(d['alpha'], d['activation'], d['service'], d['buyers'], key)
        replay = _replay(d, result['selected'], result['prices'])
        if Fraction(replay['accounting']['net_profit']) != result['optimal_profit']:
            raise AssertionError('Fixed oracle and independent replay disagree.')
        result['replay'] = replay
        self.cache[key] = result
        return result

    def stats(self):
        return {'requests': self.requests, 'hits': self.hits,
                'solves': len(self.cache), 'entries': len(self.cache),
                'scope': 'single_solve_instance_call'}


def _greedy(oracle: _FixedOracle, q: int, limit: int) -> tuple[tuple[int, ...], dict]:
    current = ()
    result = oracle(current)
    while len(current) < limit:
        best, best_set = result, current
        for u in range(1, q + 1):
            if u in current:
                continue
            candidate_set = tuple(sorted((*current, u)))
            candidate = oracle(candidate_set)
            if candidate['optimal_profit'] > best['optimal_profit']:
                best, best_set = candidate, candidate_set
        if best_set == current:
            break
        current, result = best_set, best
    return current, result


def _local(oracle: _FixedOracle, q: int, limit: int, current: tuple,
           radius: int) -> tuple[tuple[int, ...], dict]:
    result = oracle(current)
    while True:
        complement = tuple(u for u in range(1, q + 1) if u not in current)
        # Complete <=radius deletions and <=radius insertions; pure deletion
        # and pure insertion are included. The current catalog itself is skipped.
        neighborhood = set()
        for remove_count in range(min(radius, len(current)) + 1):
            for removed in combinations(current, remove_count):
                kept = set(current) - set(removed)
                for add_count in range(min(radius, len(complement), limit - len(kept)) + 1):
                    if remove_count == add_count == 0:
                        continue
                    for added in combinations(complement, add_count):
                        neighborhood.add(tuple(sorted(kept | set(added))))
        best, best_set = result, current
        for candidate_set in sorted(neighborhood):
            candidate = oracle(candidate_set)
            if candidate['optimal_profit'] > best['optimal_profit']:
                best, best_set = candidate, candidate_set
        if best_set == current:
            return current, result
        current, result = best_set, best


def solve_instance(payload: dict, method: str = 'joint', capacity: int | None = None,
                   fixed_products: list[int] | None = None) -> dict:
    """Solve one validated instance and return JSON-serializable exact results.

    joint: global optimum without an additional catalog limit.
    capacity: exact optima for every upper bound h=0,...,capacity (default Q).
    fixed: exact repricing of precisely fixed_products, including all their costs.
    G0/G1/G2: addition and complete exchange neighborhoods with exact repricing.
    contiguous: exact best consecutive catalog, a restricted structural baseline.
    """
    data = validate_instance(payload)
    if method not in METHODS:
        raise ValueError(f'Unknown method {method!r}; choose from {METHODS}.')
    q = data['parameters']['Q']
    limit = q if capacity is None else _integer(capacity, 'capacity')
    if limit > q:
        raise ValueError('capacity must be in 0,...,Q.')
    if method == 'joint' and limit != q:
        raise ValueError('Use method=capacity for a binding product-count limit.')
    if method != 'fixed' and fixed_products is not None:
        raise ValueError('fixed_products is used only by method=fixed.')
    args = (data['alpha'], data['activation'], data['service'], data['buyers'])
    oracle = None
    scope = 'joint_unrestricted'
    proof_status = 'exact'
    if method == 'joint':
        raw = solve_grouped(*args, recover=True)
    elif method == 'capacity':
        raw = solve_capacity(*args, L=limit, recover=True, recover_all=True)
        scope = 'joint_at_most_capacity'
    elif method == 'fixed':
        if fixed_products is None:
            raise ValueError('method=fixed requires fixed_products (possibly an empty list).')
        chosen = _selection(fixed_products, q)
        if len(chosen) > limit:
            raise ValueError('The fixed catalog exceeds the product-count limit.')
        oracle = _FixedOracle(data)
        raw = oracle(chosen)
        scope = 'fixed_catalog'
    else:
        oracle = _FixedOracle(data)
        if method == 'contiguous':
            raw = oracle(())
            for lo in range(1, q + 1):
                for hi in range(lo, min(q, lo + limit - 1) + 1):
                    candidate = oracle(tuple(range(lo, hi + 1)))
                    if candidate['optimal_profit'] > raw['optimal_profit']:
                        raw = candidate
            scope = 'contiguous_catalog_at_most_capacity'
            proof_status = 'exact_within_restricted_catalog_family'
        else:
            current, raw = _greedy(oracle, q, limit)
            if method in ('G1', 'G2'):
                current, raw = _local(oracle, q, limit, current, 1)
            if method == 'G2':
                current, raw = _local(oracle, q, limit, current, 2)
            scope = 'joint_at_most_capacity'
            proof_status = 'heuristic'
    replay = _replay(data, raw['selected'], raw['prices'])
    if Fraction(replay['accounting']['net_profit']) != raw['optimal_profit']:
        raise AssertionError('Solver objective and independent replay disagree.')
    result = {'schema_version': RESULT_SCHEMA_VERSION, 'instance_id': data['instance_id'],
              'input_sha256': data['input_sha256'], 'method': method,
              'proof_status': proof_status, 'optimization_scope': scope,
              'parameters': data['parameters'], 'capacity_limit': limit,
              'menu': _menu(raw['selected'], raw['prices']), **replay,
              'cache_stats': oracle.stats() if oracle else
                  {'requests': 0, 'hits': 0, 'solves': 0, 'entries': 0,
                   'scope': 'single_solve_instance_call'},
              'solver_stats': {key: raw[key] for key in
                  ('candidate_evaluations', 'group_updates', 'recovery_record_count')
                  if key in raw}}
    if method == 'capacity':
        result['frontier'] = []
        for h, (value, menu) in enumerate(zip(raw['frontier'], raw['menus'])):
            check = _replay(data, menu['selected'], menu['prices'])
            if len(menu['selected']) > h or Fraction(check['accounting']['net_profit']) != value:
                raise AssertionError('Capacity frontier replay failed.')
            result['frontier'].append({'capacity': h, 'net_profit': str(value),
                                       'menu': _menu(menu['selected'], menu['prices']), **check})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--method', choices=METHODS, default='joint')
    parser.add_argument('--capacity', type=int)
    parser.add_argument('--fixed-products', help='Comma-separated product indices; empty string = empty catalog')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}; choose a new path.')
    fixed = None if args.fixed_products is None else [int(x) for x in args.fixed_products.split(',') if x]
    payload = load_instance(args.input)
    result = solve_instance(payload, args.method, args.capacity, fixed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
