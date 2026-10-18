"""Phase-separated exact PHT and complete-neighborhood benchmark interface.

Parse/validation, preprocessing plus optimization, recovery, and canonical replay
are disjoint phases. G0/G1/G2 evaluate fixed catalogs by value only, then execute
one recovery-enabled optimization for the winning catalog and recover it once.
That final DP is counted in optimization time (never in recovery time).

The cooperative budget covers all preprocessing/optimization. A timed-out G
search finishes the best completed catalog's recovery-enabled DP; this explicitly
reported finalization can exceed the soft budget. An external process hard limit
must enforce the experiment's common finalization allowance.
"""
from bisect import bisect_right
from fractions import Fraction
from itertools import combinations
import math
import time

from pht_experiment import validate_instance, _integer, _selection, _menu, _replay
from timed_solver.cores import BudgetExpired, optimize_joint, optimize_capacity

METHODS = ('joint', 'G0', 'G1', 'G2', 'fixed', 'capacity', 'contiguous')


class _Budget:
    def __init__(self, seconds):
        self.start = time.perf_counter()
        self.seconds = seconds
        self.deadline = None if seconds is None else self.start + seconds
    def check(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise BudgetExpired()


def _fixed(data, products, recover, checkpoint=None):
    products = tuple(sorted(products))
    fixed_cost = sum(data['activation'][u - 1] for u in products)
    restricted = [(bisect_right(products, cap), theta, n) for cap, theta, n in data['buyers']]
    state = optimize_joint([data['alpha'][u - 1] for u in products], [0] * len(products),
                           [data['service'][u - 1] for u in products], restricted,
                           recover=recover, checkpoint=checkpoint)
    value = state['optimal_profit'] - fixed_cost
    def restore():
        raw = state['recover']()
        prices = {products[u - 1]: p for u, p in raw['prices'].items()}
        high_price = 1 + max((theta for _, theta, _ in data['buyers']), default=0) * (
            data['alpha'][products[-1] - 1] if products else 0)
        for u in products:
            prices.setdefault(u, high_price)
        return {'selected': list(products), 'prices': prices,
                'active_subset': [products[u - 1] for u in raw['selected']]}
    return {**state, 'optimal_profit': value, 'recover': restore if recover else None}


class _ValueOracle:
    def __init__(self, data, budget):
        self.data, self.budget = data, budget
        self.cache = {}
        self.requests = self.hits = self.solves = 0
        self.operations = self.updates = 0
        self.incumbent_set, self.incumbent_value = (), Fraction(0)
    def __call__(self, products):
        self.budget.check()
        key = tuple(sorted(products))
        self.requests += 1
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.solves += 1
        state = _fixed(self.data, key, False, self.budget.check)
        self.operations += state['candidate_evaluations']
        self.updates += state['group_updates']
        if not state['complete']:
            raise BudgetExpired()
        value = state['optimal_profit']
        self.cache[key] = value
        if value > self.incumbent_value:
            self.incumbent_value, self.incumbent_set = value, key
        return value
    def stats(self):
        return {'requests': self.requests, 'hits': self.hits, 'solves': self.solves,
                'entries': len(self.cache), 'scope': 'single_benchmark_call',
                'candidate_evaluation': 'fixed_catalog_value_only'}


def _greedy(oracle, q, limit):
    current = ()
    value = oracle(current)
    while len(current) < limit:
        best, winner = value, current
        for u in range(1, q + 1):
            if u not in current:
                candidate_set = tuple(sorted((*current, u)))
                candidate = oracle(candidate_set)
                if candidate > best:
                    best, winner = candidate, candidate_set
        if winner == current:
            break
        current, value = winner, best
    return current, value


def _local(oracle, q, limit, current, radius):
    value = oracle(current)
    while True:
        complement = tuple(u for u in range(1, q + 1) if u not in current)
        neighborhood = set()
        for remove_count in range(min(radius, len(current)) + 1):
            for removed in combinations(current, remove_count):
                kept = set(current) - set(removed)
                for add_count in range(min(radius, len(complement), limit - len(kept)) + 1):
                    if remove_count == add_count == 0:
                        continue
                    for added in combinations(complement, add_count):
                        oracle.budget.check()
                        neighborhood.add(tuple(sorted(kept | set(added))))
        best, winner = value, current
        for selected in sorted(neighborhood):
            candidate = oracle(selected)
            if candidate > best:
                best, winner = candidate, selected
        if winner == current:
            return current, value
        current, value = winner, best


def benchmark(payload, method='joint', budget_seconds=5.0, capacity=None, fixed_products=None):
    started = time.perf_counter()
    data = validate_instance(payload)
    if method not in METHODS:
        raise ValueError(f'Unknown benchmark method {method!r}.')
    if budget_seconds is not None and (isinstance(budget_seconds, bool) or not isinstance(
            budget_seconds, (int, float)) or not math.isfinite(budget_seconds) or budget_seconds < 0):
        raise ValueError('budget_seconds must be finite and nonnegative, or None.')
    q = data['parameters']['Q']
    limit = q if capacity is None else _integer(capacity, 'capacity')
    if limit > q:
        raise ValueError('capacity must not exceed Q.')
    if method == 'joint' and limit != q:
        raise ValueError('Use capacity for a binding product-count limit.')
    if method != 'fixed' and fixed_products is not None:
        raise ValueError('fixed_products is used only by fixed.')
    if method == 'fixed':
        if fixed_products is None:
            raise ValueError('fixed_products is required for fixed.')
        fixed_products = _selection(fixed_products, q)
        if len(fixed_products) > limit:
            raise ValueError('The fixed catalog exceeds capacity.')
    parsed_at = time.perf_counter()
    budget = _Budget(budget_seconds)
    args = (data['alpha'], data['activation'], data['service'], data['buyers'])
    oracle = None
    finish_optimization_seconds = 0.0
    proof_status = 'exact'
    completed = True
    if method == 'joint':
        state = optimize_joint(*args, recover=True, checkpoint=budget.check)
        completed = state['complete']
    elif method == 'capacity':
        state = optimize_capacity(*args, L=limit, checkpoint=budget.check)
        completed = state['complete']
    elif method == 'fixed':
        state = _fixed(data, fixed_products, True, budget.check)
        completed = state['complete']
    else:
        oracle = _ValueOracle(data, budget)
        proof_status = 'heuristic' if method != 'contiguous' else 'exact_within_restricted_catalog_family'
        try:
            if method == 'contiguous':
                current, value = (), oracle(())
                for lo in range(1, q + 1):
                    for hi in range(lo, min(q, lo + limit - 1) + 1):
                        selected = tuple(range(lo, hi + 1))
                        candidate = oracle(selected)
                        if candidate > value:
                            current, value = selected, candidate
            else:
                current, value = _greedy(oracle, q, limit)
                if method in ('G1', 'G2'):
                    current, value = _local(oracle, q, limit, current, 1)
                if method == 'G2':
                    current, value = _local(oracle, q, limit, current, 2)
        except BudgetExpired:
            completed = False
            current, value = oracle.incumbent_set, oracle.incumbent_value
        finish_started = time.perf_counter()
        state = _fixed(data, current, True)
        finish_optimization_seconds = time.perf_counter() - finish_started
        if state['optimal_profit'] != value:
            raise AssertionError('Recovery-enabled final DP changed the winning catalog value.')
    optimized_at = time.perf_counter()
    optimization_seconds = optimized_at - parsed_at
    overrun = max(0.0, optimization_seconds - budget_seconds) if budget_seconds is not None else 0.0
    budget_exhausted = not completed or overrun > 0
    if budget_exhausted:
        status = 'timeout_with_incumbent'
        if not completed:
            proof_status = 'feasible_incumbent'
    else:
        status = 'heuristic' if method in ('G0', 'G1', 'G2') else 'exact'
    if method == 'capacity':
        raw, frontier_menus = state['recover']()
    else:
        raw, frontier_menus = state['recover'](), None
    recovered_at = time.perf_counter()
    replay = _replay(data, raw['selected'], raw['prices'])
    if Fraction(replay['accounting']['net_profit']) != state['optimal_profit']:
        raise AssertionError('Recovered menu and solver value disagree.')
    frontier = None
    if method == 'capacity':
        frontier = []
        for h, (value, menu) in enumerate(zip(state['frontier'], frontier_menus)):
            check = _replay(data, menu['selected'], menu['prices'])
            if len(menu['selected']) > h or Fraction(check['accounting']['net_profit']) != value:
                raise AssertionError('Capacity frontier recovery failed.')
            frontier.append({'capacity': h, 'net_profit': str(value),
                             'menu': _menu(menu['selected'], menu['prices']), **check})
    replayed_at = time.perf_counter()
    stats = {key: state[key] for key in ('candidate_evaluations', 'group_updates', 'completed_threshold_index')}
    if oracle:
        stats['candidate_evaluations'] += oracle.operations
        stats['group_updates'] += oracle.updates
    result = {'schema_version': 'pht-benchmark-v2', 'instance_id': data['instance_id'],
              'input_sha256': data['input_sha256'], 'method': method, 'status': status,
              'proof_status': proof_status, 'has_incumbent': True, 'completed_optimization': completed,
              'has_valid_menu': True, 'quality_comparison_eligible': True,
              'original_problem_certificate': {
                  'certified_optimal': proof_status == 'exact' and completed,
                  'scope': {'joint': 'joint_menu_and_prices', 'capacity': 'menu_with_capacity_limit',
                            'fixed': 'prices_for_prescribed_catalog',
                            'contiguous': 'contiguous_catalog_family'}.get(method, 'joint_menu_heuristic'),
                  'basis': 'completed exact-rational dynamic program' if proof_status == 'exact' and completed
                           else 'canonical replay certifies the returned menu value'},
              'parameters': data['parameters'], 'capacity_limit': limit,
              'menu': _menu(raw['selected'], raw['prices']), **replay,
              'net_profit': str(state['optimal_profit']),
              'phase_seconds': {'parse_validation': parsed_at - started,
                  'preprocess_optimization': optimization_seconds,
                  'recovery': recovered_at - optimized_at, 'replay': replayed_at - recovered_at,
                  'total': replayed_at - started},
              'budget_seconds': budget_seconds, 'budget_exhausted': budget_exhausted,
              'finish_optimization_seconds': finish_optimization_seconds,
              'budget_exceeded_overhead_seconds': overrun,
              'timing_scope': 'disjoint phases; preprocessing_optimization includes final recovery-enabled DP; recovery excludes all DP and replay',
              'cache_stats': oracle.stats() if oracle else {'requests': 0, 'hits': 0, 'solves': 0, 'entries': 0},
              'solver_stats': stats}
    if frontier is not None:
        result['frontier'] = frontier
    return result
