"""Independent SciPy/HiGHS weak-IC MILP comparator, with exact menu replay.

The compact utility-variable formulation adapts the public artifact's
milp_reference/reference_milp.py. It does not call any PHT optimizer. The MIP
allows arbitrary utility-maximizing assignments at ties; the returned menu is
therefore evaluated separately using the paper's exact highest-quality rule.
A solver incumbent objective is not an upper bound. Only the negative of a
finite minimization dual bound is reported as a numerical profit upper bound.

solve_milp_instance(payload, time_limit=5.0, capacity=None, threads=1)
The time_limit covers model construction plus the solver's optimization call.
The caller should also enforce an external process wall limit.
"""
from __future__ import annotations

from fractions import Fraction
import math
import time
import warnings

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from pht_experiment import validate_instance, _replay


def _finite(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _float(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Model coefficients must fit finite float64 values.')
    return value


def _build_model(data, limit):
    """Continuous prices, activation/choice binaries, and payment envelopes."""
    alpha = [_float(x) for x in data['alpha']]
    activation = [_float(x) for x in data['activation']]
    service = [_float(x) for x in data['service']]
    buyers = data['buyers']
    Q, K = len(alpha), len(buyers)
    # Only a demanded product needs to remain offered in this joint problem.
    # Its price cannot exceed an eligible buyer valuation. The added one also
    # makes a strict nonpurchase price available for every zero-demand product.
    exact_values = [[theta * data['alpha'][q] for q in range(cap)]
                    for cap, theta, _ in buyers]
    U_exact = 1 + max((v for row in exact_values for v in row), default=Fraction(0))
    U = _float(U_exact)
    values = [[_float(v) for v in row] for row in exact_values]
    V = [max(row, default=0.0) for row in values]
    lower, upper, integer, objective = [], [], [], []

    def variable(hi, integral=False, cost=0.0):
        j = len(lower)
        lower.append(0.0); upper.append(hi)
        integer.append(int(integral)); objective.append(_float(cost))
        return j

    y = [variable(1.0, True, activation[q]) for q in range(Q)]
    p = [variable(U) for _ in range(Q)]
    utility = [variable(V[k]) for k in range(K)]
    outside = [variable(1.0, True) for _ in range(K)]
    x, payment = {}, {}
    for k, (cap, _, count) in enumerate(buyers):
        for q in range(cap):
            x[k, q] = variable(1.0, True, count * service[q])
            payment[k, q] = variable(U, cost=-count)
    row_idx, col_idx, coefficients, row_lo, row_hi = [], [], [], [], []

    def constraint(terms, lo=-np.inf, hi=np.inf):
        r = len(row_lo)
        for j, v in terms.items():
            if v:
                row_idx.append(r); col_idx.append(j); coefficients.append(_float(v))
        row_lo.append(lo); row_hi.append(hi)

    constraint({j: 1.0 for j in y}, hi=limit)
    for q in range(Q):
        constraint({p[q]: 1.0, y[q]: -U}, hi=0.0)
    for k, (cap, _, _) in enumerate(buyers):
        constraint({outside[k]: 1.0, **{x[k, q]: 1.0 for q in range(cap)}}, lo=1.0, hi=1.0)
        constraint({utility[k]: 1.0, outside[k]: V[k]}, hi=V[k])
        H = U + V[k]
        for q in range(cap):
            value = values[k][q]
            constraint({x[k, q]: 1.0, y[q]: -1.0}, hi=0.0)
            # All offered products lower-bound the maximal utility. The chosen
            # product also upper-bounds it. Choosing outside fixes utility zero.
            constraint({utility[k]: 1.0, p[q]: 1.0, y[q]: -H}, lo=value-H)
            constraint({utility[k]: 1.0, p[q]: 1.0, x[k, q]: H}, hi=value+H)
            constraint({payment[k, q]: 1.0, x[k, q]: -U}, hi=0.0)
            constraint({payment[k, q]: 1.0, p[q]: -1.0}, hi=0.0)
            constraint({payment[k, q]: 1.0, p[q]: -1.0, x[k, q]: -U}, lo=-U)
    matrix = coo_matrix((coefficients, (row_idx, col_idx)),
                        shape=(len(row_lo), len(lower))).tocsc()
    return {'objective': np.asarray(objective), 'integer': np.asarray(integer),
            'lower': np.asarray(lower), 'upper': np.asarray(upper),
            'row_lo': np.asarray(row_lo), 'row_hi': np.asarray(row_hi), 'matrix': matrix,
            'y': y, 'p': p, 'outside': outside, 'x': x,
            'utility': utility, 'payment': payment,
            'U_exact': str(U_exact), 'U': U,
            'dimensions': {'variables': len(lower), 'binary_variables': sum(integer),
                           'constraints': len(row_lo), 'eligible_pairs': len(x)}}


def _weak_choice_check(data, selected, prices, choices):
    """Exact feasibility of one weak-IC assignment at the stated menu."""
    offered = set(selected)
    errors = []
    for k, ((cap, theta, _), chosen) in enumerate(zip(data['buyers'], choices)):
        alternatives = {0: Fraction(0)}
        alternatives.update({q: theta*data['alpha'][q-1]-prices[q] for q in selected if q <= cap})
        if chosen not in alternatives:
            errors.append({'type_index': k, 'reason': 'unavailable_assigned_product'})
        elif alternatives[chosen] < max(alternatives.values()):
            errors.append({'type_index': k, 'reason': 'assigned_product_not_weak_utility_maximizer'})
    if any(p < 0 for p in prices.values()):
        errors.append({'reason': 'negative_price'})
    return not errors, errors


def _restore_rational_prices(data, selected, prices, choices, scale):
    """Restore near-binding price equalities without an optimization oracle.

    Edges encode p_i-p_j=theta*(alpha_i-alpha_j), with node 0 anchored
    at zero. A weighted union-find keeps a consistent spanning forest.
    Nearness only proposes equalities; every recovered weak-IC inequality
    and every nonnegative price is checked in exact arithmetic afterward.
    """
    raw = {q: Fraction(v) for q, v in prices.items()}
    tolerance = Fraction(str(1e-8*max(1.0, scale, *(abs(float(p)) for p in raw.values()))))
    candidates = []
    def propose(i, j, delta, kind, buyer=-1):
        residual = abs(raw.get(i, Fraction(0))-raw.get(j, Fraction(0))-delta)
        if residual <= tolerance:
            candidates.append((residual, kind, buyer, i, j, delta))
    for q in selected:
        propose(q, 0, Fraction(0), 'zero_price')
    for k, ((cap, theta, _), chosen) in enumerate(zip(data['buyers'], choices)):
        if chosen:
            if chosen not in raw:
                continue
            propose(chosen, 0, theta*data['alpha'][chosen-1], 'participation', k)
            for other in selected:
                if other <= cap and other != chosen:
                    propose(chosen, other, theta*(data['alpha'][chosen-1]-data['alpha'][other-1]),
                            'indifference', k)
        else:
            for other in selected:
                if other <= cap:
                    propose(other, 0, theta*data['alpha'][other-1], 'exit_boundary', k)
    candidates.sort()
    parent = {q:q for q in [0,*selected]}
    weight = {q:Fraction(0) for q in parent}  # p_q-p_parent[q]
    def find(q):
        if parent[q] != q:
            previous = parent[q]
            root, delta = find(previous)
            weight[q] += delta
            parent[q] = root
        return parent[q], weight[q]
    accepted, conflicts = [], []
    for residual, kind, buyer, i, j, delta in candidates:
        ri, wi = find(i); rj, wj = find(j)
        description = {'i':i,'j':j,'difference':str(delta),'kind':kind,
                       'type_index':buyer,'raw_residual':str(residual)}
        if ri == rj:
            if wi-wj != delta:
                conflicts.append(description)
            continue
        # Keep the lower-numbered representative; node zero stays anchored.
        if ri < rj:
            parent[rj] = ri
            weight[rj] = -delta+wi-wj
        else:
            parent[ri] = rj
            weight[ri] = delta-wi+wj
        accepted.append(description)
    recovered = {}
    for q in selected:
        root, offset = find(q)
        recovered[q] = (Fraction(0) if root == 0 else raw[root])+offset
    max_change = max((abs(recovered[q]-raw[q]) for q in selected), default=Fraction(0))
    weak_valid, errors = _weak_choice_check(data, selected, recovered, choices)
    within = max_change <= tolerance
    raw_valid, raw_errors = _weak_choice_check(data, selected, raw, choices)
    if weak_valid and within:
        chosen_prices, mode = recovered, 'near_binding_exact_difference_forest'
    elif raw_valid:
        chosen_prices, mode = raw, 'raw_exact_decimal_already_weak_ic_feasible'
    else:
        chosen_prices, mode = raw, 'numeric_recovery_issue'
    return chosen_prices, {'method': mode, 'candidate_equalities': len(candidates),
                          'accepted_equalities': accepted, 'conflicting_equalities': conflicts,
                          'neighborhood_tolerance': str(tolerance), 'max_absolute_price_change': str(max_change),
                          'candidate_within_neighborhood': within, 'candidate_exact_weak_ic_valid': weak_valid,
                          'candidate_errors': errors, 'raw_exact_weak_ic_valid': raw_valid,
                          'raw_errors': raw_errors,
                          'numeric_recovery_issue': mode == 'numeric_recovery_issue'}, raw


def _tolerance_replay_diagnostic(data, selected, prices, tolerance):
    """Diagnostic only: tolerance is never substituted for the paper's rule."""
    revenue = service = Fraction(0)
    choices = []
    for cap, theta, count in data['buyers']:
        utilities = {0: Fraction(0)}
        utilities.update({q:theta*data['alpha'][q-1]-prices[q] for q in selected if q <= cap})
        maximum = max(utilities.values())
        chosen = max(q for q, u in utilities.items() if u >= maximum-tolerance)
        choices.append(chosen)
        if chosen:
            revenue += count*prices[chosen]
            service += count*data['service'][chosen-1]
    activation = sum((data['activation'][q-1] for q in selected), Fraction(0))
    return {'choices': choices, 'net_profit': str(revenue-service-activation),
            'utility_tolerance': str(tolerance), 'purpose': 'numerical diagnostic, not exact model evaluation'}


def solve_milp_instance(payload, time_limit=5.0, capacity=None, threads=1):
    """Return weak-IC solver diagnostics and a separately replayed feasible menu.

    No epsilon changes the model or the buyer rule. Negative floating-point
    prices, if returned, are clamped to zero and recorded as recovery repairs.
    Replaying the shortest decimal representation makes the published menu an
    exactly specified rational menu; it may disagree with the weak assignment.
    """
    start = time.perf_counter()
    if isinstance(time_limit, bool) or not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('time_limit must be positive and finite.')
    if type(threads) is not int or threads < 1:
        raise ValueError('threads must be a positive integer.')
    data = validate_instance(payload)
    Q = data['parameters']['Q']
    limit = Q if capacity is None else capacity
    if type(limit) is not int or not 0 <= limit <= Q:
        raise ValueError('capacity must be an integer in 0,...,Q.')
    parsed = time.perf_counter()
    model = _build_model(data, limit)
    built = time.perf_counter()
    build_seconds = built-parsed
    remaining = float(time_limit)-build_seconds
    result = None
    caught = []
    if remaining > 0:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            result = milp(c=model['objective'], integrality=model['integer'],
                          bounds=Bounds(model['lower'], model['upper']),
                          constraints=LinearConstraint(model['matrix'], model['row_lo'], model['row_hi']),
                          options={'time_limit': remaining, 'mip_rel_gap': 0.0,
                                   'presolve': True, 'threads': threads})
            caught = [str(w.message) for w in seen]
    optimized = time.perf_counter()
    raw_status = None if result is None else int(result.status)
    labels = {0: 'optimal', 1: 'limit', 2: 'infeasible', 3: 'unbounded', 4: 'solver_error'}
    has = (result is not None and result.x is not None and
           len(result.x) == len(model['lower']) and bool(np.all(np.isfinite(result.x))))
    dual = None if result is None else _finite(getattr(result, 'mip_dual_bound', None))
    gap = None if result is None else _finite(getattr(result, 'mip_gap', None))
    nodes = None if result is None else _finite(getattr(result, 'mip_node_count', None))
    output = {'schema_version': 'pht-milp-result-v2', 'instance_id': data['instance_id'],
              'input_sha256': data['input_sha256'], 'method': 'milp',
              'parameters': data['parameters'], 'capacity_limit': limit,
              'status': 'build_budget_exhausted' if result is None else labels.get(raw_status, 'unknown'),
              'raw_status': raw_status,
              'solver_status': 'build_budget_exhausted' if result is None else labels.get(raw_status, 'unknown'),
              'message': 'Model construction used the optimization budget.' if result is None else str(result.message),
              'proof_status': 'floating_weak_ic_model_with_separate_exact_menu_replay',
              'has_incumbent': has, 'has_valid_menu': False, 'weak_ic_objective': None,
              'raw_min_incumbent_objective': None if result is None else _finite(getattr(result, 'fun', None)),
              'raw_min_dual_bound': dual, 'profit_upper_bound': None if dual is None else -dual,
              'bound_semantics': 'solver-reported numerical bound for the weak-IC relaxation; not an exact-rational certificate',
              'mip_gap': gap, 'mip_node_count': None if nodes is None else int(nodes),
              'menu': None, 'accounting': None, 'buyer_choices': None,
              'weak_ic_choices': None, 'choices_match': None, 'objective_minus_replay': None,
              'objective_agrees_with_replay': None, 'raw_incumbent_prices': None,
              'recovery_repairs': [], 'primal_residual': None,
              'price_recovery': None, 'raw_exact_replay': None, 'raw_tolerance_diagnostic': None,
              'numeric_recovery_issue': False, 'tie_assignment_mismatch': False,
              'quality_comparison_eligible': False,
              'assignment_objective_agreement': False,
              'net_profit': None,
              'solver_certificate': {
                  'model': 'weak_IC_with_arbitrary_utility_maximizing_tie_assignments',
                  'numerical_optimal_status': raw_status == 0,
                  'exact_rational_certificate': False},
              'original_problem_certificate': {
                  'certified_optimal': False,
                  'basis': 'canonical replay certifies menu value and buyer choices, not global optimality'},
              'dimensions': model['dimensions'], 'price_upper_bound': model['U_exact'],
              'solver': {'name': 'scipy.optimize.milp / HiGHS', 'scipy_version': scipy.__version__,
                         'numpy_version': np.__version__, 'optimization_budget_seconds': float(time_limit),
                         'solver_time_limit_seconds': max(0.0, remaining), 'mip_rel_gap': 0.0,
                         'threads_requested': threads, 'threads_option': 'passed through SciPy to HiGHS',
                         'warnings': caught},
              'formulation_source': 'public artifact milp_reference/reference_milp.py; compact weak-IC utility formulation',
              'numeric_diagnostics_tolerance': {'absolute': 1e-7, 'relative': 1e-8}}
    if has:
        vector = np.asarray(result.x)
        selected = [q+1 for q, j in enumerate(model['y']) if vector[j] >= 0.5]
        raw_prices = {str(q): float(vector[model['p'][q-1]]) for q in selected}
        # This produces a legal nonnegative price menu even for a small
        # solver-bound violation, without silently discarding the raw return.
        prices = {q: str(max(0.0, price)) for q, price in raw_prices.items()}
        for q, price in raw_prices.items():
            if price < 0:
                output['recovery_repairs'].append({'product': int(q), 'from': price, 'to': '0',
                                                  'reason': 'negative_solver_price_clamped_to_nonnegative_domain'})
        weak_choices = []
        for k, (cap, _, _) in enumerate(data['buyers']):
            options = [(float(vector[model['outside'][k]]), 0)]
            options.extend((float(vector[model['x'][k, q]]), q+1) for q in range(cap))
            weak_choices.append(max(options)[1])
        activity = model['matrix'] @ vector
        row_residual = float(np.maximum(np.maximum(model['row_lo']-activity, activity-model['row_hi']), 0).max(initial=0))
        bound_residual = max(float(np.maximum(model['lower']-vector, 0).max(initial=0)),
                             float(np.maximum(vector-model['upper'], 0).max(initial=0)))
        indices = np.flatnonzero(model['integer'])
        integrality_residual = float(np.abs(vector[indices]-np.rint(vector[indices])).max(initial=0))
        weak_objective = -float(model['objective'] @ vector)
        output.update(menu={'products': selected, 'prices': prices},
                      raw_incumbent_prices=raw_prices, weak_ic_choices=weak_choices,
                      weak_ic_objective=weak_objective,
                      primal_residual={'max_row_violation': row_residual,
                                       'max_bound_violation': bound_residual,
                                       'max_integrality_violation': integrality_residual,
                                       'catalog_capacity_satisfied': len(selected) <= limit})
        restored, recovery_info, raw_fraction_prices = _restore_rational_prices(
            data, selected, {int(q):Fraction(p) for q,p in prices.items()}, weak_choices, model['U'])
        output['price_recovery'] = recovery_info
        output['numeric_recovery_issue'] = recovery_info['numeric_recovery_issue']
        output['menu']['prices'] = {str(q):str(p) for q,p in restored.items()}
    recovered = time.perf_counter()
    if has and len(output['menu']['products']) <= limit:
        raw_replay = _replay(data, selected, raw_fraction_prices)
        output['raw_exact_replay'] = raw_replay
        output['raw_tolerance_diagnostic'] = _tolerance_replay_diagnostic(
            data, selected, raw_fraction_prices, Fraction(output['price_recovery']['neighborhood_tolerance']))
        replay = _replay(data, output['menu']['products'], output['menu']['prices'])
        output.update(replay)
        canonical = [item['product'] for item in replay['buyer_choices']]
        difference = output['weak_ic_objective']-float(Fraction(replay['accounting']['net_profit']))
        output['choices_match'] = canonical == output['weak_ic_choices']
        output['objective_minus_replay'] = difference
        tol = 1e-7+1e-8*max(abs(output['weak_ic_objective']), abs(float(Fraction(replay['accounting']['net_profit']))))
        output['objective_agrees_with_replay'] = abs(difference) <= tol
        output['tie_assignment_mismatch'] = (not output['numeric_recovery_issue'] and not output['choices_match'])
        output['assignment_objective_agreement'] = (not output['numeric_recovery_issue'] and
                                                   output['choices_match'] and output['objective_agrees_with_replay'])
        # Every returned nonnegative, capacity-feasible menu has well-defined
        # canonical demand. The weak-IC assignment is a separate diagnostic,
        # never a reason to remove that actual menu from quality comparisons.
        output['has_valid_menu'] = True
        output['quality_comparison_eligible'] = True
        output['net_profit'] = replay['accounting']['net_profit']
        if output['numeric_recovery_issue']:
            output['status'] += '_numeric_recovery_issue'
        elif not output['choices_match'] or not output['objective_agrees_with_replay']:
            output['status'] += '_replay_mismatch'
    elif has:
        output['status'] += '_invalid_catalog_capacity'
    ended = time.perf_counter()
    output['phase_seconds'] = {'parse_validation': parsed-start,
                               'preprocess_optimization': optimized-parsed,
                               'recovery': recovered-optimized,
                               'replay': ended-recovered, 'total': ended-start}
    output['optimization_detail_seconds'] = {'model_build': build_seconds, 'solver_call': optimized-built}
    output['optimization_budget_overrun_seconds'] = max(0.0, optimized-parsed-float(time_limit))
    output['completed_optimization'] = raw_status == 0
    output['budget_exhausted'] = (raw_status == 1 or result is None or
                                  output['optimization_budget_overrun_seconds'] > 0)
    output['timing_scope'] = ('disjoint phases; preprocessing_optimization includes model build and solver call; '
                              'recovery includes rational price restoration; replay includes canonical menu '
                              'validation and separately retained raw/tolerance diagnostics')
    return output
