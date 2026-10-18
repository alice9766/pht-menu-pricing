"""Exact PHT capacity extension; no changes to the unrestricted solver.

solve_capacity(alpha, activation, service, buyers, L=None,
               recover=True, recover_all=False)

Input arrays omit the virtual root. Buyers are (access_cap, theta, count).
All arithmetic uses int or Fraction. None denotes an infeasible value (-infinity).
The result's frontier[l] is the optimum with AT MOST l offered products.
exact_tree_values counts PHT nodes; it is NOT a forced-exact-catalog oracle.

DP arithmetic time O(Q B^2 T L^2); rolled values O(Q B (L+1));
optional recovery records O(Q B T (L+1)). Direct buyer-utility checking of
recovered menus is additional verification work, not part of the DP bound.
"""

from fractions import Fraction


def _exact(value):
    if isinstance(value, (int, Fraction)):
        return value
    raise TypeError('Use int or Fraction for exact input, not float.')


def evaluate_menu(alpha, activation, service, buyers, products, prices):
    """Canonical buyer response: buy at zero utility; prefer higher quality."""
    aa = [0] + list(alpha)
    cc = [0] + list(activation)
    ss = [0] + list(service)
    selected = sorted(products)
    profit = -sum(cc[u] for u in selected)
    choices = []
    for cap, theta, count in buyers:
        chosen, utility = 0, 0
        for u in selected:
            if u <= cap:
                candidate = theta * aa[u] - prices[u]
                if candidate > utility or (candidate == utility and u > chosen):
                    chosen, utility = u, candidate
        choices.append(chosen)
        if chosen:
            profit += count * (prices[chosen] - ss[chosen])
    return profit, choices


def solve_capacity(alpha, activation, service, buyers, L=None,
                   recover=True, recover_all=False):
    alpha = [_exact(x) for x in alpha]
    activation = [_exact(x) for x in activation]
    service = [_exact(x) for x in service]
    q = len(alpha)
    if len(activation) != q or len(service) != q:
        raise ValueError('Product input lengths disagree.')
    if any(x <= 0 for x in alpha) or any(alpha[i] >= alpha[i+1] for i in range(q-1)):
        raise ValueError('Baseline values must be strictly increasing and positive.')
    if any(x < 0 for x in activation + service):
        raise ValueError('Costs must be nonnegative.')
    if L is None:
        L = q
    if not isinstance(L, int) or L < 0:
        raise ValueError('L must be a nonnegative integer.')
    requested_limit, L = L, min(L, q)
    clean = []
    for cap, theta, count in buyers:
        theta = _exact(theta)
        if (not isinstance(cap, int) or not 0 <= cap <= q or theta <= 0
                or not isinstance(count, int) or count <= 0):
            raise ValueError('Invalid buyer type.')
        clean.append((cap, theta, count))
    buyers = clean
    caps = sorted({cap for cap, theta, count in buyers if cap > 0})
    B = len(caps)
    thresholds = sorted({0, *[theta for cap, theta, count in buyers]})
    T = len(thresholds)
    do_recover = recover or recover_all

    def empty_menu():
        return dict(selected=[], prices={}, parents={}, edge_thresholds={},
                    evaluated_profit=0, buyer_choices=[0] * len(buyers),
                    tree_node_count=0)

    result = dict(Q=q, B=B, T=T, capacity_limit=L,
                  requested_capacity_limit=requested_limit,
                  optimal_profit=0, frontier=[0] * (L+1),
                  exact_tree_values=[0] + [None] * L,
                  optimal_node_counts=[0] * (L+1),
                  candidate_evaluations=0, group_updates=0,
                  recovery_record_count=0, menus=None,
                  selected=None, prices=None, parents=None,
                  edge_thresholds=None, evaluated_profit=None, buyer_choices=None)
    if not B or not L:
        if do_recover:
            result.update(empty_menu())
            if recover_all:
                result['menus'] = [empty_menu() for _ in range(L+1)]
        return result

    aa, cc, ss = [0] + alpha, [0] + activation, [0] + service
    group = {cap: b+1 for b, cap in enumerate(caps)}
    beta = [0] * (q+1)
    d = 1
    for u in range(1, q+1):
        while d <= B and caps[d-1] < u:
            d += 1
        beta[u] = d
    last = [0] * (B+2)
    for b in range(2, B+2):
        last[b] = caps[b-2]
    threshold_index = {t: r for r, t in enumerate(thresholds)}
    buckets = [[] for _ in thresholds]
    for cap, theta, count in buyers:
        if cap:
            buckets[threshold_index[theta]].append((group[cap], count))

    def new_layer():
        return [[[0] + [None] * L for _ in range(B+2)] for _ in range(q+1)]

    # H_k(a,b,T) is zero only for k=0, and infeasible for positive k.
    prev = new_layer()
    W = [0] * (B+1)
    records = {} if do_recover else None
    candidate_evaluations = group_updates = 0
    for r in range(T-1, -1, -1):
        t = thresholds[r]
        for d, count in buckets[r]:
            W[d] += count
        D = [0] * (B+2)
        for b in range(2, B+2):
            D[b] = D[b-1] + W[b-1]
        z = [t * aa[u] - ss[u] for u in range(q+1)]
        curr = new_layer()
        for b in range(1, B+2):
            # M[d][j] aggregates roots u>a with j nodes BELOW u.
            best = [[None] * L for _ in range(B+1)]
            best_u = [[None] * L for _ in range(B+1)]
            for a in range(last[b], -1, -1):
                values = prev[a][b][:]
                selected_records = [None] * (L+1)
                for d in range(max(1, beta[a]), b):
                    shift = -z[a] * (D[b] - D[d])
                    left = curr[a][d]
                    for j in range(L):
                        aggregate = best[d][j]
                        if aggregate is None:
                            continue
                        for ell in range(L-j):
                            left_value = left[ell]
                            if left_value is None:
                                continue
                            k = ell + j + 1
                            candidate_evaluations += 1
                            candidate = left_value + shift + aggregate
                            if values[k] is None or candidate > values[k]:
                                values[k] = candidate
                                selected_records[k] = (best_u[d][j], ell)
                curr[a][b] = values
                if do_recover:
                    for k in range(1, L+1):
                        if selected_records[k] is not None:
                            records[a, b, r, k] = selected_records[k]
                if a:
                    d = beta[a]
                    increment = z[a] * (D[b] - D[d]) - cc[a]
                    for j in range(L):
                        if values[j] is None:
                            continue
                        group_updates += 1
                        candidate = values[j] + increment
                        if best[d][j] is None or candidate > best[d][j]:
                            best[d][j], best_u[d][j] = candidate, a
        prev = curr

    exact_values = prev[0][B+1][:]
    frontier, counts = [], []
    running_value, running_count = 0, 0
    for k, value in enumerate(exact_values):
        if value is not None and value > running_value:
            running_value, running_count = value, k
        frontier.append(running_value)
        counts.append(running_count)
    result.update(optimal_profit=frontier[-1], frontier=frontier,
                  exact_tree_values=exact_values, optimal_node_counts=counts,
                  candidate_evaluations=candidate_evaluations,
                  group_updates=group_updates,
                  recovery_record_count=len(records) if do_recover else 0)
    if not do_recover:
        return result

    def recover_count(count):
        parents, edge_thresholds = {}, {}
        stack = [(0, B+1, 0, count)]
        while stack:
            a, b, r, k = stack.pop()
            if not k:
                continue
            while r < T and (a, b, r, k) not in records:
                r += 1
            if r == T:
                raise AssertionError('Feasible positive count has no recovery record.')
            u, ell = records[a, b, r, k]
            j = k - ell - 1
            if u in parents:
                raise AssertionError('Recovered duplicate product.')
            parents[u], edge_thresholds[u] = a, thresholds[r]
            stack.append((a, beta[u], r, ell))
            stack.append((u, b, r, j))
        selected = sorted(parents)
        if len(selected) != count:
            raise AssertionError('Recovered product count differs from tree state.')
        prices = {0: 0}
        for u in selected:
            a = parents[u]
            prices[u] = prices[a] + edge_thresholds[u] * (aa[u] - aa[a])
        prices.pop(0)
        profit, choices = evaluate_menu(alpha, activation, service, buyers, selected, prices)
        if profit != exact_values[count]:
            raise AssertionError(('Recovered profit mismatch', profit, exact_values[count]))
        return dict(selected=selected, prices=prices, parents=parents,
                    edge_thresholds=edge_thresholds, evaluated_profit=profit,
                    buyer_choices=choices, tree_node_count=count)

    cache = {}
    desired = set(counts) if recover_all else {counts[-1]}
    for count in sorted(desired):
        cache[count] = recover_count(count)
    result.update(cache[counts[-1]])
    if recover_all:
        result['menus'] = [cache[count] for count in counts]
    return result
