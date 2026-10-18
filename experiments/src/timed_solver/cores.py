"""Exact PHT recurrences with deferred recovery and cooperative checkpoints.

Recurrences are copied from the frozen pht_grouped_exact/pht_capacity snapshots.
Checkpoint interruption retains the last fully computed threshold layer. That
layer optimizes over a subset of candidate thresholds and represents a feasible
incumbent. Recovery never reads records from an unfinished threshold layer.
"""
class BudgetExpired(Exception):
    pass


def _setup(alpha, activation, service, buyers):
    q = len(alpha)
    caps = sorted({cap for cap, _, _ in buyers if cap > 0})
    B = len(caps)
    thresholds = sorted({0, *[theta for _, theta, _ in buyers]})
    group = {cap: b + 1 for b, cap in enumerate(caps)}
    beta = [0] * (q + 1)
    b = 1
    for u in range(1, q + 1):
        while b <= B and caps[b - 1] < u:
            b += 1
        beta[u] = b
    last = [0] * (B + 2)
    for b in range(2, B + 2):
        last[b] = caps[b - 2]
    ti = {t: r for r, t in enumerate(thresholds)}
    buckets = [[] for _ in thresholds]
    for cap, theta, n in buyers:
        if cap:
            buckets[ti[theta]].append((group[cap], n))
    return q, B, thresholds, beta, last, buckets, [0] + alpha, [0] + activation, [0] + service


def _empty_menu():
    return dict(selected=[], prices={}, parents={}, edge_thresholds={})


def optimize_joint(alpha, activation, service, buyers, recover=True, checkpoint=None):
    q, B, ts, beta, last, buckets, aa, cc, ss = _setup(alpha, activation, service, buyers)
    T = len(ts)
    prev = [[0] * (B + 2) for _ in range(q + 1)]
    records = {} if recover else None
    W = [0] * (B + 1)
    operations = updates = 0
    complete, completed_r = True, T
    if B:
        try:
            for r in range(T - 1, -1, -1):
                if checkpoint:
                    checkpoint()
                t = ts[r]
                for d, n in buckets[r]:
                    W[d] += n
                D = [0] * (B + 2)
                for b in range(2, B + 2):
                    D[b] = D[b - 1] + W[b - 1]
                z = [t * aa[u] - ss[u] for u in range(q + 1)]
                curr = [[0] * (B + 2) for _ in range(q + 1)]
                for b in range(1, B + 2):
                    best, best_u = [None] * (B + 1), [None] * (B + 1)
                    for a in range(last[b], -1, -1):
                        if checkpoint:
                            checkpoint()
                        value, chosen = prev[a][b], None
                        for d in range(max(1, beta[a]), b):
                            if best[d] is None:
                                continue
                            operations += 1
                            candidate = curr[a][d] - z[a] * (D[b] - D[d]) + best[d]
                            if candidate > value:
                                value, chosen = candidate, best_u[d]
                        curr[a][b] = value
                        if recover and chosen is not None:
                            records[a, b, r] = chosen
                        if a:
                            d = beta[a]
                            term = value + z[a] * (D[b] - D[d]) - cc[a]
                            updates += 1
                            if best[d] is None or term > best[d]:
                                best[d], best_u[d] = term, a
                prev, completed_r = curr, r
        except BudgetExpired:
            complete = False
    value = prev[0][B + 1]

    def recover_menu():
        if records is None:
            raise ValueError('Optimization was requested without recovery records.')
        parents, thresholds = {}, {}
        stack = [(0, B + 1, completed_r)]
        while stack:
            a, b, r = stack.pop()
            while r < T and (a, b, r) not in records:
                r += 1
            if r == T:
                continue
            u = records[a, b, r]
            if u in parents:
                raise AssertionError('Duplicate recovered node.')
            parents[u], thresholds[u] = a, ts[r]
            stack.append((a, beta[u], r))
            stack.append((u, b, r))
        prices = {0: 0}
        for u in sorted(parents):
            a = parents[u]
            prices[u] = prices[a] + thresholds[u] * (aa[u] - aa[a])
        prices.pop(0)
        return dict(selected=sorted(parents), prices=prices, parents=parents, edge_thresholds=thresholds)

    return dict(optimal_profit=value, complete=complete, recover=recover_menu if recover else None,
                candidate_evaluations=operations, group_updates=updates,
                completed_threshold_index=completed_r, Q=q, B=B, T=T)


def optimize_capacity(alpha, activation, service, buyers, L, checkpoint=None):
    q, B, ts, beta, last, buckets, aa, cc, ss = _setup(alpha, activation, service, buyers)
    T = len(ts)
    def new_layer():
        return [[[0] + [None] * L for _ in range(B + 2)] for _ in range(q + 1)]
    prev, records = new_layer(), {}
    W = [0] * (B + 1)
    operations = updates = 0
    complete, completed_r = True, T
    if B and L:
        try:
            for r in range(T - 1, -1, -1):
                if checkpoint:
                    checkpoint()
                t = ts[r]
                for d, n in buckets[r]:
                    W[d] += n
                D = [0] * (B + 2)
                for b in range(2, B + 2):
                    D[b] = D[b - 1] + W[b - 1]
                z = [t * aa[u] - ss[u] for u in range(q + 1)]
                curr = new_layer()
                for b in range(1, B + 2):
                    best = [[None] * L for _ in range(B + 1)]
                    best_u = [[None] * L for _ in range(B + 1)]
                    for a in range(last[b], -1, -1):
                        if checkpoint:
                            checkpoint()
                        values, choices = prev[a][b][:], [None] * (L + 1)
                        for d in range(max(1, beta[a]), b):
                            shift = -z[a] * (D[b] - D[d])
                            left = curr[a][d]
                            for j in range(L):
                                aggregate = best[d][j]
                                if aggregate is None:
                                    continue
                                for ell in range(L - j):
                                    if left[ell] is None:
                                        continue
                                    k = ell + j + 1
                                    operations += 1
                                    candidate = left[ell] + shift + aggregate
                                    if values[k] is None or candidate > values[k]:
                                        values[k], choices[k] = candidate, (best_u[d][j], ell)
                        curr[a][b] = values
                        for k in range(1, L + 1):
                            if choices[k] is not None:
                                records[a, b, r, k] = choices[k]
                        if a:
                            d = beta[a]
                            increment = z[a] * (D[b] - D[d]) - cc[a]
                            for j in range(L):
                                if values[j] is None:
                                    continue
                                updates += 1
                                candidate = values[j] + increment
                                if best[d][j] is None or candidate > best[d][j]:
                                    best[d][j], best_u[d][j] = candidate, a
                prev, completed_r = curr, r
        except BudgetExpired:
            complete = False
    exact_values = prev[0][B + 1]
    frontier, counts = [], []
    running, count = 0, 0
    for k, value in enumerate(exact_values):
        if value is not None and value > running:
            running, count = value, k
        frontier.append(running)
        counts.append(count)

    def recover_count(count):
        parents, thresholds = {}, {}
        stack = [(0, B + 1, completed_r, count)]
        while stack:
            a, b, r, k = stack.pop()
            if not k:
                continue
            while r < T and (a, b, r, k) not in records:
                r += 1
            if r == T:
                raise AssertionError('Missing recovery record for positive node count.')
            u, ell = records[a, b, r, k]
            if u in parents:
                raise AssertionError('Duplicate recovered node.')
            parents[u], thresholds[u] = a, ts[r]
            stack.append((a, beta[u], r, ell))
            stack.append((u, b, r, k - ell - 1))
        prices = {0: 0}
        for u in sorted(parents):
            a = parents[u]
            prices[u] = prices[a] + thresholds[u] * (aa[u] - aa[a])
        prices.pop(0)
        if len(parents) != count:
            raise AssertionError('Recovered count differs.')
        return dict(selected=sorted(parents), prices=prices, parents=parents, edge_thresholds=thresholds)

    def recover_menus():
        cache = {count: recover_count(count) for count in set(counts)}
        return cache[counts[-1]], [cache[count] for count in counts]

    return dict(optimal_profit=frontier[-1], frontier=frontier, optimal_node_counts=counts,
                complete=complete, recover=recover_menus, candidate_evaluations=operations,
                group_updates=updates, completed_threshold_index=completed_r, Q=q, B=B, T=T)
