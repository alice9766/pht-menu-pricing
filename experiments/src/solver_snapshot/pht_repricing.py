"""Exact repricing of a strictly fixed offered set under the PHT model.

Every given product remains offered and its activation cost is charged, even
when demand is zero.  The auxiliary selection problem has zero activation
costs; omitted products are restored at a price above all buyer valuations.
"""
from bisect import bisect_right
from .pht_grouped_exact import exact, evaluate_menu, solve_grouped


def solve_fixed(alpha, activation, service, buyers, fixed_products):
    alpha = [exact(x) for x in alpha]
    activation = [exact(x) for x in activation]
    service = [exact(x) for x in service]
    q = len(alpha)
    if len(activation) != q or len(service) != q:
        raise ValueError("Product array lengths disagree.")
    if any(x <= 0 for x in alpha) or any(x >= y for x, y in zip(alpha, alpha[1:])):
        raise ValueError("Baseline values must be positive and strictly increasing.")
    if any(x < 0 for x in activation + service):
        raise ValueError("Costs must be nonnegative.")
    cleaned = []
    for cap, theta, count in buyers:
        theta = exact(theta)
        if not isinstance(cap, int) or not 0 <= cap <= q:
            raise ValueError("Invalid access cap.")
        if theta <= 0 or not isinstance(count, int) or count <= 0:
            raise ValueError("Invalid buyer type.")
        cleaned.append((cap, theta, count))
    buyers = cleaned
    products = list(fixed_products)
    if any(not isinstance(u, int) or not 1 <= u <= q for u in products):
        raise ValueError("Invalid fixed product index.")
    if len(set(products)) != len(products):
        raise ValueError("Fixed product indices must not repeat.")
    products.sort()
    fixed_cost = sum(activation[u - 1] for u in products)
    if not products:
        return dict(value=0, optimal_profit=0, auxiliary_value=0,
                    activation_total=0, products=[], selected=[], active_subset=[],
                    prices={}, buyer_choices=[0] * len(buyers), evaluated_profit=0)
    restricted_buyers = [(bisect_right(products, cap), theta, count)
                         for cap, theta, count in buyers]
    auxiliary = solve_grouped([alpha[u - 1] for u in products],
                              [0] * len(products),
                              [service[u - 1] for u in products],
                              restricted_buyers, recover=True)
    active = [products[u - 1] for u in auxiliary['selected']]
    prices = {products[u - 1]: price for u, price in auxiliary['prices'].items()}
    high_price = 1 + max((theta for _, theta, _ in buyers), default=0) * alpha[products[-1] - 1]
    for u in products:
        prices.setdefault(u, high_price)
    expected = auxiliary['optimal_profit'] - fixed_cost
    profit, choices = evaluate_menu(alpha, activation, service, buyers, products, prices)
    auxiliary_choices = [products[u - 1] if u else 0 for u in auxiliary['buyer_choices']]
    if profit != expected or choices != auxiliary_choices:
        raise AssertionError(("Fixed-set recovery failed", profit, expected, choices, auxiliary_choices))
    return dict(value=expected, optimal_profit=expected,
                auxiliary_value=auxiliary['optimal_profit'], activation_total=fixed_cost,
                products=products, selected=products, active_subset=active,
                prices=prices, buyer_choices=choices, evaluated_profit=profit)


if __name__ == '__main__':
    # Fixed activation cost remains payable even if nobody can buy.
    print(solve_fixed([1, 2], [3, 7], [0, 0], [(0, 2, 1)], [1, 2]))
