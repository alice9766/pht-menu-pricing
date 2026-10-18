"""Exact PHT DP with license-boundary compression and grouped last-child maxima.

API: solve_grouped(alpha, activation, service, buyers, recover=True)
Products use 1-based indices in outputs; input arrays omit the virtual root.
Each buyer is (access_cap, positive_theta, positive_integer_count).
Integers and fractions.Fraction inputs are exact. Floats are deliberately rejected.
All DP value layers are rolled over thresholds. Recovery stores O(Q B T) choices;
value-only mode uses O(Q B) DP storage, excluding input and threshold buckets.
"""
from fractions import Fraction
from itertools import accumulate


def exact(x):
    if isinstance(x, (int, Fraction)):
        return x
    raise TypeError('Use int or Fraction for exact input, not float.')


def evaluate_menu(alpha, activation, service, buyers, products, prices):
    q=len(alpha)
    aa=[0]+list(alpha)
    cc=[0]+list(activation)
    ss=[0]+list(service)
    selected=sorted(products)
    profit=-sum(cc[u] for u in selected)
    choices=[]
    for cap,theta,count in buyers:
        chosen=0
        utility=0
        for u in selected:
            if u<=cap:
                val=theta*aa[u]-prices[u]
                if val>utility or (val==utility and u>chosen):
                    chosen,utility=u,val
        choices.append(chosen)
        if chosen:
            profit+=count*(prices[chosen]-ss[chosen])
    return profit,choices


def solve_grouped(alpha, activation, service, buyers, recover=True, reuse_inactive=False):
    alpha=[exact(z) for z in alpha]
    activation=[exact(z) for z in activation]
    service=[exact(z) for z in service]
    q=len(alpha)
    if len(activation)!=q or len(service)!=q:
        raise ValueError('Product input lengths disagree.')
    if any(x<=0 for x in alpha) or any(alpha[i]>=alpha[i+1] for i in range(q-1)):
        raise ValueError('Baseline values must be strictly increasing and positive.')
    if any(x<0 for x in activation+service):
        raise ValueError('Costs must be nonnegative.')
    clean=[]
    for cap,theta,count in buyers:
        theta=exact(theta)
        if not isinstance(cap,int) or not 0<=cap<=q or theta<=0 or not isinstance(count,int) or count<=0:
            raise ValueError('Invalid buyer type.')
        clean.append((cap,theta,count))
    buyers=clean
    caps=sorted({cap for cap,theta,n in buyers if cap>0})
    B=len(caps)
    thresholds=sorted({0,*[theta for cap,theta,n in buyers]})
    T=len(thresholds)
    if not B:
        return {'optimal_profit':0,'selected':[] if recover else None,'prices':{} if recover else None,
                'parents':{} if recover else None,'edge_thresholds':{} if recover else None,
                'evaluated_profit':0 if recover else None,'buyer_choices':[0]*len(buyers) if recover else None,
                'candidate_evaluations':0,'group_updates':0,'Q':q,'B':B,'T':T}
    aa=[0]+alpha;cc=[0]+activation;ss=[0]+service
    group={cap:b+1 for b,cap in enumerate(caps)}
    beta=[0]*(q+1)
    b=1
    for u in range(1,q+1):
        while b<=B and caps[b-1]<u:b+=1
        beta[u]=b
    # Eligible product prefix for each external group boundary b.
    last=[0]*(B+2)
    for b in range(1,B+2):
        last[b]=caps[b-2] if b>=2 else 0
    threshold_index={t:x for x,t in enumerate(thresholds)}
    buckets=[[] for _ in thresholds]
    for cap,theta,n in buyers:
        if cap:
            buckets[threshold_index[theta]].append((group[cap],n))
    W=[0]*(B+1)
    # H(a,b,T)=0 because the empty forest is always feasible.
    prev=[[0]*(B+2) for _ in range(q+1)]
    records={} if recover else None
    candidate_evaluations=group_updates=0
    for x in range(T-1,-1,-1):
        t=thresholds[x]
        for d,n in buckets[x]:W[d]+=n
        D=[0]*(B+2)
        for b in range(2,B+2):D[b]=D[b-1]+W[b-1]
        z=[t*aa[u]-ss[u] for u in range(q+1)]
        cutoff=[0]*(B+2)
        if reuse_inactive:
            changed=[0]*(B+1)
            for d,n in buckets[x]:changed[d]=caps[d-1]
            for b in range(2,B+2):cutoff[b]=max(cutoff[b-1],changed[b-1])
            if not cutoff[B+1]:continue
        curr=[row[:] for row in prev] if reuse_inactive else [[0]*(B+2) for _ in range(q+1)]
        for b in range(1,B+2):
            if reuse_inactive and not cutoff[b]:continue
            best=[None]*(B+1)
            best_u=[None]*(B+1)
            # Before a is processed, best[d] contains exactly roots u>a.
            for a in range(last[b],-1,-1):
                val=prev[a][b]
                chosen=None
                for d in (range(max(1,beta[a]),b) if not reuse_inactive or a<cutoff[b] else ()):
                    if best[d] is None:continue
                    candidate_evaluations+=1
                    cand=curr[a][d]-z[a]*(D[b]-D[d])+best[d]
                    if cand>val:
                        val=cand
                        chosen=best_u[d]
                curr[a][b]=val
                # Missing record means inherit the next-higher threshold layer.
                if recover and chosen is not None:
                    records[a,b,x]=chosen
                if a:
                    d=beta[a]
                    term=curr[a][b]+z[a]*(D[b]-D[d])-cc[a]
                    group_updates+=1
                    if best[d] is None or term>best[d]:
                        best[d],best_u[d]=term,a
        prev=curr
    result={'optimal_profit':prev[0][B+1],
            'candidate_evaluations':candidate_evaluations,'group_updates':group_updates,
            'Q':q,'B':B,'T':T,'selected':None,'prices':None,'parents':None,
            'edge_thresholds':None,'evaluated_profit':None,'buyer_choices':None}
    if not recover:return result
    parents={};edge_thresholds={}
    # Iterative recovery avoids depending on the Python recursion limit.
    stack=[(0,B+1,0)]
    while stack:
        a,b,x=stack.pop()
        while x<T and (a,b,x) not in records:x+=1
        if x==T:continue
        u=records[a,b,x]
        if u in parents:raise AssertionError('Recovered duplicate product.')
        parents[u]=a;edge_thresholds[u]=thresholds[x]
        stack.append((a,beta[u],x))
        stack.append((u,b,x))
    selected=sorted(parents)
    prices={0:0}
    for u in selected:
        a=parents[u]
        prices[u]=prices[a]+edge_thresholds[u]*(aa[u]-aa[a])
    prices.pop(0)
    profit,choices=evaluate_menu(alpha,activation,service,buyers,selected,prices)
    if profit!=result['optimal_profit']:
        raise AssertionError(('Recovered menu profit mismatch',profit,result['optimal_profit']))
    result.update(selected=selected,prices=prices,parents=parents,
                  edge_thresholds=edge_thresholds,evaluated_profit=profit,buyer_choices=choices)
    return result

if __name__=='__main__':
    print(solve_grouped([1,2,3],[0,0,0],[0,0,0],[(2,4,1),(3,2,1)]))


def solve_cubic_value(alpha, activation, service, buyers):
    """Uncompressed last-child DP, O(Q^3 T) time and O(Q^2) rolled space.
    Returns an exact value and transition count; intended as a comparison oracle.
    """
    q=len(alpha)
    aa=[0]+[exact(z) for z in alpha]
    cc=[0]+[exact(z) for z in activation]
    ss=[0]+[exact(z) for z in service]
    buyers=[(cap,exact(theta),n) for cap,theta,n in buyers]
    caps=sorted({cap for cap,theta,n in buyers if cap>0})
    B=len(caps)
    ts=sorted({0,*[theta for cap,theta,n in buyers]})
    group={cap:b+1 for b,cap in enumerate(caps)}
    beta=[0]*(q+2)
    b=1
    for u in range(1,q+1):
        while b<=B and caps[b-1]<u:b+=1
        beta[u]=b
    beta[q+1]=B+1
    ti={t:x for x,t in enumerate(ts)}
    buckets=[[] for _ in ts]
    for cap,theta,n in buyers:
        if cap:buckets[ti[theta]].append((group[cap],n))
    W=[0]*(B+1)
    prev=[[0]*(q+2) for _ in range(q+1)]
    count=0
    for x in range(len(ts)-1,-1,-1):
        for b,n in buckets[x]:W[b]+=n
        D=[0]*(B+2)
        for b in range(2,B+2):D[b]=D[b-1]+W[b-1]
        z=[ts[x]*aa[u]-ss[u] for u in range(q+1)]
        curr=[[0]*(q+2) for _ in range(q+1)]
        for width in range(2,q+2):
            for a in range(q+2-width):
                v=a+width
                val=prev[a][v]
                for u in range(a+1,v):
                    if beta[u]>=beta[v]:continue
                    count+=1
                    g=(z[u]-z[a])*(D[beta[v]]-D[beta[u]])-cc[u]
                    val=max(val,curr[a][u]+g+curr[u][v])
                curr[a][v]=val
        prev=curr
    return {'optimal_profit':prev[0][q+1],'candidate_evaluations':count,'Q':q,'B':B,'T':len(ts)}
