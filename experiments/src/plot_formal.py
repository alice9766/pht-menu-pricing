"""Render descriptive figures from retained formal records; no result selection."""
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import argparse, json, statistics, os
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS=['joint','G0','G1','G2','MILP']
LABELS={'joint':'PHT-DP','G0':'G0','G1':'G1','G2':'G2','MILP':'MILP'}
COLORS={'joint':'#1b6ca8','G0':'#d18424','G1':'#40885c','G2':'#92559b','MILP':'#666666'}
PROFILES=['balanced','activation_heavy','service_heavy']
ASSOC=['positive','independent','negative']

def read_latest(path):
    latest={}
    for line in path.open():
        r=json.loads(line); latest[r['job_id']]=r
    return list(latest.values())

def mean(v):return sum(v)/len(v) if v else None
def number(x):return float(Fraction(str(x)))
def inv(menu):
    p=[Fraction(menu['prices'][str(q)]) for q in sorted(menu['products'])]
    return any(a>b for a,b in zip(p,p[1:]))
def eligible_result(r):
    z=r.get('result') or {}
    return (r.get('runner_status')=='completed' and z.get('replay_verified')
            and z.get('proof_status')=='exact' and not z.get('budget_exhausted',False)
            and z.get('accounting'))
def clustered_mean(pairs):
    d=defaultdict(list)
    for meter,v in pairs:d[meter].append(v)
    return mean([mean(v) for v in d.values()])

def main(root):
    out=root/'figures/formal';out.mkdir(parents=True,exist_ok=True)
    meta={}
    registry=json.loads((root/'data/formal/input_registry.json').read_text())
    for source in registry:
        for line in (root/source['input_path']).open():
            x=json.loads(line); meta[x['instance_id']]=x['metadata']
    isolated=read_latest(root/'results/formal_isolated/results.jsonl')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':220,'axes.grid':True,'grid.alpha':.20,
                         'font.family':'DejaVu Sans','pdf.fonttype':42})
    values={}
    def save(fig,name):
        fig.savefig(out/(name+'.png'),bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)

    # Figure 1: every returned timing in the small paired comparison; status
    # counts and value comparisons belong in the accompanying result tables.
    groups={m:[] for m in METHODS};opt={};recovery={};replay={}
    for r in isolated:
        if r['suite']=='real_comparison':groups[r['method']].append(r)
    for m in METHODS:
        opt[m]=[r['result']['phase_seconds']['preprocess_optimization'] for r in groups[m] if r.get('result')]
        recovery[m]=[r['result']['phase_seconds']['recovery'] for r in groups[m] if r.get('result')]
        replay[m]=[r['result']['phase_seconds']['replay'] for r in groups[m] if r.get('result')]
    fig,axes=plt.subplots(1,2,figsize=(10.3,3.6),constrained_layout=True)
    axes[0].boxplot([opt[m] for m in METHODS],tick_labels=[LABELS[m] for m in METHODS],showfliers=False)
    axes[0].set_yscale('log');axes[0].set_ylabel('Optimization time (s)');axes[0].set_title('(a) Paired six-product instances')
    ix=np.arange(len(METHODS));width=.26
    for shift,(key,d) in zip([-1,0,1],[('Optimization',opt),('Menu recovery',recovery),('Verification',replay)]):
        axes[1].bar(ix+shift*width,[statistics.median(d[m]) for m in METHODS],width,label=key)
    axes[1].set_xticks(ix,[LABELS[m] for m in METHODS]);axes[1].set_yscale('log')
    axes[1].set_ylabel('Median phase time (s)');axes[1].set_title('(b) Disjoint timing phases');axes[1].legend(fontsize=8)
    save(fig,'sec6_01_method_timing')
    values['method_timing']={m:{'scheduled':len(groups[m]),'returned':len(opt[m]),
        'median_optimization_s':statistics.median(opt[m]) if opt[m] else None,
        'median_recovery_s':statistics.median(recovery[m]) if recovery[m] else None,
        'median_replay_s':statistics.median(replay[m]) if replay[m] else None} for m in METHODS}

    # Median of repeated runs first; medians across all three seeds and cost
    # profiles second. Soft timeouts retain their measured phase durations;
    # missing timings are counted separately, never filled with a budget value.
    repeat=defaultdict(list)
    for r in isolated:
        if r['suite']=='synthetic_scaling':repeat[r['instance_id'],r['method']].append(r)
    coords=defaultdict(list)
    for (iid,m),rs in repeat.items():
        ts=[r['result']['phase_seconds']['preprocess_optimization'] for r in rs
            if r.get('result') and r['result'].get('phase_seconds')]
        md=meta[iid];d=md['dimensions'];complete=all(r['runner_status']=='completed' and
            not (r.get('result') or {}).get('budget_exhausted',False) and
            not (r.get('result') or {}).get('optimization_budget_overrun_seconds',0) and
            (r.get('result') or {}).get('raw_status',0)==0 for r in rs)
        coords[(d['Q'],d['B'],d['T'],m)].append((statistics.median(ts) if ts else None,complete,len(rs)-len(ts)))
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    base=(32,4,9);grid=[[8,16,32,64,128],[1,2,4,8,16],[3,5,9,17,33]]
    scaling_rows=[]
    for j,ax in enumerate(axes):
        for m in METHODS:
            xs=[];ys=[];partial=[]
            for x in grid[j]:
                d=list(base);d[j]=x;k=tuple(d)+(m,)
                if k not in coords:continue
                rows=coords[k];xs.append(x)
                observed=[v for v,c,n in rows if v is not None]
                ys.append(statistics.median(observed) if observed else float('nan'))
                partial.append(not all(c for v,c,n in rows))
                scaling_rows.append({'axis':'QBT'[j],'x':x,'method':m,'instances':len(rows),
                    'median_observed_optimization_s':ys[-1] if observed else None,
                    'all_repeats_complete_instances':sum(c for v,c,n in rows),
                    'missing_phase_timings':sum(n for v,c,n in rows)})
            ax.plot(xs,ys,'o-',color=COLORS[m],label=LABELS[m],markersize=4)
            for x,y,b in zip(xs,ys,partial):
                if b:ax.scatter([x],[y],marker='x',s=65,color=COLORS[m],zorder=5)
        ax.axhline(5,ls='--',color='#999999',lw=1)
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(grid[j],[str(x) for x in grid[j]])
        ax.set_xlabel(['Products Q','Permission groups B','Thresholds T'][j]);ax.set_title(['(a) Q scan','(b) B scan','(c) T scan'][j])
        ax.set_ylabel('Observed optimization time (s)')
    axes[0].legend(fontsize=8,ncol=2);save(fig,'sec6_02_scalability')
    values['scaling']=scaling_rows

    caps=defaultdict(list)
    for r in isolated:
        if r['suite']=='real_capacity' and eligible_result(r):
            z=r['result'];md=meta[r['instance_id']];v=Fraction(z['accounting']['net_profit'])
            for row in z['frontier']:
                if v>0:caps[md['cost_profile'],row['capacity']].append((md['source_client'],float(Fraction(row['net_profit'])/v)))
    fig,ax=plt.subplots(figsize=(6.7,3.8),constrained_layout=True)
    capacity_rows=[]
    for profile in PROFILES:
        ys=[clustered_mean(caps[profile,l]) for l in range(7)]
        ax.plot(range(7),ys,'o-',label=profile.replace('_',' '))
        capacity_rows.append({'profile':profile,'capacity':list(range(7)),'cluster_equal_weight_fraction':ys})
    ax.set_xlabel('Maximum offered products L');ax.set_ylabel('Fraction of unrestricted optimal profit')
    ax.set_xticks(range(7));ax.set_ylim(-.02,1.04);ax.legend(fontsize=8)
    save(fig,'sec6_03_capacity');values['capacity']=capacity_rows

    # Full-pool business results: retain the meter as the aggregation unit.
    rates=defaultdict(list);paired=defaultdict(dict);full_counts=defaultdict(int);gammas=defaultdict(list)
    for r in read_latest(root/'results/formal_bulk/results.jsonl'):
        if not eligible_result(r):continue
        z=r['result'];md=meta[r['instance_id']]
        if r['suite']=='real_full':
            rates[md['cost_profile'],md['association']].append((md['source_client'],float(inv(z['menu']))))
            paired[md['source_client'],md['evaluation_month'],md['cost_profile']][md['association']]=number(z['accounting']['net_profit'])
            full_counts[len(z['menu']['products'])]+=1
        gamma=md.get('quality_mapping_gamma','1')
        if r['suite']=='mapping_sensitivity':
            gammas[gamma,md['association']].append((md['source_client'],float(inv(z['menu']))))
    differences=defaultdict(lambda:defaultdict(list))
    for (meter,month,profile),p in paired.items():
        if 'positive' in p and 'negative' in p and p['positive']>0:
            differences[profile][meter].append(100*(p['negative']-p['positive'])/p['positive'])
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.8),constrained_layout=True)
    axes[0].boxplot([[mean(v) for v in differences[p].values()] for p in PROFILES],
                     tick_labels=['Balanced','Activation-heavy','Service-heavy'],showfliers=False)
    axes[0].axhline(0,color='#777777',ls='--',lw=1);axes[0].set_ylabel('Negative vs positive association: profit change (%)')
    axes[0].set_title('(a) Mean within meter, then meter distribution')
    for i,assoc in enumerate(ASSOC):
        ys=[100*clustered_mean(rates[p,assoc]) for p in PROFILES]
        axes[1].bar(np.arange(3)+(i-1)*.25,ys,.25,label=assoc)
    axes[1].set_xticks(np.arange(3),['Balanced','Activation-heavy','Service-heavy'])
    axes[1].set_ylim(0,105);axes[1].set_ylabel('Returned optima with a price inversion (%)')
    axes[1].set_title('(b) Meter-equal-weight proportions');axes[1].legend(fontsize=8)
    save(fig,'sec6_04_association')
    values['association']={'price_inversion_rates':[
        {'profile':p,'association':a,'meter_equal_weight_rate':clustered_mean(rates[p,a])} for p in PROFILES for a in ASSOC],
        'menu_size_counts':dict(full_counts)}
    (out/'figure_values.json').write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n')
    captions='''图A：216个同输入比较实例的优化耗时及分阶段中位耗时。每方法同一实例运行一次；图中展示有返回计时的任务，完成状态与MILP恢复诊断另见汇总表。恢复和核验与优化分别计时。
图B：预定Q、B、T网格上的实际观测优化耗时。先对同实例三次运行取中位数，再汇总三个种子与三个成本情景。叉号表示该坐标方法存在未完成优化或超出优化预算的运行；菜单恢复诊断另外统计。有返回的软超时保留实际阶段计时，不解释为已完成求解。没有阶段计时的任务仅进入状态及覆盖率统计，不填预算值参与中位数。Q=64、128按协议只运行PHT-DP。
图C：在售数量上限与相对无约束最优净利润。对同输入计算比例，再按测点等权汇总；曲线分成本情景，关联情景保持配对。
图D：完整池的关联情景分析。左图先在测点内跨月份平均，再展示测点间分布；右图是返回的最优菜单存在价格逆序的测点等权比例。这些比例表示存在对应返回最优解，不推断非单调定价的必要性。
箱线图为保持版面清晰不绘制须外散点，逐实例结果与极值仍保留在结果表中。
'''
    (out/'图表说明.txt').write_text(captions)
    print(json.dumps({'figures':4,'output':str(out)},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();main(a.package)
