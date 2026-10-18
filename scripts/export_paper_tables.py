"""Generate experiment parameter and result tables from inputs and computed summaries."""
from pathlib import Path
import argparse
import csv
import json
from collections import Counter
from fractions import Fraction
import re

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())
def lines(path):
    with path.open() as f:return [json.loads(x) for x in f if x.strip()]
def csv_file(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def export(base,summary,out):
    out.mkdir(parents=True,exist_ok=True)
    protocol=read(base/'protocol/data_protocol.json');final=read(base/'protocol/final_protocol.json')
    comp=lines(base/'data/final/comparison_instances.jsonl')
    synth=lines(base/'data/final/scaling_instances.jsonl')
    full=lines(base/'data/all/instances.jsonl')
    selected={(x['metadata']['source_client'],x['metadata']['evaluation_month']) for x in comp}
    windows={(x['metadata']['source_client'],x['metadata']['evaluation_month']) for x in full}
    meters={w[0] for w in windows}
    coords=sorted({(len(x['products']),len({k['access_cap'] for k in x['buyers'] if k['access_cap']>0}),
                   len({Fraction(k['theta']) for k in x['buyers']})+1) for x in synth})
    x=comp[0]
    caps=sorted({k['access_cap'] for k in x['buyers']})
    theta=sorted({Fraction(k['theta']) for k in x['buyers']})
    pop=sum(k['count'] for k in x['buyers'])
    group=Counter();valuation=Counter()
    for k in x['buyers']:group[k['access_cap']]+=k['count'];valuation[k['theta']]+=k['count']
    assert set(group.values())=={40} and set(valuation.values())=={30}
    table1=[
      {'panel':'a','parameter':'Products and buyers','value':f"Q={len(x['products'])}, B={len(caps)}, K={len(x['buyers'])}, T={len(theta)+1}"},
      {'panel':'a','parameter':'Product delay (min)','value':', '.join(map(str,protocol['products']['delay_minutes_in_quality_order']))},
      {'panel':'a','parameter':'Access limits','value':', '.join(map(str,caps))},
      {'panel':'a','parameter':'Valuation coefficients','value':', '.join(map(str,theta))},
      {'panel':'a','parameter':'Buyer population','value':f'{pop} in total, with 40 per access group and 30 per valuation level'},
      {'panel':'a','parameter':'Dependence scenarios','value':', '.join(protocol['buyers']['association_order'])},
      {'panel':'a','parameter':'Data windows','value':f"{protocol['source']['expected_window_count']} candidate windows; {len(windows)} valid windows; {len(meters)} meters"},
      {'panel':'a','parameter':'Full scenario pool','value':f'{len(windows)} windows x 9 scenarios = {len(full)}'},
      {'panel':'a','parameter':'Method comparison subset','value':f'{len(selected)} windows x 9 scenarios = {len(comp)}'},
      {'panel':'b','parameter':'Baseline (Q,B,T)','value':str(tuple(final['scaling']['base_dimensions'][d] for d in ['Q','B','T']))},
      {'panel':'b','parameter':'Candidate products Q','value':', '.join(map(str,sorted({q for q,b,t in coords if b==4 and t==9})))},
      {'panel':'b','parameter':'Access groups B','value':', '.join(map(str,final['scaling']['axis_values']['B']))},
      {'panel':'b','parameter':'Candidate thresholds T','value':', '.join(map(str,final['scaling']['axis_values']['T']))},
      {'panel':'b','parameter':'Joint growth configurations','value':'; '.join(str(tuple(i[d] for d in ['Q','B','T'])) for i in final['scaling']['additional_coordinates'] if i['axis']=='joint_growth')},
      {'panel':'b','parameter':'Buyer population','value':str(final['scaling']['population'])},
      {'panel':'b','parameter':'Total instances','value':f"{len(coords)} configurations x {len(final['scaling']['seeds'])} seeds x {len(final['scaling']['cost_profiles'])} cost configurations = {len(synth)}"}]
    s=read(summary/'summary.json');rows=[]
    for suite in ['real_comparison','synthetic_scaling']:
        for method in ['joint','G0','G1','G2','MILP']:
            t=next(x for x in s['timing_summary'] if x['suite']==suite and x['method']==method)
            coverage=next(x for x in s['quality_coverage'] if x['suite']==suite and x['method']==method)
            valid=coverage['instances_with_valid_menu']
            if method=='joint':matches=valid;loss=0.0
            else:
                q=next(x for x in s['quality_summary'] if x['suite']==suite and x['method']==method and x['grouping']=='all')
                matches=q['within_numerical_tolerance']
                weight='meter_equal' if suite=='real_comparison' else 'instance_weighted'
                loss=100*q['relative_gap'][weight]['mean']
            rows.append({'panel':'a' if suite=='real_comparison' else 'b',
                         'method':'PHT-DP' if method=='joint' else method,
                         'valid':valid,'planned':t['scheduled_instances'],'optimal':matches,
                         'loss_percent':loss,'completed':t['instances_all_repeats_completed']})
    csv_file(out/'table_I_parameters.csv',table1);csv_file(out/'table_II_results.csv',rows)
    def escape(value):
        return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    def tabular(path, headers, body):
        nl = '\n'
        row_end = r' \\' + nl
        contents = r'\begin{tabular}{' + 'l'*len(headers) + '}' + nl
        contents += ' & '.join(headers) + row_end + r'\hline' + nl
        for row in body:
            contents += ' & '.join(escape(value) for value in row) + row_end
        contents += r'\end{tabular}' + nl
        path.write_text(contents)
    for panel in ('a','b'):
        tabular(out/f'table_I_{panel}.tex', ['Parameter','Value or configuration'],
                [[x['parameter'],x['value']] for x in table1 if x['panel']==panel])
        tabular(out/f'table_II_{panel}.tex',
                ['Method','Valid / planned','Optimal / valid','Loss (\\%)','Completed / planned'],
                [[x['method'],f"{x['valid']}/{x['planned']}",f"{x['optimal']}/{x['valid']}",
                  f"{x['loss_percent']:.4f}",f"{x['completed']}/{x['planned']}"]
                 for x in rows if x['panel']==panel])
    print('Experiment parameter and result tables written from the supplied run.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.base,a.summary,a.output)
