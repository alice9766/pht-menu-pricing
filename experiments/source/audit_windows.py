"""Audit official UCI 321; no buyer inference, pricing, or DP.
Run download_official.py first, then this file. Eight prespecified evaluation
months (February/August in 2011--2014); preceding month is reference period.
All delay versions compared on identical within-month timestamps after a
240-minute warmup. Quality index is dimensionless, not a calibrated currency
value. Raw delay order is preserved, including any monotonicity failures.
"""
from pathlib import Path
import zipfile, json, hashlib, time
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parent
zpath=root/'electricityloaddiagrams20112014.zip'
selected=[f'{y}-{m:02d}' for y in range(2011,2015) for m in (2,8)]
refs={m:str(pd.Period(m,freq='M')-1) for m in selected}
needed=set(selected)|set(refs.values())
parts={m:[] for m in needed}; count=0; columns=None; missing=negative=zero=None;firstpos=None;lastpos=None
all_times=[]; start=time.monotonic()
with zipfile.ZipFile(zpath) as z:
 info=z.getinfo('LD2011_2014.txt')
 print('member',info.filename,info.file_size,flush=True)
 with z.open(info) as f:
  for chunk in pd.read_csv(f,sep=';',decimal=',',chunksize=12000):
   times=pd.to_datetime(chunk.iloc[:,0],format='%Y-%m-%d %H:%M:%S')
   values=chunk.iloc[:,1:].to_numpy(dtype=float)
   if columns is None:
    columns=list(chunk.columns[1:]);n=len(columns)
    missing=np.zeros(n,dtype=np.int64);negative=missing.copy();zero=missing.copy()
    firstpos=np.full(n,np.datetime64('NaT'),dtype='datetime64[ns]');lastpos=firstpos.copy()
   missing+=np.isnan(values).sum(0);negative+=(values<0).sum(0);zero+=(values==0).sum(0)
   ts=times.to_numpy(dtype='datetime64[ns]');all_times.append(ts)
   pos=values>0
   for j in range(len(columns)):
    ix=np.flatnonzero(pos[:,j])
    if len(ix):
     if np.isnat(firstpos[j]):firstpos[j]=ts[ix[0]]
     lastpos[j]=ts[ix[-1]]
   # Source timestamps are treated as interval endpoint labels; shifting by one
   # 15-minute interval gives complete calendar windows, including midnight end.
   calendar=(times-pd.Timedelta(minutes=15)).dt.strftime('%Y-%m')
   for m in set(calendar.unique())&needed:
    mask=(calendar==m).to_numpy()
    frame=pd.DataFrame(values[mask],columns=columns,index=pd.DatetimeIndex(ts[mask]))
    parts[m].append(frame)
   count+=len(chunk)
   print('parsed',count,flush=True)
windows={m:pd.concat(ps).sort_index() for m,ps in parts.items()}
times=np.concatenate(all_times)
steps=np.diff(times).astype('timedelta64[s]').astype(int)
records=[];delay_minutes=[240,120,60,30,15,0];lagsteps=[x//15 for x in delay_minutes]
for m in selected:
 frame=windows[m];ref=windows[refs[m]];v=frame.to_numpy();refv=ref.to_numpy();scale=np.mean(refv,axis=0)
 eval_period=pd.Period(m,freq='M');ref_period=pd.Period(refs[m],freq='M')
 complete=len(frame)==eval_period.days_in_month*96 and len(ref)==ref_period.days_in_month*96
 compare=v[16:]
 for j,client in enumerate(columns):
  ev=v[:,j];rv=refv[:,j];positive=bool(np.any(ev>0))
  eligible=bool(complete and np.isfinite(ev).all() and np.isfinite(rv).all() and scale[j]>0 and positive)
  # This secondary flag distinguishes late-start reference periods; a series
  # with zero values may still have genuine zero consumption after activation.
  reference_active_first_day=bool(np.any(rv[:96]>0))
  errs=[];alphas=[]
  if eligible:
   for lag in lagsteps:
    historical=v[16-lag:len(v)-lag,j] if lag else compare[:,j]
    e=float(np.mean(np.abs(compare[:,j]-historical))/scale[j])
    errs.append(e);alphas.append(1/(1+e))
  row={'client':client,'evaluation_month':m,'reference_month':refs[m],
       'evaluation_rows':len(frame),'comparison_rows':len(compare),'reference_rows':len(ref),
       'complete_window':complete,'reference_mean_kw':float(scale[j]),
       'evaluation_mean_kw':float(ev.mean()),'evaluation_zero_fraction':float(np.mean(ev==0)),
       'reference_zero_fraction':float(np.mean(rv==0)),
       'first_positive_timestamp':None if np.isnat(firstpos[j]) else str(firstpos[j]),
       'reference_active_first_day':reference_active_first_day,'eligible':eligible,
       'strict_alpha_increase':bool(np.all(np.diff(alphas)>0)) if eligible else None,
       'weak_alpha_increase':bool(np.all(np.diff(alphas)>=0)) if eligible else None}
  for d,e,a in zip(delay_minutes,errs,alphas):row[f'nmae_delay_{d}']=e;row[f'alpha_delay_{d}']=a
  records.append(row)
df=pd.DataFrame(records);df.to_csv(root/'window_quality_audit.csv',index=False)
df[df.eligible].to_json(root/'eligible_window_quality.json',orient='records',indent=2)
per_month=[]
for m,grp in df.groupby('evaluation_month'):
 e=grp[grp.eligible];strict=e[e.strict_alpha_increase==True];fail=e[e.strict_alpha_increase==False]
 per_month.append({'month':m,'candidate_meter_windows':len(grp),'eligible':len(e),
  'strict_alpha':len(strict),'nonstrict_alpha':len(fail),
  'strict_with_reference_active_first_day':int((strict.reference_active_first_day==True).sum()),
  'ineligible_allzero_evaluation':int((grp.evaluation_zero_fraction==1).sum()),
  'ineligible_nonpositive_reference':int((grp.reference_mean_kw<=0).sum())})
summary={'source_manifest':json.loads((root/'source_manifest.json').read_text()),
 'rows':count,'product_source_columns':len(columns),'column_range':[columns[0],columns[-1]],
 'first_timestamp':str(times[0]),'last_timestamp':str(times[-1]),
 'unique_step_seconds':np.unique(steps).tolist(),'duplicate_timestamps':int(pd.Index(times).duplicated().sum()),
 'missing_cells':int(missing.sum()),'negative_cells':int(negative.sum()),'zero_cells':int(zero.sum()),
 'allzero_source_columns':int(np.isnat(firstpos).sum()),
 'first_positive_after_2011':int(np.sum(firstpos>=np.datetime64('2012-01-01'))),
 'evaluation_months':selected,'reference_months':refs,'delay_minutes_in_quality_order':delay_minutes,
 'complete_calendar_windows':int(df.complete_window.sum()),'candidate_meter_windows':len(df),
 'eligible_meter_windows':int(df.eligible.sum()),
 'strictly_increasing_alpha_windows':int((df.strict_alpha_increase==True).sum()),
 'not_strictly_increasing_alpha_windows':int((df.strict_alpha_increase==False).sum()),
 'weakly_increasing_alpha_windows':int((df.weak_alpha_increase==True).sum()),
 'per_month':per_month,'elapsed_seconds':round(time.monotonic()-start,2),
 'interpretation':['Meter identity indexes an independent data source, not buyer type or product tier.',
 'Evaluation windows avoid March/October DST artifacts; zeros before activation remain distinguished by flags.',
 'Each window uses one meter source and six delayed versions; no multivariate quality aggregation.',
 'Alpha is a dimensionless technical quality proxy, not an observed monetary valuation.',
 'Do not reinterpret raw workload dispersion as theta, license, or buyer count.',
 'Window eligibility is documented above; monotonicity failures are retained in output.']}
(root/'audit_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
# Small complete source-window CSV for one deterministic existing source.
# Take first lexicographic eligible source in 2012-02; retain raw signed values.
e=df[(df.evaluation_month=='2012-02')&df.eligible].sort_values('client')
if len(e):
 client=e.iloc[0].client
 pd.concat([windows['2012-01'][[client]],windows['2012-02'][[client]]]).to_csv(root/'sample_complete_reference_and_evaluation.csv',index_label='timestamp_interval_end')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
