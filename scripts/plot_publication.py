#!/usr/bin/env python3
"""Publication-sized Section 6 figures, drawn only from retained experiment data.

Original-paper palette: blue #2166AC, orange #B85C00, green #16856B.
All exported PDFs are vector artwork; PNGs are 600 dpi. No solver is run.
"""
from pathlib import Path
from fractions import Fraction
import json
import hashlib
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.patches import Rectangle, ConnectionPatch, Patch

BLUE='#2166AC'; ORANGE='#B85C00'; GREEN='#16856B'; PURPLE='#80639A'; GRAY='#77838D'
METHODS=['joint','G0','G1','G2','MILP']
LABELS={'joint':'PHT-DP','G0':'G0','G1':'G1','G2':'G2','MILP':'MILP'}
COLORS=dict(zip(METHODS,[BLUE,ORANGE,GREEN,PURPLE,GRAY]))
MARKERS=dict(zip(METHODS,['o','s','^','D','v']))
LINES=dict(zip(METHODS,['-','--','-.',':','--']))

def style():
    plt.rcParams.update({'font.family':'serif','font.serif':['Nimbus Roman','STIXGeneral','DejaVu Serif'],
        'font.size':8.5,'mathtext.fontset':'stix','axes.labelsize':8.5,'axes.titlesize':8.5,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
        'xtick.major.size':3,'ytick.major.size':3,'xtick.major.width':.6,'ytick.major.width':.6,
        'xtick.direction':'out','ytick.direction':'out','axes.labelpad':3,
        'grid.color':'#DBE2E7','grid.linewidth':.5,'axes.axisbelow':True,
        'legend.frameon':False,'legend.handlelength':2.1,'legend.borderaxespad':.2,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})

def save(fig,out,name):
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'):
        fig.savefig(out/f'{name}.{ext}',dpi=600,facecolor='white')
    plt.close(fig)

def efficiency(data,out):
    # One common log axis: filled median bars, observed 5th--95th percentiles.
    fig,ax=plt.subplots(figsize=(3.4,2.50))
    fig.subplots_adjust(left=.17,right=.985,bottom=.235,top=.90)
    ticks=[]
    for i,m in enumerate(METHODS):
        row=next(r for r in data['timing'] if r['method']==m)
        values=np.asarray(row['optimization_seconds'],float)*1000
        assert len(values)==row['all_repeats_completed_instances']
        lo,med,hi=np.percentile(values,[5,50,95])
        color=COLORS[m]
        ax.bar(i,med-.1,bottom=.1,width=.58,color=to_rgba(color,.90 if m=='joint' else .32),
               edgecolor=color,lw=.85,zorder=2)
        ax.errorbar(i,med,yerr=[[med-lo],[hi-med]],fmt='o',ms=3.3,
                    color=color,ecolor=color,elinewidth=1.0,capsize=3,capthick=.9,zorder=4)
        label=f'{med:.3f}' if med<100 else f'{med:,.1f}'
        ax.annotate(label,(i,hi),xytext=(0,5),textcoords='offset points',
                    ha='center',va='bottom',fontsize=8,color=color,
                    fontweight='bold' if m=='joint' else 'normal')
        ticks.append(f"{LABELS[m]}\n({len(values)})")
    ax.set_yscale('log');ax.set_ylim(.1,12000);ax.set_xlim(-.6,4.6)
    ax.set_yticks([.1,1,10,100,1000,10000],['0.1','1','10','100','1,000','10,000'])
    ax.yaxis.set_minor_locator(NullLocator());ax.grid(axis='y')
    ax.set_xticks(range(5),ticks);ax.tick_params(axis='x',length=0,pad=5)
    ax.get_xticklabels()[0].set_color(BLUE);ax.get_xticklabels()[0].set_fontweight('bold')
    ax.set_ylabel('Optimization time (ms, log scale)')
    fig.text(.17,.957,'Median and 5th–95th percentile',fontsize=8,color='#46535C')
    fig.text(.17,.04,'Parentheses: completed instances out of 864',fontsize=7.5,color='#52616B')
    save(fig,out,'fig6_1')

def scaling(data,out):
    specs=[('Q',[(q,4,9) for q in [8,16,32,64,128,256,512]],r'$B=4,\ T=9$'),
           ('B',[(32,b,9) for b in [1,2,4,8,16]],r'$Q=32,\ T=9$'),
           ('T',[(32,4,t) for t in [3,5,9,17,33]],r'$Q=32,\ B=4$')]
    lookup={(r['axis'],tuple(r['coordinate']),r['method']):r for r in data['scaling']}
    fig,axes=plt.subplots(1,3,figsize=(7.0,2.6),sharey=True)
    fig.subplots_adjust(left=.075,right=.985,bottom=.23,top=.80,wspace=.18)
    handles=[]
    for i,(axis,points,fixed) in enumerate(specs):
        ax=axes[i];ax.set_xscale('log',base=2);ax.set_yscale('log')
        ax.set_ylim(.8,8000);ax.set_yticks([1,10,100,1000],['1','10','100','1,000'])
        ax.yaxis.set_minor_locator(NullLocator());ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis='y'); ax.axhline(5000,color='#8A9197',ls=(0,(4,3)),lw=.7)
        if i==2:ax.text(.98,.99,'5 s budget',transform=ax.transAxes,ha='right',va='top',fontsize=7.3,color='#65727C')
        values=[p['QBT'.index(axis)] for p in points]
        ax.set_xlim(values[0]/1.16,values[-1]*1.22);ax.set_xticks(values,list(map(str,values)))
        for m in ['G2','G1','G0','joint']:
            rows=[lookup.get((axis,p,m)) for p in points]
            ys=[np.nan if row is None or row['median_completed_optimization_seconds'] is None else 1000*row['median_completed_optimization_seconds'] for row in rows]
            if m=='joint':
                intervals=np.array([np.percentile(row['completed_instance_medians'],[25,75])*1000 for row in rows])
                ax.fill_between(values,intervals[:,0],intervals[:,1],color=BLUE,alpha=.13,lw=0,zorder=1)
            ax.plot(values,ys,color=COLORS[m],ls=LINES[m],lw=1.9 if m=='joint' else 1.0,
                    alpha=1 if m=='joint' else .86,zorder=4 if m=='joint' else 2)
            for x,y,row in zip(values,ys,rows):
                if not np.isfinite(y):continue
                full=row['all_repeats_completed_instances']==row['scheduled_instances']
                ax.plot(x,y,marker=MARKERS[m],ms=4.4 if m=='joint' else 3.6,mew=.85,
                        mfc=COLORS[m] if full else 'white',mec=COLORS[m],ls='none',zorder=5 if m=='joint' else 3)
            if m=='joint':
                ax.annotate(f'{ys[-1]:.2f} ms',(values[-1],ys[-1]),
                            xytext=(-3,-15),textcoords='offset points',ha='right',va='top',
                            color=BLUE,fontsize=8.2,fontweight='bold',
                            bbox={'facecolor':'white','edgecolor':'none','alpha':.87,'pad':1})
        ax.text(.5,1.07,fixed,ha='center',va='bottom',transform=ax.transAxes,fontsize=8.3)
        ax.set_xlabel({'Q':r'(a) Candidate products $Q$','B':r'(b) Access groups $B$','T':r'(c) Candidate thresholds $T$'}[axis])
    axes[0].set_ylabel('Optimization time (ms)')
    handles=[Line2D([],[],color=COLORS[m],ls=LINES[m],marker=MARKERS[m],ms=4.2,
                   lw=1.9 if m=='joint' else 1.,label='PHT-DP (ours)' if m=='joint' else LABELS[m]) for m in METHODS[:4]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,1.0),ncol=4,columnspacing=2.3)
    fig.text(.53,.02,'Blue band: PHT-DP interquartile range    |    Hollow markers: partially completed configurations',
             ha='center',fontsize=7.5,color='#52616B')
    save(fig,out,'fig6_2')

def capacity(data,out):
    fig,ax=plt.subplots(figsize=(3.4,2.50))
    fig.subplots_adjust(left=.17,right=.98,bottom=.19,top=.95)
    specs=[('Balanced',BLUE,'o','-'),('High activation cost',ORANGE,'s','--'),('High service cost',GREEN,'^','-.')]
    inset=ax.inset_axes([.48,.36,.48,.40])
    inset.set_facecolor('white'); inset.set_zorder(8)
    inset.spines['top'].set_visible(True);inset.spines['right'].set_visible(True)
    for row,(label,color,marker,line) in zip(data['capacity']['curves'],specs):
        xs=np.array(row['capacity'])[1:]; ys=100*np.array(row['meter_equal_mean_fraction'])[1:]
        lower=np.array([x['lower'] for x in row['meter_equal_mean_ci95']])[1:]*100
        upper=np.array([x['upper'] for x in row['meter_equal_mean_ci95']])[1:]*100
        ax.fill_between(xs,lower,upper,color=color,alpha=.10,linewidth=0)
        ax.plot(xs,ys,color=color,marker=marker,ls=line,label=label,lw=1.15,ms=3.7,mew=.8,
                mfc='white' if marker=='s' else color)
        ax.errorbar(xs,ys,yerr=[ys-lower,upper-ys],color=color,fmt='none',elinewidth=.65,capsize=2,capthick=.65)
        inset.plot(xs,ys,color=color,marker=marker,ls=line,lw=1.1,ms=4,mew=.85,
                   mfc='white' if marker=='s' else color)
    ax.set_xticks(range(1,7));ax.set_xlim(.8,6.2);ax.set_ylim(94,100.55)
    ax.set_yticks([94,96,98,100]);ax.grid(axis='y')
    ax.set_xlabel(r'Maximum offered products $L$')
    ax.set_ylabel(r'Net profit retained (%)')
    inset.set_xlim(1.9,3.12);inset.set_ylim(99.20,100.10)
    inset.set_xticks([2,3]);inset.set_yticks([99.3,99.6,100],['99.3','99.6','100'])
    inset.tick_params(labelsize=7.2,pad=1,length=2)
    inset.grid(axis='y',lw=.4)
    inset.set_title(r'Zoom: $L=2$--$3$',fontsize=7.8,pad=4)
    # A true zoom of the means: equal values remain superimposed.
    rect=Rectangle((1.9,99.20),1.22,.90,fill=False,edgecolor='#7D8C96',lw=.65,ls=(0,(3,2)))
    ax.add_patch(rect)
    connector=ConnectionPatch(xyA=(3.12,99.20),coordsA=ax.transData,
                              xyB=(.0,1.),coordsB=inset.transAxes,color='#8A969F',lw=.65)
    ax.add_artist(connector)
    ax.text(5.97,100.18,'Coincident at 100%',ha='right',va='bottom',fontsize=7.5,color='#52616B')
    ax.legend(loc='lower center',bbox_to_anchor=(.56,.025),fontsize=7.8,labelspacing=.3,
              ncol=1,handlelength=2.2)
    save(fig,out,'fig6_3')

def menu(example,out):
    cases={c['association']:c for c in example['cases']}
    product_rows=lambda a:sorted([p for p in cases[a]['products'] if p['offered']],key=lambda p:p['product'])
    pos=product_rows('positive');ind=product_rows('independent');neg=product_rows('negative')
    assert [p['price_exact'] for p in pos]==[p['price_exact'] for p in ind]
    # Near-square plotting areas at the unchanged single-column width.
    fig,(ax,bx)=plt.subplots(1,2,figsize=(3.4,2.50),gridspec_kw={'width_ratios':[1.0,1.0]})
    fig.subplots_adjust(left=.145,right=.985,bottom=.18,top=.83,wspace=.18)
    for rows,label,color,marker,line in [(pos,'Pos. / Indep.',BLUE,'o','-'),(neg,'Negative',GREEN,'^','--')]:
        values=[float(Fraction(p['price_exact'])) for p in rows]
        ax.plot([0,1],values,color=color,marker=marker,ls=line,lw=1.45,ms=4.3,zorder=3)
    vals_pos=[float(Fraction(p['price_exact'])) for p in pos]
    vals_neg=[float(Fraction(p['price_exact'])) for p in neg]
    ax.annotate(f'{vals_pos[0]:.3f}',(0,vals_pos[0]),xytext=(-3,8),textcoords='offset points',
                ha='left',fontsize=8,color='#46535C')
    ax.annotate(f'{vals_pos[1]:.3f}',(1,vals_pos[1]),xytext=(-1,7),textcoords='offset points',
                ha='right',fontsize=8,color=BLUE)
    ax.annotate(f'{vals_neg[1]:.3f}',(1,vals_neg[1]),xytext=(-1,-12),textcoords='offset points',
                ha='right',fontsize=8,color=GREEN)
    ax.fill_between([0,1],[vals_pos[0],vals_pos[0]],vals_neg,color=GREEN,alpha=.07,lw=0)
    ax.axhline(vals_pos[0],color='#B7C3CA',lw=.65,ls=(0,(3,2)),zorder=1)
    ax.text(.05,1.99,r'$p_4<p_2$',ha='left',fontsize=8,color=GREEN)
    ax.set_xlim(-.18,1.20);ax.set_ylim(1.55,3.06)
    ax.set_yticks([1.6,2.,2.4,2.8]);ax.grid(axis='y')
    ax.set_xticks([0,1],['$q=2$\n120 min','$q=4$\n30 min']);ax.tick_params(axis='x',length=0,pad=4,labelsize=8)
    ax.set_ylabel('Posted price')
    assoc_handles=[Line2D([],[],color=BLUE,marker='o',ms=3.5,lw=1.2,label='Pos. / Indep.'),
                   Line2D([],[],color=GREEN,marker='^',ms=3.5,lw=1.2,ls='--',label='Negative')]
    ax.legend(handles=assoc_handles,loc='lower left',bbox_to_anchor=(-.13,1.02),fontsize=8,
              handlelength=1.4,handletextpad=.4,labelspacing=.35)
    associations=['positive','independent','negative']
    for i,a in enumerate(associations):
        rows=product_rows(a);counts=[p['subscribed_buyers'] for p in rows]
        left=0
        for j,(n,col) in enumerate(zip(counts+[120-sum(counts)],[BLUE,GREEN,'#E5EAED'])):
            bx.barh(i,n,left=left,height=.50,color=col,edgecolor='white',linewidth=.6,
                    hatch='//' if j==1 else None,zorder=3)
            bx.text(left+n/2,i,str(n),ha='center',va='center',color='white' if j<2 else '#46535C',
                    fontsize=8,zorder=4)
            left+=n
        bx.text(120,i-.31,['Positive','Independent','Negative'][i],ha='right',va='bottom',
                fontsize=8,color='#46535C')
    bx.set_yticks([]);bx.set_ylim(2.55,-.85);bx.set_xlim(0,124)
    bx.set_xticks([0,60,120]);bx.grid(axis='x')
    choice_handles=[Patch(facecolor=BLUE,label='Product 2'),Patch(facecolor=GREEN,hatch='//',label='Product 4'),
                    Patch(facecolor='#E5EAED',label='No subscription')]
    products_legend=bx.legend(handles=choice_handles[:2],loc='lower left',
              bbox_to_anchor=(-.015,1.115),fontsize=8,ncol=2,
              handlelength=.9,handletextpad=.3,columnspacing=.6,borderaxespad=.1)
    bx.add_artist(products_legend)
    bx.legend(handles=choice_handles[2:],loc='lower left',bbox_to_anchor=(-.015,1.005),
              fontsize=8,handlelength=.9,handletextpad=.3,borderaxespad=.1)
    # Panel identifiers stay inside the plotting areas; descriptions are in
    # the paper caption rather than a separate bottom band.
    for panel,letter in [(ax,'(a)'),(bx,'(b)')]:
        panel.text(.025,.975,letter,transform=panel.transAxes,
                   ha='left',va='top',fontsize=8,color='#303B43')
    save(fig,out,'fig6_4')

def main(root):
    from matplotlib import font_manager
    for font in (root/'assets/plot_fonts').glob('NimbusRoman-*.otf'):
        font_manager.fontManager.addfont(str(font))
    style();source=root/'figures/review_plot_data.json';example=root/'evidence/menu_example.json'
    data=json.loads(source.read_text());ex=json.loads(example.read_text())
    out=root/'figures'
    efficiency(data,out);capacity(data,out);menu(ex,out)
    # Figure 6-2 is the accepted vector artwork; regenerate only if absent.
    if not (out/'fig6_2.pdf').exists(): scaling(data,out)
    manifest={'source_plot_data_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'source_menu_example_sha256':hashlib.sha256(example.read_bytes()).hexdigest(),
      'new_experiments':0,'new_optimizations':0,'main_figures':4,'main_panels':7,
      'original_paper_palette':['#2166AC','#B85C00','#16856B'],
      'font':'Nimbus Roman / STIX math','raster_dpi':600,'vector_format':'PDF',
      'figure_width_inches':{'fig6_1':3.4,'fig6_2':7.0,'fig6_3':3.4,'fig6_4':3.4},
      'efficiency_statistic':'per-instance median over three completed repeats',
      'capacity_plotted_L':[1,2,3,4,5,6],'capacity_L0':0,
      'scale_methods':['PHT-DP','G0','G1','G2'],'scale_MILP':'retained in comparison table, timing figure, and original supplementary figures',
      'archived_previous_figures':'supplement/previous_figures',
      'menu_price_curves_combined':'positive and independent have identical prices; choices shown separately',
      'visual_revision':6,'efficiency_distribution':'single-axis median bars and 5th--95th percentile whiskers, all methods in ms on a common log axis; completed counts shown',
      'scaling_band':'PHT-DP 25th--75th percentiles of completed instance medians',
      'capacity_inset':{'x':[1.9,3.12],'y':[99.20,100.10],'data':'unaltered mean curves; confidence intervals shown on main axes'},
      'menu_layout':'single-column horizontal two panels with near-square plotting areas; direct price and buyer count annotations',
      'menu_size_inches':[3.4,2.50],
      'menu_panel_labels':'inside top-left corners; short paper caption; no separate bottom title band',
      'menu_typography_pt':{'axis_labels':8.5,'ticks':8,'annotations':8},
      'menu_plotting_area_inches':[1.3101,1.625],
      'menu_legend_rows':2,
      'font_sizes_pt':{'axis_labels':8.5,'tick_labels':8,'small_labels':7.5},
      'table_layout':'one single-column table with vertically stacked panels (a) and (b)'}
    (out/'publication_design_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();main(a.root.resolve())
