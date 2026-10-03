#!/usr/bin/env python3
"""Create Fig. 5 execution-time and speedup plots from accepted runs only."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import statistics

def read(path):
    with path.open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv); src=a.input/'fig05_accepted_runs.csv'; a.output.mkdir(parents=True,exist_ok=False)
    rows=read(src) if src.is_file() else []
    rows=[r for r in rows if r.get('accepted','').lower() in ('true','1') and r.get('status')=='OK']
    groups={}
    for r in rows:
        key=(r.get('fig05_comparison'),r.get('tile_shape'),r.get('unroll'),r.get('profile_pass','no'))
        groups.setdefault(key,[]).append(r)
    summary=[]
    for (comparison,tile,u,profile),items in sorted(groups.items()):
        ts=[num(r.get('total_sec')) for r in items]; ts=[x for x in ts if x is not None]
        if not ts: continue
        summary.append({'comparison':comparison,'tile_shape':tile,'unroll':u,'profile_pass':profile,
                        'n':len(ts),'time_mean_sec':statistics.fmean(ts),
                        'time_sd_sec':statistics.stdev(ts) if len(ts)>1 else 0.0})
    with (a.output/'fig05_matched_summary.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['comparison','tile_shape','unroll','profile_pass','n','time_mean_sec','time_sd_sec']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)
    notices=[]
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except ImportError:
        notices.append('Matplotlib unavailable; CSV summary written, no plots generated.')
    else:
        for profile in sorted({r['profile_pass'] for r in summary}):
            use=[r for r in summary if r['profile_pass']==profile]
            if not use: continue
            labels=[f"{r['tile_shape']} U{r['unroll']}" for r in use]
            rvv={ (r['tile_shape'],r['unroll']):r for r in use if r['comparison']=='RVV_8CORE'}
            targets=[r for r in use if r['comparison']!='RVV_8CORE']
            fig,ax=plt.subplots(figsize=(max(8,0.75*len(use)),4.5)); x=range(len(use))
            ax.bar(list(x),[r['time_mean_sec'] for r in use],yerr=[r['time_sd_sec'] for r in use],capsize=3,color=['#0072B2' if r['comparison']=='RVV_8CORE' else '#D55E00' for r in use])
            ax.set_xticks(list(x),[f"{r['comparison']}\n{l}" for r,l in zip(use,labels)],rotation=55,ha='right'); ax.set_ylabel('Mean total time (s)'); ax.set_title(f'Fig. 5: end-to-end execution ({profile})'); fig.tight_layout(); fig.savefig(a.output/f'fig05_{profile}_execution.pdf'); fig.savefig(a.output/f'fig05_{profile}_execution.png',dpi=300); plt.close(fig)
            speed=[]; sl=[]
            for r in targets:
                b=rvv.get((r['tile_shape'],r['unroll']))
                if b and r['time_mean_sec']:
                    speed.append(b['time_mean_sec']/r['time_mean_sec']); sl.append(f"{r['comparison']}\n{r['tile_shape']} U{r['unroll']}")
            if speed:
                fig,ax=plt.subplots(figsize=(max(7,0.8*len(speed)),4)); ax.bar(range(len(speed)),speed,color='#009E73'); ax.axhline(1,color='black',lw=.7); ax.set_xticks(range(len(speed)),sl,rotation=55,ha='right'); ax.set_ylabel('Speedup over matched 8-core RVV'); ax.set_title(f'Fig. 5: matched speedups ({profile})'); fig.tight_layout(); fig.savefig(a.output/f'fig05_{profile}_speedup.pdf'); fig.savefig(a.output/f'fig05_{profile}_speedup.png',dpi=300); plt.close(fig)
            else: notices.append(f'No matched speedups for {profile}.')
    (a.output/'fig05_plot_report.json').write_text(json.dumps({'accepted_rows':len(rows),'summaries':len(summary),'notices':notices},indent=2)+'\n',encoding='utf-8')
    print(f'Accepted rows: {len(rows)}; summaries: {len(summary)}')
    for n in notices: print('NOTICE: '+n)
    return 0
if __name__=='__main__': raise SystemExit(main())
