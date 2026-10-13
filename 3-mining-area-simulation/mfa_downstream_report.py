"""Validate, compare and report complete spatial rerun outputs."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from downstream_worker import read, dump, tag, raster

METRICS = {
    'extraction_Gt': ('extraction-rasters', 'allocated-extraction-2023-2050', 1e-3),
    'area_km2': ('mining-area', 'mining-area-expansion-2050', 1.),
    'overlap': ('biodiversity-risk', 'mining-biodiversity-overlap-2050', 1.),
}


def delta(before, after):
    return dict(before=before, after=after,
                change=None if before is None else after-before,
                percent=None if before in (0, None) else (after/before-1)*100)


HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>下游重算前后对比</title><style>body{max-width:1200px;margin:32px auto;padding:0 24px;background:#f5f7f8;color:#203039;font-family:system-ui,'Microsoft YaHei',sans-serif}p{line-height:1.8}select{padding:8px;margin:4px}table{border-collapse:collapse;width:100%;background:white;margin:16px 0}th,td{padding:9px;border-bottom:1px solid #dce4e8;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#e6eef1;position:sticky;top:0}.scroll{max-height:560px;overflow:auto}img{width:100%}.note{color:#526773}h1{font-size:28px}a{color:#096779}</style>
<h1>开采面积与生物多样性：重算前后对比</h1>
<p>沿用现有空间模型及静态输入，将其需求换成本轮修正后完整重算的 MFA 结果。时期为 2023—2050 年。面积单位为 km²，开采量为 Gt，生物多样性指标为 species-range equivalents（空间重叠风险指标）。</p>
<p class="note">L/M/H 为原模型的低／中／高需求、储量和面积关系。下游需求沿用“周转和增长两部分分别取年度分位数，再相加”的定义；不能直接当作 MFA 模拟均值或全球累计量的分位数。区间为完整 27 种 D/R/A 组合的最小值至最大值。</p>
<p>本轮另有一项输入补齐假设：瑞士 1931 年混凝土比例缺失，按 1932—2000 年恒定值向前延用一年，取 72.3%。原始表格保持不变；详见上级目录的“输入补齐说明.txt”。</p>
<p id="knee"></p><p><a href="../前后结果对比.html">查看 MFA 开采量对比</a> · <a href="下游重算结果说明.txt">查看结果说明</a></p>
<img src="downstream-before-after.png" alt="主要情景开采面积和生物多样性风险的修正前后对比">
<h2>全部情景组合</h2><label>情景 <select id="scenario"><option value="">全部</option></select></label><label>β <select id="beta"><option value="">全部</option></select></label><label>指标 <select id="metric"><option value="area_km2">矿区面积（km²）</option><option value="overlap">生物多样性风险</option><option value="extraction_Gt">开采量（Gt）</option></select></label><label>组合 <select id="dra"><option value="central">M/M/M</option><option value="all">全部 D/R/A</option></select></label>
<div class="scroll"><table><thead><tr><th>情景 / β / D-R-A</th><th>修正前</th><th>修正后</th><th>变化量</th><th>变化比例</th></tr></thead><tbody id="cases"></tbody></table></div>
<h2>主要情景的国家分布</h2><p class="note">按开采所在地的国家网格统计，与 MFA 的需求经济体口径不同。按所选指标绝对变化量排序。</p><select id="countrycase"></select>
<div class="scroll"><table><thead><tr><th>国家</th><th>修正前</th><th>修正后</th><th>变化量</th><th>变化比例</th></tr></thead><tbody id="countries"></tbody></table></div>
<script>const data=__DATA__;const $=x=>document.getElementById(x);const num=x=>x===null?'无原结果':x.toLocaleString('zh-CN',{maximumFractionDigits:5});const pct=x=>x===null?'—':`${x>=0?'+':''}${x.toFixed(3)}%`;const cells=r=>`<td>${num(r.before)}</td><td>${num(r.after)}</td><td>${num(r.change)}</td><td>${pct(r.percent)}</td>`;
for(const s of [...new Set(data.cases.map(r=>r.scenario))])$('scenario').add(new Option(s,s));for(const b of [...new Set(data.cases.map(r=>r.beta))].sort((a,b)=>a-b))$('beta').add(new Option(b,b));for(const c of data.signposts)$('countrycase').add(new Option(c.label,c.tag));$('knee').textContent=`按原规则重新选取保护强度：β ${data.knee.before.beta} → ${data.knee.after.beta}。以下主图固定 β=0.7，展示相同参数下的变化。`;
function render(){const m=$('metric').value;$('cases').innerHTML=data.cases.filter(r=>(!$('scenario').value||r.scenario===$('scenario').value)&&(!$('beta').value||r.beta===Number($('beta').value))&&($('dra').value==='all'||r.demand+r.reserve+r.area==='MMM')).map(r=>`<tr><td>${r.scenario} / ${r.beta} / ${r.demand}-${r.reserve}-${r.area}</td>${cells(r[m])}</tr>`).join('');$('countries').innerHTML=data.countries.filter(r=>r.tag===$('countrycase').value).sort((a,b)=>Math.abs(b[m].change??0)-Math.abs(a[m].change??0)).map(r=>`<tr><td>${r.country}</td>${cells(r[m])}</tr>`).join('')};for(const id of ['scenario','beta','metric','dra','countrycase'])$(id).addEventListener('input',render);render();</script></html>'''


def build(base, cases):
    base=Path(base);out=base/'downstream';results=out/'staged/results';before=out/'before/results'
    manifest=read(out/'manifest.json')
    knee=read(out/'knee-comparison.json')
    signposts=[('BAU',0.,'BAU'),('CE',0.,'CE'),('BAU',.7,'CON (β=0.7)'),('CE',.7,'IAP (β=0.7)')]
    if knee['after']['beta']!=.7:
        signposts += [('BAU',knee['after']['beta'],'CON (new selected β)'),('CE',knee['after']['beta'],'IAP (new selected β)')]
    labels={(s,b):label for s,b,label in signposts}
    country_grid=raster(results/'country-rasters/country-raster-eck4-50km.tif').astype(int)
    cpv=pd.read_csv(out/'inputs/data/country-pseudo-values.csv')
    countries={int(r['value']):str(r['country']) for _,r in cpv.iterrows()}
    country_codes=[int(c) for c in np.unique(country_grid) if c>0]
    comparisons=[];country_rows=[];taxon_rows=[];max_area=0.;max_mass=0.;max_unmet=0.
    for case in cases:
        s,b,d,r,a=case
        item=dict(tag=tag(case),scenario=s,beta=b,demand=d,reserve=r,area=a)
        for phase in ['allocation','area','risk']:
            state=read(out/'state'/phase/f'{tag(case)}.json')
            if state['status']!='complete':raise ValueError(f'Incomplete downstream case: {tag(case)}')
            if phase=='area':
                max_area=max(max_area,state['max_subcell_area_km2']);max_mass=max(max_mass,state['max_mass_error_Mt'])
            if phase=='allocation':max_unmet=max(max_unmet,abs(state['unmet_Mt']))
        local={code:dict(tag=tag(case),country=countries.get(code,str(code))) for code in country_codes}
        include_countries=(s,b) in labels and (d,r,a)==('M','M','M')
        for metric,(folder,prefix,factor) in METRICS.items():
            rel=Path(folder)/f'{prefix}-{tag(case)}.tif'
            actual=raster(results/rel)
            prior=raster(before/rel) if (before/rel).exists() else None
            if prior is not None:
                with rio.open(results/rel) as n,rio.open(before/rel) as o:
                    assert (n.shape,n.crs,n.transform)==(o.shape,o.crs,o.transform),str(rel)
            item[metric]=delta(None if prior is None else float(prior.sum(dtype=float))*factor,float(actual.sum(dtype=float))*factor)
            if include_countries:
                n=np.bincount(country_grid.ravel(),weights=actual.ravel())*factor
                o=None if prior is None else np.bincount(country_grid.ravel(),weights=prior.ravel())*factor
                for code in country_codes:local[code][metric]=delta(None if o is None else float(o[code]),float(n[code]))
        if include_countries:
            country_rows += list(local.values())
            rel=Path('biodiversity-risk')/f'mining-biodiversity-overlap-2050-{tag(case)}.csv'
            new=pd.read_csv(results/rel,index_col=0)
            old=pd.read_csv(before/rel,index_col=0) if (before/rel).exists() else None
            for category in new.index:
                for taxon in new.columns:
                    taxon_rows.append(dict(tag=tag(case),category=category,taxon=taxon,**delta(None if old is None else float(old.loc[category,taxon]),float(new.loc[category,taxon]))))
        comparisons.append(item)
    lookup={(x['scenario'],x['beta'],x['demand'],x['reserve'],x['area']):x for x in comparisons}
    policy=[]
    for s,b,label in signposts:
        rec=dict(label=label,tag=tag((s,b,'M','M','M')),scenario=s,beta=b)
        ensemble=[x for x in comparisons if x['scenario']==s and x['beta']==b]
        assert len(ensemble)==27,(s,b,len(ensemble))
        for metric in METRICS:
            rec[metric]=dict(lookup[s,b,'M','M','M'][metric])
            for version in ['before','after']:
                values=[x[metric][version] for x in ensemble if x[metric][version] is not None]
                rec[metric][version+'_range']=[min(values),max(values)] if len(values)==27 else None
        policy.append(rec)
    demand=[]
    for s in manifest['scenarios']:
        for m in ['sand','gravel','limestone']:
            for level in 'LMH':
                rel=Path('extraction-rasters')/f'{m}-extraction-demand-2023-2050-{s}-50km-{level}.tif'
                demand.append(dict(scenario=s,material=m,level=level,**delta(float(raster(before/rel).sum(dtype=float))/1e9,float(raster(results/rel).sum(dtype=float))/1e9)))
    old_cost=pd.read_csv(before/'extraction-rasters/cost-analysis-DM-RM.csv')
    new_cost=pd.read_csv(results/'extraction-rasters/cost-analysis-DM-RM.csv')
    cost=[]
    for _, row in new_cost.iterrows():
        prior=old_cost[(old_cost.scenario==row.scenario)&(old_cost.beta==row.beta)].iloc[0]
        cost.append(dict(scenario=row.scenario,beta=float(row.beta),**delta(float(prior.cost_Tusd_central),float(row.cost_Tusd_central))))
    validation=dict(status='passed',cases=len(cases),fixed_comparison_cases=len(manifest['cases']),
                    demand_scenarios=len(manifest['scenarios']),max_subcell_area_km2=max_area,
                    max_fine_grid_mass_error_Mt=max_mass,max_absolute_allocation_rounding_residual_Mt=max_unmet,
                    preflight_reproduced_baseline=read(out/'preflight-validation.json'),
                    mass_nonnegative_capacity_and_risk_checks=True)
    if max_area>1+1e-9 or max_mass>1e-5:raise ValueError('Downstream physical validation failed')
    dump(out/'validation.json',validation)
    payload=dict(cases=comparisons,countries=country_rows,signposts=policy,knee=knee)
    dump(out/'spatial-comparison.json',payload)
    dump(out/'demand-comparison.json',demand)
    dump(out/'taxon-comparison.json',taxon_rows)
    dump(out/'cost-comparison.json',cost)
    # Machine-readable analysis tables, without changing source-data workbooks.
    rows=[]
    for x in comparisons:
        for metric in METRICS:rows.append(dict(tag=x['tag'],metric=metric,**x[metric]))
    pd.DataFrame(rows).to_csv(out/'downstream-before-after.csv',index=False,encoding='utf-8-sig')
    selected_rows=[]
    for x in comparisons:
        if x['scenario'] in ['BAU','CE'] and x['beta']==knee['after']['beta']:
            selected_rows.append(dict(scenario='CON' if x['scenario']=='BAU' else 'IAP',beta=x['beta'],demand=x['demand'],reserve=x['reserve'],area=x['area'],overlap=x['overlap']['after']))
    pd.DataFrame(selected_rows).to_csv(results/'biodiversity-risk/biodiversity-risk-summary-selected-signposts-DRA-factorial.csv',index=False)
    dump(out/'selected-signposts.json',dict(beta=knee['after']['beta'],policies=policy))
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),constrained_layout=True)
    x=np.arange(4);width=.34
    for ax,metric,label in zip(axes,['area_km2','overlap'],['Mining area (km²)','Mining–biodiversity overlap (species-range equivalents)']):
        old=[p[metric]['before'] for p in policy[:4]];new=[p[metric]['after'] for p in policy[:4]]
        ax.bar(x-width/2,old,width,color='#a6b0b9',label='Before')
        ax.bar(x+width/2,new,width,color='#087f8c',label='Corrected MFA')
        ax.set_xticks(x,['BAU','CE','CON (β=0.7)','IAP (β=0.7)']);ax.set_ylabel(label)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False)
    fig.savefig(out/'downstream-before-after.png',dpi=180);plt.close(fig)
    text=['下游全量重算与前后对比','',f"范围：17 个空间需求情景，{len(cases)} 组 D/R/A/β 组合。",
          '比较基准：本轮 MFA 全量重算开始后、下游重算开始前保存的正式下游结果。',
          '静态地理、储量、适宜性、物种生境和面积关系不变；新需求来自本轮完整 MFA 输出。',
          'MFA 输入补齐假设：瑞士1931年混凝土比例缺失，按1932—2000年恒定值72.3%向前延用一年。原始工作簿未改，具体记录见上级目录 input-adjustments.json。',
          '空间层沿用 SSP2 和既有 17 个情景；其余 SSP 与 wedge 情景仍按既有流程仅在 MFA 层计算。',
          '下游需求取周转、增长两部分的年度分位数分别相加，不等于 MFA 模拟均值。',
          '面积／风险区间为全 27 种组合的范围，不是蒙特卡洛置信区间。','',
          f"保护强度选点（原规则和 0.001 容差不变）：β {knee['before']['beta']} → {knee['after']['beta']}。",
          '以下比较固定各自 β 参数；若选点改变，另列新选点结果。','']
    for p in policy:
        text.append(p['label'])
        for metric,label in [('extraction_Gt','开采量 Gt'),('area_km2','开采面积 km²'),('overlap','生物多样性重叠风险')]:
            row=p[metric];old='无原结果' if row['before'] is None else f"{row['before']:.6f}"
            change='无可比基准' if row['percent'] is None else f"{row['percent']:+.3f}%"
            limits=row['after_range']
            text.append(f"  {label}：{old} → {row['after']:.6f}（{change}）；新范围 {limits[0]:.6f}–{limits[1]:.6f}")
    text += ['',f'验证：单个 1 km 网格面积最大 {max_area:.12f} km²；细网格质量最大误差 {max_mass:.3g} Mt。',
             '下游前后结果对比.html 可筛选所有组合，并查看主要情景的国家分布。',
             'spatial-comparison.json 包含全球及国家对比；taxon-comparison.json 包含类群与保护等级对比；cost-comparison.json 包含运输成本对比。',
             '旧结果完整保存在 before/results；正式结果仅在全部核验通过后安装。',
             '本轮产出为模型结果与比较报告；投稿文稿、原论文图件和上传数据包仍是重算前版本。']
    (out/'下游重算结果说明.txt').write_text('\n'.join(text),encoding='utf-8-sig')
    (out/'下游前后结果对比.html').write_text(HTML.replace('__DATA__',json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/')),encoding='utf-8')
    mfa=read(base/'global-comparison.json')
    headline=['SSP2 及下游重算结果','', '本报告在 SSP2 及下游完成后生成；其他 SSP 与最终汇总表的完成状态见“运行状态.txt”。', '', 'MFA 累计开采量（三种材料合计，模拟均值，Gt）：']
    for s in ['historical','BAU','CE']:
        row=next(r for r in mfa if r['scenario']==s and r['material']=='total')
        headline.append(f"{s}: {row['before_cumulative_mean_Gt']:.3f} → {row['after_cumulative_mean_Gt']:.3f}（{row['change_percent']:+.3f}%）")
    (base/'SSP2及下游重算结果.txt').write_text('\n'.join(headline+['','']+text),encoding='utf-8-sig')


if __name__=='__main__':
    base=Path(sys.argv[1]).resolve()
    build(base,read(base/'downstream/computed-cases.json'))
