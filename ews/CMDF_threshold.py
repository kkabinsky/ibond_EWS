import warnings; warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

# ====================================================================
# CONFIG
# ====================================================================
DATA   = r'D:\tadgan_gaf\dataset_bond\Rev01_Database_final.dta'
TARGET = 'ln_pd12m'
OUTDIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
def out(name): return os.path.join(OUTDIR, name)
ANALYSIS_START = 2010                 # primary window; 2001 for full robustness
N_BOOT = 120
BASE_SEED = 42
EXPAND_FIRST_CUT = 2013
EXPAND_STEP = 2
MIN_OBS, MIN_FIRMS = 200, 8

RUN_STEPS = {
    'model_comparison': True,
    'shap_plots':       True,
    'thresholds':       True,   
    'stability':        True,
    'pd_translation':   True,
    'risk_zones':       True,
    'annex_figures':    True,
}

XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1.0, n_jobs=-1, random_state=42)

FEATURES = [
    'lnTotalAssets','AgeYear','ROA','ROE','EBITtoTA','REtoTA','DE_w','TDTA',
    'LTDtoTA','STDtoTA','CurrentRatio','QuickRatio','CashRatio','WorkingCapitaltoTA_w',
    'cf_Interestcoverageratio_w','cf_DebtServiceCoverageRatio_w','amihud_monthly_100_w',
    'adj_illiq_kz_w','ln_amihud','Policyrate','GDPgrowth','Unemploymentratenationalesti',
]
MACRO_SKIP = {'L_Policyrate','L_GDPgrowth','L_Unemploymentratenationalesti'}

KINK_VARS = {
    'L_ROA':'Profitability: ROA','L_DE_w':'Leverage: D/E','L_REtoTA':'Profitability: RE/TA',
    'L_CurrentRatio':'Liquidity: Current Ratio','L_QuickRatio':'Liquidity: Quick Ratio',
    'L_CashRatio':'Liquidity: Cash Ratio','L_amihud_monthly_100_w':'Illiquidity: Amihud (x100)',
    'L_adj_illiq_kz_w':'Illiquidity: Kang-Zhang','L_ln_amihud':'Illiquidity: ln(Amihud)',
    'L_cf_DebtServiceCoverageRatio_w':'Coverage: DSCR','L_cf_Interestcoverageratio_w':'Coverage: Interest Coverage',
}
ROBUST_ZONE = {'L_DE_w':'D/E','L_CurrentRatio':'Current Ratio','L_CashRatio':'Cash Ratio',
               'L_ln_amihud':'ln(Amihud)','L_adj_illiq_kz_w':'Kang-Zhang','L_REtoTA':'RE/TA'}
# determinants shown in the paper figure (module level: also used to trim the
# SHAP cache in step 4 so memory stays bounded)
PANELS=[('L_DE_w','Leverage (D/E)'),('L_ROA','Profitability (ROA)'),
        ('L_ln_amihud','Market illiquidity'),('L_CurrentRatio','Current Ratio'),
        ('L_REtoTA','RE/TA'),('L_cf_DebtServiceCoverageRatio_w','DSCR')]

PRETTY = {'L_lnTotalAssets':'ln(Assets)','L_AgeYear':'Firm Age','L_ROA':'ROA','L_ROE':'ROE',
 'L_EBITtoTA':'EBIT/TA','L_REtoTA':'RE/TA','L_DE_w':'D/E','L_TDTA':'Total Debt/TA',
 'L_LTDtoTA':'LT Debt/TA','L_STDtoTA':'ST Debt/TA','L_CurrentRatio':'Current Ratio',
 'L_QuickRatio':'Quick Ratio','L_CashRatio':'Cash Ratio','L_WorkingCapitaltoTA_w':'WC/TA',
 'L_cf_Interestcoverageratio_w':'Interest Coverage','L_cf_DebtServiceCoverageRatio_w':'DSCR',
 'L_amihud_monthly_100_w':'Amihud','L_adj_illiq_kz_w':'Kang-Zhang','L_ln_amihud':'ln(Amihud)',
 'L_Policyrate':'Policy Rate','L_GDPgrowth':'GDP Growth','L_Unemploymentratenationalesti':'Unemployment'}

NAVY='#1f3a5f'; RUST='#a8501a'; INK='#1a1a1a'; GRID='#d8d8d8'
GREEN='#2e7d4f'; AMBER='#e0a52e'; RED='#c0392b'


# ====================================================================
# SHARED HELPERS
# ====================================================================
def build_sample(df_full, flag, add_esg=False, start=ANALYSIS_START):
    col_map = {
        'DE': 'DE_w',
        'WorkingCapitaltoTA': 'WorkingCapitaltoTA_w',
        'cf_Interestcoverageratio': 'cf_Interestcoverageratio_w',
        'cf_DebtServiceCoverageRatio': 'cf_DebtServiceCoverageRatio_w',
        'amihud_monthly_100': 'amihud_monthly_100_w',
        'adj_illiq_kz': 'adj_illiq_kz_w'
    }
    for orig, new in col_map.items():
        if orig in df_full.columns and new not in df_full.columns:
            df_full[new] = df_full[orig]
    if flag not in df_full.columns:
        df_full[flag] = 1
    df = df_full.sort_values(['firm_id','month_year']).copy()
    feats = FEATURES + (['ESGScore'] if add_esg else [])
    lagged = {f'L_{v}': df.groupby('firm_id', observed=True)[v].shift(1) for v in feats}
    df = pd.concat([df, pd.DataFrame(lagged, index=df.index)], axis=1)
    df = df[df[flag] == 1].copy()
    df['year'] = df['year'].astype(int)
    df = df[df['year'] >= start].copy()
    lag = [f'L_{v}' for v in feats]
    df = df[lag + [TARGET,'firm_id','year','month_year']].dropna(subset=lag+[TARGET]).reset_index(drop=True)
    return df, lag


def fit_segmented(x, sv, cand):
    best=None; ones=np.ones(len(x)); ss=float(sv@sv)
    for c in cand:
        h=np.clip(x-c,0,None); A=np.column_stack([ones,x,h]); AtA=A.T@A; Atb=A.T@sv
        try: b=np.linalg.solve(AtA,Atb)
        except np.linalg.LinAlgError: b=np.linalg.lstsq(A,sv,rcond=None)[0]
        sse=ss-2*float(b@Atb)+float(b@AtA@b)
        if best is None or sse<best['sse']:
            best=dict(c=float(c),b1=float(b[1]),b2=float(b[2]),sse=sse)
    return best


def two_stage_candidates(x):
    coarse=np.unique(np.quantile(x,np.linspace(0.05,0.95,25)))
    return coarse


def threshold_from(x_all, sv_all, grp, seed):
    rng=np.random.default_rng(seed)
    lo,hi=np.nanpercentile(x_all,[1,99])
    m=(x_all>=lo)&(x_all<=hi)&np.isfinite(x_all)&np.isfinite(sv_all)
    x,sv,gg=x_all[m],sv_all[m],grp[m]
    if len(x)<MIN_OBS or np.unique(gg).size<MIN_FIRMS: return None
    edge=bool(np.mean(np.isclose(x,np.min(x),atol=1e-9))>0.10)
    coarse=two_stage_candidates(x); b0=fit_segmented(x,sv,coarse)
    span=(coarse[1]-coarse[0]) if len(coarse)>1 else 1.0
    cand=np.unique(np.concatenate([coarse,np.linspace(b0['c']-span,b0['c']+span,15)]))
    base=fit_segmented(x,sv,cand)

    # The SSE surface is often nearly flat near the optimum, so the single
    # best-fitting candidate can jump between grid choices even when the
    # underlying kink is unchanged. Report the median breakpoint across several
    # grid resolutions, which is stable across reruns, and record how flat the
    # surface is so unstable determinants can be flagged rather than trusted.
    grid_bps=[]
    for ng in (15,25,35,45):
        gc_=np.unique(np.quantile(x,np.linspace(0.05,0.95,ng)))
        gb=fit_segmented(x,sv,gc_)
        sp_=(gc_[1]-gc_[0]) if len(gc_)>1 else 1.0
        gcand=np.unique(np.concatenate([gc_,np.linspace(gb['c']-sp_,gb['c']+sp_,15)]))
        grid_bps.append(fit_segmented(x,sv,gcand)['c'])
    bp_robust=float(np.median(grid_bps))
    bp_grid_spread=float(np.max(grid_bps)-np.min(grid_bps))
    # refit slopes at the reported breakpoint so they stay consistent with it
    base=fit_segmented(x,sv,[bp_robust])
    # Firm-clustered bootstrap. Index arrays are built with a preallocated
    # buffer and released each iteration to keep peak memory low on large panels.
    firms=np.unique(gg)
    order_by_firm=np.argsort(gg,kind='stable')
    gg_sorted=gg[order_by_firm]
    starts=np.searchsorted(gg_sorted,firms,side='left')
    ends=np.searchsorted(gg_sorted,firms,side='right')
    counts=ends-starts
    boot=[]
    for _ in range(N_BOOT):
        pick=rng.integers(0,len(firms),size=len(firms))
        total=int(counts[pick].sum())
        idx=np.empty(total,dtype=np.int64); pos=0
        for f in pick:
            n=counts[f]
            idx[pos:pos+n]=order_by_firm[starts[f]:ends[f]]; pos+=n
        boot.append(fit_segmented(x[idx],sv[idx],cand)['c'])
        del idx
    ci=np.percentile(boot,[2.5,97.5])
    rng_x=float(np.nanmax(x)-np.nanmin(x)) or 1.0
    return dict(breakpoint=bp_robust,ci_low=float(ci[0]),ci_high=float(ci[1]),
                slope_before=base['b1'],slope_after=base['b1']+base['b2'],
                n_obs=int(len(x)),edge_concentrated=edge,
                bp_grid_spread=round(bp_grid_spread,4),
                grid_stable=bool(bp_grid_spread<=0.05*rng_x),
                kink_direction='upward' if base['b2']>0 else 'downward')


def binned(x, sv, n=30):
    s=pd.DataFrame({'x':x,'sv':sv}).dropna(); s=s[np.isfinite(s['x'])]
    try: s['b']=pd.qcut(s['x'],q=n,duplicates='drop')
    except Exception: return None,None
    g=s.groupby('b',observed=True).agg(x=('x','median'),sv=('sv','median')).reset_index(drop=True)
    return g['x'].values,g['sv'].values


# ====================================================================
# STEP 2 — MODEL COMPARISON
# ====================================================================
def step_model_comparison(df_full):
    print('\n[STEP 2] Model comparison (expanding window)')
    def models():
        return {
            'Ridge (linear)': lambda: Ridge(alpha=1.0),
            'Random Forest' : lambda: RandomForestRegressor(n_estimators=120,max_depth=10,
                                min_samples_leaf=30,n_jobs=-1,random_state=42),
            'XGBoost'       : lambda: xgb.XGBRegressor(**XGB_PARAMS),
            'LightGBM'      : lambda: lgb.LGBMRegressor(n_estimators=400,max_depth=4,learning_rate=0.05,
                                subsample=0.8,colsample_bytree=0.8,min_child_samples=20,reg_lambda=1.0,
                                n_jobs=-1,random_state=42,verbose=-1),
        }
    rows=[]
    for label,flag,esg in [('Expanded','sample_noESG',False),('ESG','sample',True)]:
        df,lag=build_sample(df_full,flag,esg)
        X=df[lag].reset_index(drop=True); y=df[TARGET].reset_index(drop=True); yr=df['year'].reset_index(drop=True)
        cuts=[c for c in range(EXPAND_FIRST_CUT,int(yr.max()),3)]
        for name,ctor in models().items():
            r2,rm,ma,sp=[],[],[],[]
            for t in cuts:
                tr=(yr<=t).values; te=((yr>t)&(yr<=t+3)).values
                if te.sum()<100 or tr.sum()<500: continue
                Xtr,Xte=X[tr],X[te]
                if name=='Ridge (linear)':
                    sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
                m=ctor(); m.fit(Xtr,y[tr]); p=m.predict(Xte)
                r2.append(r2_score(y[te],p)); rm.append(np.sqrt(mean_squared_error(y[te],p)))
                ma.append(mean_absolute_error(y[te],p)); sp.append(spearmanr(y[te],p).correlation)
            rows.append(dict(sample=label,model=name,R2_median=float(np.median(r2)),
                RMSE=float(np.mean(rm)),MAE=float(np.mean(ma)),Spearman=float(np.nanmean(sp))))
            print(f'  {label:9s} {name:16s} R2={np.median(r2):.3f} Spearman={np.nanmean(sp):.3f}')
    pd.DataFrame(rows).to_csv(out('model_comparison.csv'),index=False)
    print('model_comparison.csv')


# ====================================================================
# STEP 3 — SHAP GLOBAL + LOCAL
# ====================================================================
def step_shap_plots(df_full):
    print('\n[STEP 3] SHAP global + local')
    df,lag=build_sample(df_full,'sample_noESG',False)
    X=df[lag]; y=df[TARGET]
    m=xgb.XGBRegressor(**XGB_PARAMS); m.fit(X,y)
    expl=shap.TreeExplainer(m); sv=expl.shap_values(X)
    names=[PRETTY.get(c,c) for c in lag]

    abs_sv=np.abs(sv); mabs=abs_sv.mean(0)
    firms=df['firm_id'].values; uniq=np.unique(firms)
    fidx={f:np.where(firms==f)[0] for f in uniq}; rng=np.random.default_rng(42)
    boot=np.zeros((200,abs_sv.shape[1]))
    for b in range(200):
        s=rng.choice(uniq,size=len(uniq),replace=True)
        boot[b]=abs_sv[np.concatenate([fidx[f] for f in s])].mean(0)
    ci_lo=np.percentile(boot,2.5,0); ci_hi=np.percentile(boot,97.5,0)
    order=np.argsort(mabs)[::-1]

    fig,ax=plt.subplots(figsize=(8.5,7.5)); yy=np.arange(len(order)); vals=mabs[order]
    ax.barh(yy,vals,color='#1f5f8b',xerr=[vals-ci_lo[order],ci_hi[order]-vals],
            error_kw=dict(ecolor='#333',lw=1,capsize=3))
    ax.set_yticks(yy); ax.set_yticklabels([names[i] for i in order]); ax.invert_yaxis()
    ax.set_xlabel('mean |SHAP| \u00b1 firm-clustered bootstrap 95% CI')
    ax.set_title('Global determinant importance (mean |SHAP| with 95% CI)',fontweight='bold')
    ax.grid(axis='x',color='#ddd',lw=0.5); plt.tight_layout()
    plt.savefig(out('SHAP_global_bar.png'),dpi=180,bbox_inches='tight'); plt.close()

    plt.figure()
    shap.plots.beeswarm(shap.Explanation(values=sv,data=X.values,feature_names=names),
                        max_display=len(names),show=False)
    plt.title('Global SHAP summary (beeswarm, all determinants)',fontweight='bold')
    plt.tight_layout(); plt.savefig(out('SHAP_global_beeswarm.png'),dpi=180,bbox_inches='tight'); plt.close()

    pred=m.predict(X)
    picks={'high':np.argsort(pred)[-int(len(pred)*0.01)],'median':np.argsort(pred)[len(pred)//2],
           'low':np.argsort(pred)[int(len(pred)*0.01)]}
    for tag,i in picks.items():
        plt.figure()
        shap.plots.waterfall(shap.Explanation(values=sv[i],base_values=expl.expected_value,
                             data=X.iloc[i].values,feature_names=names),max_display=12,show=False)
        plt.title(f'Local explanation — {tag} risk (firm={df["firm_id"].iloc[i]}, '
                  f'year={df["year"].iloc[i]})',fontsize=10,fontweight='bold')
        plt.tight_layout(); plt.savefig(out(f'SHAP_local_{tag}.png'),dpi=170,bbox_inches='tight'); plt.close()

    pd.DataFrame({'determinant':[names[i] for i in order],'mean_abs_shap':mabs[order],
                  'ci_low':ci_lo[order],'ci_high':ci_hi[order]}).to_csv(out('shap_global_importance.csv'),index=False)

    # --- appendix grid: SHAP dependence for ALL determinants ---
    ncol=3; nrow=int(np.ceil(len(lag)/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(4.2*ncol,3.0*nrow))
    axes=np.atleast_1d(axes).ravel()
    for k,col in enumerate(lag):
        ax=axes[k]; xv=X[col].values.astype(float); svv=sv[:,k]
        ok=np.isfinite(xv)&np.isfinite(svv); xv,svv=xv[ok],svv[ok]
        if len(xv)==0: ax.set_visible(False); continue
        lo,hi=np.nanpercentile(xv,[1,99]); mk=(xv>=lo)&(xv<=hi)
        ax.scatter(xv[mk],svv[mk],s=4,alpha=0.12,color=NAVY,edgecolors='none',rasterized=True)
        bx,bsv=binned(xv,svv)
        if bx is not None: ax.plot(bx,bsv,color=INK,lw=1.6)
        ax.axhline(0,color='gray',lw=0.6,ls='--')
        ax.set_title(PRETTY.get(col,col),fontsize=9,fontweight='bold')
        ax.tick_params(labelsize=7); ax.grid(True,color=GRID,lw=0.4,alpha=0.5); ax.set_axisbelow(True)
    for k in range(len(lag),len(axes)): axes[k].set_visible(False)
    fig.suptitle('SHAP dependence — all determinants (Expanded sample)',fontsize=13,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.985])
    plt.savefig(out('shap_dependence_all.png'),dpi=110,bbox_inches='tight'); plt.close()
    print('  -> SHAP_global_bar.png, SHAP_global_beeswarm.png, SHAP_local_*.png, '
          'shap_dependence_all.png, shap_global_importance.csv')


# ====================================================================
# STEP 4 — THRESHOLDS (both samples) + FIGURE + ALE
# ====================================================================
def fit_oof_shap_expanding(df, lag):
    """Expanding-window out-of-sample SHAP + performance metrics."""
    X=df[lag].reset_index(drop=True); y=df[TARGET].reset_index(drop=True)
    g=df['firm_id'].reset_index(drop=True); yr=df['year'].reset_index(drop=True)
    S=np.full((len(X),len(lag)),np.nan)
    r2,rm,ma,sp,lin=[],[],[],[],[]
    for t in range(EXPAND_FIRST_CUT,int(yr.max()),EXPAND_STEP):
        tr=(yr<=t).values; te=((yr>t)&(yr<=t+EXPAND_STEP)).values
        if te.sum()<100 or tr.sum()<500: continue
        m=xgb.XGBRegressor(**XGB_PARAMS); m.fit(X[tr],y[tr]); p=m.predict(X[te])
        r2.append(r2_score(y[te],p)); rm.append(np.sqrt(mean_squared_error(y[te],p)))
        ma.append(mean_absolute_error(y[te],p)); sp.append(spearmanr(y[te],p).correlation)
        sc=StandardScaler().fit(X[tr]); lm=Ridge(alpha=1.0).fit(sc.transform(X[tr]),y[tr])
        lin.append(r2_score(y[te],lm.predict(sc.transform(X[te]))))
        S[te]=shap.TreeExplainer(m).shap_values(X[te])
    met=dict(xgb_r2=float(np.mean(r2)),xgb_rmse=float(np.mean(rm)),xgb_mae=float(np.mean(ma)),
             xgb_spearman=float(np.nanmean(sp)),lin_r2_median=float(np.median(lin)),
             oos_coverage=float(np.mean(~np.isnan(S[:,0]))))
    return X,S,g,met


def ale_1d(model,X,col,n_bins=40):
    x=X[col].values; q=np.unique(np.quantile(x,np.linspace(0,1,n_bins+1)))
    if len(q)<3: return None,None
    idx=np.clip(np.digitize(x,q[1:-1]),0,len(q)-2); eff=np.zeros(len(q)-1)
    for b in range(len(q)-1):
        mk=idx==b
        if mk.sum()<5: continue
        Xlo=X[mk].copy(); Xhi=X[mk].copy(); Xlo[col]=q[b]; Xhi[col]=q[b+1]
        eff[b]=np.mean(model.predict(Xhi.values)-model.predict(Xlo.values))
    acc=np.cumsum(eff); centers=(q[:-1]+q[1:])/2
    w=np.array([(idx==b).sum() for b in range(len(q)-1)],float)
    acc=acc-np.average(acc,weights=w if w.sum()>0 else None)
    return centers,acc


def step_thresholds(df_full):
    print('\n[STEP 4] Threshold discovery + performance + ALE')
    cache={}; all_rows=[]; perf=[]
    for label,flag,esg in [('Expanded','sample_noESG',False),('ESG','sample',True)]:
        df,lag=build_sample(df_full,flag,esg)
        X,S,g,met=fit_oof_shap_expanding(df,lag)
        perf.append(dict(sample=label,**met))
        print(f'  {label}: XGB R2={met["xgb_r2"]:.3f} Spearman={met["xgb_spearman"]:.3f} '
              f'Linear R2(med)={met["lin_r2_median"]:.3f} cov={met["oos_coverage"]:.0%}')
        for k,(col,name) in enumerate(KINK_VARS.items()):
            if col not in lag or col in MACRO_SKIP: continue
            r=threshold_from(X[col].values.astype(float),S[:,lag.index(col)].astype(float),
                             g.values,seed=BASE_SEED+k)
            if r: all_rows.append(dict(determinant=name,column=col,sample=label,**r))
        # Keep only the columns the paper figure plots, not the full SHAP matrix.
        # Holding both samples' full matrices is what exhausts memory on large panels.
        panel_cols=[c for c,_ in PANELS if c in lag]
        cache[label]=(X[panel_cols].copy(),
                      S[:,[lag.index(c) for c in panel_cols]].copy(),
                      panel_cols,g)
        del X,S; import gc; gc.collect()
    thr=pd.DataFrame(all_rows)
    thr.to_csv(out('threshold_table.csv'),index=False)
    pd.DataFrame(perf).to_csv(out('performance.csv'),index=False)

    # paper figure
    Xx,Sx,colx,_=cache['Expanded']; Xe,Se,cole,_=cache['ESG']
    fig,axes=plt.subplots(len(PANELS),2,figsize=(11,2.7*len(PANELS)))
    for rr,(col,title) in enumerate(PANELS):
        for cc,(X,S,cols,lbl,color) in enumerate([(Xx,Sx,colx,'Expanded',NAVY),(Xe,Se,cole,'ESG',RUST)]):
            ax=axes[rr,cc]
            if col not in cols: ax.set_visible(False); continue
            j=cols.index(col); xv=X[col].values.astype(float); svv=S[:,j]
            ok=np.isfinite(xv)&np.isfinite(svv); xv,svv=xv[ok],svv[ok]
            if len(xv)==0: ax.set_visible(False); continue
            lo,hi=np.nanpercentile(xv,[2,98]); mk=(xv>=lo)&(xv<=hi)
            ax.scatter(xv[mk],svv[mk],s=5,alpha=0.10,color=color,edgecolors='none',rasterized=True)
            bx,bsv=binned(xv,svv)
            if bx is not None: ax.plot(bx,bsv,color=INK,lw=1.8)
            row=thr[(thr.column==col)&(thr['sample']==lbl)]
            if len(row):
                bp=row['breakpoint'].iloc[0]
                if lo<=bp<=hi: ax.axvline(bp,color=color,lw=1.5,ls='--')
            ax.axhline(0,color='gray',lw=0.7,ls=':'); ax.grid(True,color=GRID,lw=0.5,alpha=0.6)
            ax.set_title(f'{title} — {lbl}',fontsize=9); ax.tick_params(labelsize=7)
    plt.tight_layout(); plt.savefig(out('Figure_thresholds.png'),dpi=180,bbox_inches='tight'); plt.close()

    # ALE validation — report grid-robust breakpoints (median over several bin counts)
    df,lag=build_sample(df_full,'sample_noESG',False)
    m=xgb.XGBRegressor(**XGB_PARAMS); m.fit(df[lag],df[TARGET])
    def ale_bp_robust(col):
        bps=[]
        for nb in (20,30,40,50):
            cen,acc=ale_1d(m,df[lag],col,n_bins=nb)
            if cen is None: continue
            lo,hi=np.percentile(cen,[5,95]); mk=(cen>=lo)&(cen<=hi)
            if mk.sum()<6: continue
            cand=np.quantile(cen[mk],np.linspace(0.1,0.9,20))
            bps.append(fit_segmented(cen[mk],acc[mk],cand)['c'])
        return float(np.median(bps)) if bps else np.nan
    ale_rows=[]
    for col in ROBUST_ZONE:
        if col not in lag: continue
        ale_bp=ale_bp_robust(col)
        sub=thr[(thr.column==col)&(thr['sample']=='Expanded')]
        shap_bp=float(sub['breakpoint'].iloc[0]) if len(sub) else np.nan
        ale_rows.append(dict(determinant=PRETTY.get(col,col),column=col,
            ale_bp=round(ale_bp,4),shap_bp=round(shap_bp,4),
            abs_diff=round(abs(ale_bp-shap_bp),4) if not np.isnan(ale_bp) else None))
    pd.DataFrame(ale_rows).to_csv(out('ale_validation.csv'),index=False)
    print('threshold_table.csv, performance.csv, ale_validation.csv, Figure_thresholds.png')
    return cache, thr


# ====================================================================
# STEP 5 — TEMPORAL STABILITY
# ====================================================================
def step_stability(df_full):
    print('\n[STEP 5] Temporal stability (2010-17 vs 2018-26)')
    df,lag=build_sample(df_full,'sample_noESG',False)
    rows=[]
    for era,(a,b) in [('2010-2017',(2010,2017)),('2018-2026',(2018,2026))]:
        sub=df[(df.year>=a)&(df.year<=b)]
        if len(sub)<500: continue
        m=xgb.XGBRegressor(**XGB_PARAMS); m.fit(sub[lag],sub[TARGET])
        S=shap.TreeExplainer(m).shap_values(sub[lag])
        for col in ROBUST_ZONE:
            if col not in lag: continue
            r=threshold_from(sub[col].values.astype(float),S[:,lag.index(col)],
                             sub['firm_id'].values,seed=7)
            if r: rows.append(dict(determinant=PRETTY.get(col,col),era=era,breakpoint=round(r['breakpoint'],3)))
    piv=pd.DataFrame(rows).pivot_table(index='determinant',columns='era',values='breakpoint')
    piv.to_csv(out('stability_era.csv'))
    print('stability_era.csv')


# ====================================================================
# STEP 6 — PD TRANSLATION
# ====================================================================
def step_pd_translation(df_full, thr):
    print('\n[STEP 6] Threshold -> PD translation')
    df,lag=build_sample(df_full,'sample_noESG',False)
    m=xgb.XGBRegressor(**XGB_PARAMS); m.fit(df[lag],df[TARGET]); med=df[lag].median()
    rows=[]
    for _,r in thr[thr['sample']=='Expanded'].iterrows():
        col=r['column']; bp=r['breakpoint']
        if col not in lag: continue
        p05,p95=np.nanpercentile(df[col],[5,95])
        # below = midpoint of [p05, bp] ; above = midpoint of [bp, p95]
        # both clipped to the observed range so we never extrapolate past the data
        below=max(p05,(p05+bp)/2) if bp>p05 else p05
        above=min(p95,(bp+p95)/2) if bp<p95 else p95
        # flag if breakpoint sits outside the central data range (translation unreliable)
        bp_in_range=bool(p05<=bp<=p95)
        def pd_at(v): row=med.copy(); row[col]=v; return float(np.exp(m.predict(row.values.reshape(1,-1))[0]))
        rows.append(dict(determinant=r['determinant'],column=col,breakpoint=round(bp,3),
            below_at=round(below,3),above_at=round(above,3),
            pd_below=round(pd_at(below),4),pd_at_bp=round(pd_at(bp),4),
            pd_above=round(pd_at(above),4),bp_in_data_range=bp_in_range))
    pd.DataFrame(rows).to_csv(out('pd_translation.csv'),index=False)
    print('pd_translation.csv')


# ====================================================================
# STEP 7 — RISK ZONES
# ====================================================================
def step_risk_zones(df_full, thr):
    print('\n[STEP 7] Regulatory risk zones')
    df,lag=build_sample(df_full,'sample_noESG',False)
    m=xgb.XGBRegressor(**XGB_PARAMS); m.fit(df[lag],df[TARGET])
    S=shap.TreeExplainer(m).shap_values(df[lag]); med=df[lag].median()
    def zero_cross(x,sv):
        s=pd.DataFrame({'x':x,'sv':sv}).dropna(); s=s[np.isfinite(s['x'])]
        try: s['b']=pd.qcut(s['x'],q=30,duplicates='drop')
        except: return None
        g=s.groupby('b',observed=True).agg(x=('x','median'),sv=('sv','median')).reset_index(drop=True).sort_values('x').reset_index(drop=True)
        for i in range(1,len(g)):
            if g['sv'].iloc[i-1]<=0 and g['sv'].iloc[i]>0: return float((g['x'].iloc[i-1]+g['x'].iloc[i])/2)
        return None
    rows=[]
    for col,name in ROBUST_ZONE.items():
        if col not in lag: continue
        sub=thr[(thr.column==col)&(thr['sample']=='Expanded')]
        if not len(sub): continue
        bp=float(sub['breakpoint'].iloc[0]); zc=zero_cross(df[col].values.astype(float),S[:,lag.index(col)])
        def pd_at(v): row=med.copy(); row[col]=v; return float(np.exp(m.predict(row.values.reshape(1,-1))[0]))
        rows.append(dict(determinant=name,column=col,amber_start=round(zc,3) if zc is not None else None,
            red_start=round(bp,3),pd_at_red=round(pd_at(bp),4)))
    tab=pd.DataFrame(rows); tab.to_csv(out('risk_zones.csv'),index=False)
    plot=tab.dropna(subset=['amber_start']).reset_index(drop=True)
    if len(plot):
        fig,ax=plt.subplots(figsize=(9,0.8*len(plot)+1.5))
        for i,r in plot.iterrows():
            x=df[r['column']].values; lo,hi=np.nanpercentile(x,[2,98])
            a,rd=sorted([r['amber_start'],r['red_start']])
            ax.barh(i,a-lo,left=lo,color=GREEN,alpha=0.8,edgecolor='white')
            ax.barh(i,rd-a,left=a,color=AMBER,alpha=0.85,edgecolor='white')
            ax.barh(i,hi-rd,left=rd,color=RED,alpha=0.8,edgecolor='white')
            ax.text(rd,i,f' bp={r["red_start"]:.2f}',va='center',fontsize=8,fontweight='bold')
        ax.set_yticks(range(len(plot))); ax.set_yticklabels(plot['determinant'])
        ax.set_xlabel('Determinant value (lagged)')
        ax.set_title('Regulatory Risk Zones (green: low · amber: elevated · red: escalating)',
                     fontsize=11,fontweight='bold')
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=GREEN,label='Green'),Patch(color=AMBER,label='Amber'),
                           Patch(color=RED,label='Red')],loc='lower right',fontsize=8)
        plt.tight_layout(); plt.savefig(out('Figure_risk_zones.png'),dpi=190,bbox_inches='tight'); plt.close()
    print('risk_zones.csv, Figure_risk_zones.png')

# ====================================================================
# MAIN
# ====================================================================
def main():
    print('Loading', DATA)
    df_full=pd.read_stata(DATA if os.path.exists(DATA) else out(DATA))
    thr=None; cache=None
    if RUN_STEPS['model_comparison']: step_model_comparison(df_full)
    if RUN_STEPS['shap_plots']:       step_shap_plots(df_full)
    if RUN_STEPS['thresholds']:       cache,thr=step_thresholds(df_full)
    if RUN_STEPS['stability']:        step_stability(df_full)

    # PD translation and risk zones need the threshold table. If step 4 was
    # skipped this run, fall back to a previously saved threshold_table.csv so
    # the later steps still produce output instead of silently doing nothing.
    if thr is None and os.path.exists(out('threshold_table.csv')):
        thr=pd.read_csv(out('threshold_table.csv'))
        print('\n(using existing threshold_table.csv for steps 6-7)')
    if RUN_STEPS['pd_translation']:
        if thr is not None: step_pd_translation(df_full,thr)
        else: print('\n[STEP 6] skipped — no threshold_table.csv available')
    if RUN_STEPS['risk_zones']:
        if thr is not None: step_risk_zones(df_full,thr)
        else: print('\n[STEP 7] skipped — no threshold_table.csv available')

    print('\nAll requested steps complete.')
    print(f'All outputs (CSVs + figures) written to: {OUTDIR}')
    expected=['model_comparison.csv','shap_global_importance.csv','threshold_table.csv',
              'performance.csv','ale_validation.csv','stability_era.csv',
              'pd_translation.csv','risk_zones.csv',
              'SHAP_global_bar.png','SHAP_global_beeswarm.png','SHAP_local_high.png',
              'SHAP_local_median.png','SHAP_local_low.png','shap_dependence_all.png',
              'Figure_thresholds.png','Figure_risk_zones.png']
    missing=[f for f in expected if not os.path.exists(out(f))]
    print(f'Outputs present: {len(expected)-len(missing)}/{len(expected)}')
    if missing: print('  MISSING:', ', '.join(missing))

if __name__=='__main__':
    main()