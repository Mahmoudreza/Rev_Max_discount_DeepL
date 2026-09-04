"""Shared arm1/arm2/arm3 episode functions for budget-protocol jobs."""
from __future__ import annotations
import math, os, sys
import numpy as np
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env.polblogs_loader import load_polblogs
from src.env.budget_revenue_env import BudgetEnvConfig
from src.env.graph_generators import (generate_forest_fire,
                                       generate_modular_forest_fire,
                                       load_rice_facebook)
from src.evaluation.dp_calibrated_v2_obs import calibrate_v2_obs_table
from src.evaluation.dp_calibrated_v2 import _deg_class
from src.evaluation.budget_baselines import _make_env, greedy_discount_budget
try:
    from src.evaluation.ie_budget import ie_strategy_budget
except ImportError:
    from src.evaluation.budget_baselines import ie_strategy_budget

C = 0.30; N_TRIALS = 10; N_SIMS = 30; SHORTLIST_N = 50
TIERS = (1.0, 0.8, 0.5, 0.2, 0.0)
T_CRIT = stats.t.ppf(0.975, df=9)
NETS = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]


def load_graph(net):
    if net == "polblogs":   return load_polblogs()
    if net == "FF_1000":    return generate_forest_fire(1000, 0.37, 0.32, seed=0)
    if net == "Rice_FB":    return load_rice_facebook()
    if net == "Modular_FF": return generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0)
    if net == "FF_2000":    return generate_forest_fire(2000, 0.37, 0.32, seed=1)
    raise ValueError(net)


def _ib(x, ib):
    for i in range(len(ib)-2, 0, -1):
        if x >= ib[i]: return i
    return 0


def _shortlist(remaining, env, graph, infl_bnd, n=SHORTLIST_N, infl_cache=None):
    if len(remaining) <= n: return list(remaining)
    def _gi(nd):
        if infl_cache is not None:
            if nd not in infl_cache: infl_cache[nd] = env.get_current_influence(nd)
            return infl_cache[nd]
        return env.get_current_influence(nd)
    scored = [(-_ib(float(_gi(nd)), infl_bnd), -graph.degree(nd), nd) for nd in remaining]
    scored.sort()
    return [nd for _,_,nd in scored[:n]]


def _score_a2(cands, env, graph, A, V, class_bnd, infl_bnd, b):
    best_sc=-1e18; best_i=None; best_tau=None
    for nd in cands:
        infl=env.get_current_influence(nd); ib=min(_ib(float(infl),infl_bnd),A.shape[1]-1)
        cls=int(_deg_class(int(graph.degree(nd)),class_bnd)); v_nd=float(V[cls,ib])
        for ti,tau in enumerate(TIERS):
            price_v=v_nd*(1.0-tau)
            if b-C+price_v<-1e-9: continue
            sc=float(A[cls,ib,ti])*price_v
            if sc>best_sc: best_sc=sc; best_i=nd; best_tau=tau
    return best_i, best_tau


def _score_a3(cands, env, graph, A, V, class_bnd, infl_bnd, b, lam, infl_cache=None):
    def _gi(nd):
        if infl_cache is not None:
            if nd not in infl_cache: infl_cache[nd] = env.get_current_influence(nd)
            return infl_cache[nd]
        return env.get_current_influence(nd)
    best_sc=-1e18; best_i=None; best_tau=None
    for nd in cands:
        infl=_gi(nd); ib=min(_ib(float(infl),infl_bnd),A.shape[1]-1)
        cls=int(_deg_class(int(graph.degree(nd)),class_bnd)); v_nd=float(V[cls,ib])
        nbr=0.0
        for j in graph.neighbors(nd):
            if j in env.offered: continue
            xj=float(_gi(j)); ibj=min(_ib(xj,infl_bnd),A.shape[1]-1)
            clsj=int(_deg_class(int(graph.degree(j)),class_bnd))
            xjp=min(1.0,xj+1.0/max(1,graph.degree(j))); ibjp=min(_ib(xjp,infl_bnd),A.shape[1]-1)
            nbr+=float(V[clsj,ibjp])-float(V[clsj,ibj])
        for ti,tau in enumerate(TIERS):
            price_v=v_nd*(1.0-tau)
            if b-C+price_v<-1e-9: continue
            sc=float(A[cls,ib,ti])*price_v+lam*nbr
            if sc>best_sc: best_sc=sc; best_i=nd; best_tau=tau
    return best_i, best_tau


def arm1_episode(env, ordering, A, class_bnd, infl_bnd):
    revenue=0.0; n_sk=0
    for node in ordering:
        if env._check_bankrupt() or len(env.offered)>=env.n: break
        if node in env.offered: continue
        est=float(env._estimate_valuation(node)); infl=env.get_current_influence(node); b=env.B
        ib=min(_ib(float(infl),infl_bnd),A.shape[1]-1); cls=int(_deg_class(int(env.graph.degree(node)),class_bnd))
        bv=-1e18; bd=None
        for ti,d in enumerate(TIERS):
            price=est*(1.0-d)
            if b-C+price<-1e-9: continue
            tv=float(A[cls,ib,ti])*price
            if tv>bv: bv=tv; bd=d
        if bd is None:
            env.offered.add(node); env.t+=1; env.budget_history.append(env.B); n_sk+=1; continue
        _,reward,done,_=env.step(env.node_to_idx[node],bd); revenue+=reward
        if done: break
    return revenue, n_sk


def _sel_episode(arm, lam, env, graph, A, V, class_bnd, infl_bnd):
    all_nodes=set(graph.nodes()); revenue=0.0; n_sk=0
    infl_cache = {}  # belt-and-suspenders: local episode cache (O(n^2) fix)
    while not env._check_bankrupt() and len(env.offered)<env.n:
        remaining=all_nodes-env.offered
        if not remaining: break
        b=env.B; cands=_shortlist(remaining,env,graph,infl_bnd,infl_cache=infl_cache)
        fn=_score_a2 if arm==2 else _score_a3
        kw={'infl_cache': infl_cache} if arm==3 else {}
        args=(cands,env,graph,A,V,class_bnd,infl_bnd,b) if arm==2 else (cands,env,graph,A,V,class_bnd,infl_bnd,b,lam)
        best_i,best_tau=fn(*args, **kw)
        if best_i is None and len(remaining)>len(cands):
            args2=(list(remaining),env,graph,A,V,class_bnd,infl_bnd,b) if arm==2 else (list(remaining),env,graph,A,V,class_bnd,infl_bnd,b,lam)
            best_i,best_tau=fn(*args2, **kw)
        if best_i is None: break
        est=float(env._estimate_valuation(best_i)); price=est*(1.0-best_tau)
        if b-C+price<-1e-9:
            env.offered.add(best_i); env.t+=1; env.budget_history.append(env.B); n_sk+=1; continue
        _,reward,done,_=env.step(env.node_to_idx[best_i],best_tau); revenue+=reward
        for nb in graph.neighbors(best_i): infl_cache.pop(nb, None)
        if done: break
    return revenue, n_sk


def arm2_episode(env, graph, A, V, class_bnd, infl_bnd):
    return _sel_episode(2, None, env, graph, A, V, class_bnd, infl_bnd)


def arm3_episode(env, graph, A, V, class_bnd, infl_bnd, lam):
    return _sel_episode(3, lam, env, graph, A, V, class_bnd, infl_bnd)


def _raw(r, n):
    v=r.get("total_revenue",r.get("revenue",{}))
    if isinstance(v,dict): return v.get("all",[v.get("mean",0.0)]*n)
    return [float(v)]*n


def run_baselines(graph, B, n_trials=N_TRIALS):
    r_ie=ie_strategy_budget(graph,B,C,n_trials=n_trials)
    r_gd=greedy_discount_budget(graph,B,C,n_trials=n_trials)
    return _raw(r_ie,n_trials), _raw(r_gd,n_trials)


def calibrate(graph, cfg=None):
    if cfg is None: cfg=BudgetEnvConfig()
    return calibrate_v2_obs_table(graph,cfg,n_sims=N_SIMS,seed=0)


def make_envs(graph, B, n_trials=N_TRIALS, cfg=None):
    if cfg is None: cfg=BudgetEnvConfig()
    envs=[]
    for s in range(n_trials):
        e=_make_env(graph,B=B,c=C,seed=s,weight_high=cfg.weight_high); e.reset(); envs.append(e)
    return envs, cfg


def _stats(vals):
    a=np.array(vals,float); m=float(a.mean()); s=float(a.std(ddof=1))
    se=s/math.sqrt(len(a))
    return {"mean":round(m,2),"std":round(s,2),
            "ci_lo":round(m-T_CRIT*se,2),"ci_hi":round(m+T_CRIT*se,2),
            "raw":[round(x,2) for x in a.tolist()]}


def _pt(a, b):
    d=np.array(a,float)-np.array(b,float)
    m=float(d.mean()); se=float(d.std(ddof=1))/math.sqrt(len(d))
    p=float(stats.ttest_1samp(d,0).pvalue)
    return {"diff":round(m,3),"ci":[round(m-T_CRIT*se,3),round(m+T_CRIT*se,3)],
            "p":round(p,4),"sig":bool(abs(m)>T_CRIT*se)}
