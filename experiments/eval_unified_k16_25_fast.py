#!/usr/bin/env python3
"""eval_unified_k16_25_fast.py — ITEM 2: gatefail checkpoint at k=[16,20,25], approx_bc, CPU."""
import sys,os,json,hashlib,time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, networkx as nx
_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G,normalized=True,**kw: _orig_bc(G,k=min(200,G.number_of_nodes()),normalized=normalized,**kw)

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.env.revenue_env import RevenueEnv

# Faster get_current_influence: O(deg) not O(|S|)
def _fast_gci(self, node):
    nb = list(self.graph.neighbors(node))
    if not nb: return 0.0
    tw = sum(self._link_weights.get((node,n),0.0) for n in nb)
    if tw==0: return 0.0
    return sum(self._link_weights.get((node,n),0.0) for n in nb if n in self.S)/tw
RevenueEnv.get_current_influence = _fast_gci
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast

C=0.3; SEEDS=[42,123,7]; K_EVAL=[16,20,25]
CKPT="results/checkpoints/rev_gnn_lstm_unified_gatefail.pt"
EXP_SHA="00071438"
DEV=torch.device("cpu")

def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]
def _ei(G):
    E=list(G.edges()); nm={v:i for i,v in enumerate(G.nodes())}
    s=[nm[u] for u,_ in E]+[nm[v] for _,v in E]; d=[nm[v] for _,v in E]+[nm[u] for u,_ in E]
    return torch.tensor([s,d],dtype=torch.long,device=DEV)

def _load_pol():
    enc=GraphSAGEEncoder(21,64,2,0.); lstm=EpisodeLSTM(graph_dim=64,lstm_hidden=64,n_layers=1)
    pol=SequentialJointPolicy(enc,lstm,gnn_dim=64,context_dim=64)
    pol.load_state_dict(torch.load(CKPT,map_location=DEV,weights_only=True)); return pol.to(DEV).eval()

@torch.no_grad()
def eval_one(pol, G, cache, ei, k, seed):
    B=k*C
    cfg=BudgetEnvConfig(budget_B=B,production_cost=C,seed=seed,weight_high=2.0)
    env=BudgetRevenueEnv(G,cfg); env.reset(); n=G.number_of_nodes(); pol.reset_episode(DEV)
    rev=0.
    for _ in range(n):
        if not env.available_nodes or env._check_bankrupt(): break
        base=compute_node_features_fast(cache,env.S,env.offered,env.t,k,env)
        col=np.full((n,1),env.B/(40*C),dtype=np.float32)
        x=torch.FloatTensor(np.concatenate([base,col],axis=1)).to(DEV)
        av=torch.tensor([v not in env.offered and v not in env.S for v in G.nodes()],dtype=torch.bool,device=DEV)
        if not av.any(): break
        sc,h,ctx,_=pol.forward(x,ei,av); idx=int(sc.argmax())
        comb=torch.cat([h[idx],ctx]); disc=float(pol.get_discount_distribution(comb).mean.item())
        est=env._estimate_valuation(env.nodes[idx]); price=est*(1.-disc)
        if env.B-C+price<-1e-9: env.offered.add(env.nodes[idx]); env.t+=1; env.budget_history.append(env.B); pol.update_sequence_state(disc,False,0.); continue
        obs,rew,done,info=env.step(idx,disc)
        if info["accepted"]: rev+=info["offered_price"]
        pol.update_sequence_state(disc,info["accepted"],info.get("revenue_step",0.))
        if done: break
    return rev

def main():
    sha=_sha8(CKPT); assert sha==EXP_SHA, f"SHA fail: {sha}"
    print(f"SHA OK: {sha}")
    import random; random.seed(42); np.random.seed(42)
    G=generate_forest_fire(1000,p=0.37,pb=0.32,seed=42)
    print(f"Graph: n={G.number_of_nodes()} edges={G.number_of_edges()}")
    static=compute_static_features(G); cache=build_graph_feature_cache(G,static); ei=_ei(G)
    print("Features done.")
    pol=_load_pol()
    res={}
    for k in K_EVAL:
        vals=[eval_one(pol,G,cache,ei,k,s) for s in SEEDS]
        m=round(float(np.mean(vals)),2)
        print(f"k={k}: {vals} mean={m}")
        res[k]={"mean":m,"all":[round(v,2) for v in vals]}
    # Harness check
    k20=res[20]["mean"]
    status="OK" if abs(k20-369.6)<2.0 else f"FAIL (expected 369.6±2, got {k20})"
    print(f"Harness k=20: {k20} — {status}")
    out={"sha8":sha,"seeds":SEEDS,"results":{str(k):v for k,v in res.items()},"harness_k20":k20,"harness_status":status}
    os.makedirs("results/logs",exist_ok=True)
    json.dump(out,open("results/logs/unified_k16_25_fast.json","w"),indent=2)
    print("Saved results/logs/unified_k16_25_fast.json")

if __name__=="__main__": main()
