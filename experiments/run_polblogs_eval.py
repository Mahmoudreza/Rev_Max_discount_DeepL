#!/usr/bin/env python3
"""run_polblogs_eval.py — Idea 1 zero-shot eval on Polblogs LCC.

Methods: IE-Strategy, Greedy-Discount, Rev-GNN-IM-RL, Rev-GNN-LSTM (sha 8fbc4648)
Protocols: (a) single-seed=42, (b) 5-seed [0..4]
MC: n_mc_samples=5 (dense IC on n=1222 requires low MC; consistent with Rice-FB=10 precedent)
Betweenness: approx k=200
Output: results/logs/polblogs_eval.json
"""
import sys,os,json,hashlib,time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch, networkx as nx
_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G,normalized=True,**kw: _orig_bc(G,k=min(200,G.number_of_nodes()),normalized=normalized,**kw)

from src.env.polblogs_loader import load_polblogs, polblogs_stats
from src.evaluation.baselines import ie_strategy, greedy_discount
from src.env.revenue_env import RevenueEnv, RevenueEnvConfig

# Faster get_current_influence: iterate over neighbors (O(deg)) not S (O(|S|))
def _fast_gci(self, node):
    neighbors = list(self.graph.neighbors(node))
    if not neighbors: return 0.0
    tw = sum(self._link_weights.get((node,nb),0.0) for nb in neighbors)
    if tw==0: return 0.0
    return sum(self._link_weights.get((node,nb),0.0) for nb in neighbors if nb in self.S) / tw
RevenueEnv.get_current_influence = _fast_gci
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.models.policies.joint_policy import JointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast
from types import SimpleNamespace as NS

CKPT_DIR  = "results/checkpoints"
LSTM_CKPT = os.path.join(CKPT_DIR, "rev_gnn_lstm.pt")
IMRL_CKPT = os.path.join(CKPT_DIR, "rev_gnn_im_rl.pt")
LSTM_SHA  = "8fbc4648"
HID=64; FEAT_DIM=20
MC=5  # n_mc_samples for polblogs (dense IC graph)
SINGLE_SEED=42; FIVE_SEEDS=[0,1,2,3,4]

def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def _cfg(seed,k=20):
    return NS(influence=NS(model="monotone",b=1.0,weight_low=0.,weight_high=2.,n_mc_samples=MC),
              reward=NS(type="flat",gamma=1.0),budget=NS(k=k),project=NS(seed=seed))

def _env_cfg(seed):
    return RevenueEnvConfig(influence_model="monotone",b=1.0,weight_low=0.,weight_high=2.,
                            n_mc_samples=MC,reward_type="flat",gamma=1.0,seed=seed)

def _load_lstm(dev):
    enc=GraphSAGEEncoder(FEAT_DIM,HID,2,0.); lstm=EpisodeLSTM(graph_dim=HID,lstm_hidden=HID,n_layers=1)
    pol=SequentialJointPolicy(enc,lstm,gnn_dim=HID,context_dim=HID)
    pol.load_state_dict(torch.load(LSTM_CKPT,map_location=dev,weights_only=True)); return pol.to(dev).eval()

def _load_imrl(dev):
    enc=GraphSAGEEncoder(FEAT_DIM,HID,2,0.); pol=JointPolicy(enc,hidden_dim=HID)
    pol.load_state_dict(torch.load(IMRL_CKPT,map_location=dev,weights_only=True)); return pol.to(dev).eval()

def _ei(G,dev):
    E=list(G.edges()); nm={v:i for i,v in enumerate(G.nodes())}
    s=[nm[u] for u,_ in E]+[nm[v] for _,v in E]; d=[nm[v] for _,v in E]+[nm[u] for u,_ in E]
    return torch.tensor([s,d],dtype=torch.long,device=dev)

@torch.no_grad()
def eval_lstm(pol,G,seed,dev,cache,ei):
    env=RevenueEnv(G,_env_cfg(seed)); env.reset(); nodes=list(G.nodes()); n=len(nodes)
    pol.reset_episode(dev); rev,off=0.0,set()
    for _ in range(n):
        if len(off)==n: break
        x=torch.FloatTensor(compute_node_features_fast(cache,env.S,off,env.t,0,env)).to(dev)
        av=torch.tensor([v not in off for v in nodes],dtype=torch.bool,device=dev)
        sc,h,ctx,_=pol.forward(x,ei,av); idx=int(sc.argmax()); v=nodes[idx]
        disc=float(pol.get_discount_distribution(torch.cat([h[idx],ctx])).mean.item())
        price=env._estimate_valuation(v)*(1.-disc); off.add(v)
        if env._true_valuation(v)>=price: rev+=price; env.S.add(v)
        env.t+=1; pol.update_sequence_state(disc,env._true_valuation(v)>=price,price if env._true_valuation(v)>=price else 0.)
    return rev

@torch.no_grad()
def eval_imrl(pol,G,seed,dev,cache,ei):
    env=RevenueEnv(G,_env_cfg(seed)); env.reset(); nodes=list(G.nodes()); n=len(nodes)
    rev,off=0.0,set()
    for _ in range(n):
        if len(off)==n: break
        x=torch.FloatTensor(compute_node_features_fast(cache,env.S,off,env.t,0,env)).to(dev)
        av=torch.tensor([v not in off for v in nodes],dtype=torch.bool,device=dev)
        idx,disc,_=pol.select_and_price(x,ei,av,greedy=True); v=nodes[idx]
        price=env._estimate_valuation(v)*(1.-disc); off.add(v)
        if env._true_valuation(v)>=price: rev+=price; env.S.add(v)
        env.t+=1
    return rev

def run_proto(G,seeds,dev,lstm_pol,imrl_pol,cache,ei):
    ie_r,gd_r,imrl_r,lstm_r=[],[],[],[]
    for seed in seeds:
        t1=time.time(); print(f"  seed={seed}...",flush=True)
        ie_r.append(ie_strategy(G,_cfg(seed))); print(f"    IE={ie_r[-1]:.1f} ({time.time()-t1:.0f}s)",flush=True)
        t1=time.time(); gd_r.append(greedy_discount(G,_cfg(seed))); print(f"    GD={gd_r[-1]:.1f} ({time.time()-t1:.0f}s)",flush=True)
        imrl_r.append(eval_imrl(imrl_pol,G,seed,dev,cache,ei))
        lstm_r.append(eval_lstm(lstm_pol,G,seed,dev,cache,ei))
        print(f"    IMRL={imrl_r[-1]:.1f} LSTM={lstm_r[-1]:.1f}",flush=True)
    def ms(x): return {"mean":round(float(np.mean(x)),2),"all":[round(v,2) for v in x]}
    return {"ie":ms(ie_r),"greedy_disc":ms(gd_r),"imrl":ms(imrl_r),"lstm":ms(lstm_r)}

def main():
    t0=time.time()
    G=load_polblogs(); s=polblogs_stats(G)
    print(f"polblogs LCC: n={s['n']} m={s['m']} mean_deg={s['mean_deg']:.2f} density={s['density']:.4f}",flush=True)
    print("computing features (approx_bc k=200)...",flush=True)
    static=compute_static_features(G); cache=build_graph_feature_cache(G,static)
    print("features done.",flush=True)
    # CPU is faster than MPS for small-graph inference (lower dispatch latency)
    dev=torch.device("cpu")
    sha_l=_sha8(LSTM_CKPT); assert sha_l==LSTM_SHA,f"LSTM SHA fail:{sha_l}"
    sha_r=_sha8(IMRL_CKPT)
    print(f"Device:{dev} LSTM={sha_l} IMRL={sha_r}",flush=True)
    lstm_pol=_load_lstm(dev); imrl_pol=_load_imrl(dev); ei=_ei(G,dev)
    print("=== Protocol (a): seed=42 ===",flush=True)
    res_a=run_proto(G,[SINGLE_SEED],dev,lstm_pol,imrl_pol,cache,ei)
    for k,v in res_a.items(): print(f"  {k}: {v['mean']}",flush=True)
    print("=== Protocol (b): seeds 0..4 ===",flush=True)
    res_b=run_proto(G,FIVE_SEEDS,dev,lstm_pol,imrl_pol,cache,ei)
    for k,v in res_b.items(): print(f"  {k}: {v['mean']}  per={v['all']}",flush=True)
    g5=res_b["greedy_disc"]["mean"]; l5=res_b["lstm"]["mean"]
    print(f"LSTM vs Greedy margin: {100*(l5-g5)/g5:+.1f}%  LSTM={l5:.2f} GD={g5:.2f}",flush=True)
    out={"graph":s,"lstm_sha8":sha_l,"imrl_sha8":sha_r,"n_mc_samples":MC,
         "betweenness":"approx_k200","protocol_a_seed":SINGLE_SEED,"protocol_b_seeds":FIVE_SEEDS,
         "protocol_a":res_a,"protocol_b":res_b,"wall_seconds":time.time()-t0}
    os.makedirs("results/logs",exist_ok=True)
    json.dump(out,open("results/logs/polblogs_eval.json","w"),indent=2)
    print(f"Saved results/logs/polblogs_eval.json ({time.time()-t0:.0f}s)",flush=True)

if __name__=="__main__": main()
