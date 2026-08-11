#!/usr/bin/env python3
"""run_topology_arms_eval.py — Zero-shot 5-seed eval for topology arm checkpoints.

Protocol: 5 seeds (0..4), BudgetRevenueEnv k=50 (generous budget; budget_col=1.0),
greedy action selection. Networks: polblogs, FF n=1000, Rice-FB, Modular-FF, FF n=2000.
Reference baselines read from frozen JSONs (never rerun).

GATE A (arm A, polblogs only): STRONG>=530.4  PARTIAL>=420.0  else FAIL
GATE B (arm B, general):       STRONG: polblogs>=530.4 AND FF1000>=440.0 AND Rice>=190.0
                               PARTIAL: polblogs>=420.0 with same floors  else FAIL

Output: results/logs/topology_arms_eval.json
"""
from __future__ import annotations
import hashlib, json, os, sys, time
import numpy as np
import torch
import networkx as nx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Approximate betweenness (k=200 pivots) to avoid O(n^3) on dense graphs
_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import generate_forest_fire, generate_modular_forest_fire, load_rice_facebook
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast

CKPT_DIR = "results/checkpoints"
LOG_DIR  = "results/logs"
BA_CKPT  = os.path.join(CKPT_DIR, "rev_gnn_lstm_ba.pt")
MIX_CKPT = os.path.join(CKPT_DIR, "rev_gnn_lstm_densemix.pt")
C = 0.3; B_MAX = 12.0; K_EVAL = 50; SEEDS = list(range(5)); WEIGHT_HIGH = 2.0
N_MC = 5  # MC samples for IC influence (5 consistent with polblogs scripts; sufficient for relative ranking)


def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def _edge_index(G, device):
    edges=list(G.edges())
    if not edges: return torch.zeros((2,0),dtype=torch.long,device=device)
    nmap={v:i for i,v in enumerate(G.nodes())}
    src=[nmap[u] for u,_ in edges]+[nmap[v] for _,v in edges]
    dst=[nmap[v] for _,v in edges]+[nmap[u] for u,_ in edges]
    return torch.tensor([src,dst],dtype=torch.long,device=device)

def _feat(cache, env, k):
    base=compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
    # Budget col clamped to 1.0 (unconstrained proxy)
    return np.concatenate([base, np.ones((cache["n"],1), dtype=np.float32)], axis=1)

def _avail(env, n, device):
    m=torch.zeros(n,dtype=torch.bool,device=device)
    for i in env.available_nodes: m[i]=True
    return m

def _load_policy(ckpt, device):
    enc=GraphSAGEEncoder(in_dim=21,hidden_dim=64,n_layers=2)
    lstm=EpisodeLSTM(graph_dim=64,lstm_hidden=64,n_layers=1)
    pol=SequentialJointPolicy(enc,lstm,gnn_dim=64,context_dim=64)
    sd=torch.load(ckpt,map_location=device,weights_only=True)
    if "policy_state_dict" in sd: sd=sd["policy_state_dict"]
    elif "model_state_dict" in sd: sd=sd["model_state_dict"]
    pol.load_state_dict(sd,strict=True)
    return pol.to(device).eval()

@torch.no_grad()
def eval_episode(policy, G, cache, ei, k, seed, device):
    cfg=BudgetEnvConfig(budget_B=k*C,production_cost=C,seed=seed,weight_high=WEIGHT_HIGH,n_mc_samples=N_MC)
    env=BudgetRevenueEnv(G,cfg); env.reset()
    policy.reset_episode(device); rev=0.0
    for _ in range(G.number_of_nodes()):
        if not env.available_nodes or env._check_bankrupt(): break
        x=torch.FloatTensor(_feat(cache,env,k)).to(device)
        av=_avail(env,G.number_of_nodes(),device)
        if not av.any(): break
        sc,h,ctx,_=policy.forward(x,ei,av)
        ni=int(sc.argmax().item())
        d=float(policy.get_discount_distribution(torch.cat([h[ni],ctx])).mean.item())
        obs,rw,done,info=env.step(ni,d)
        if info["accepted"]: rev+=info["offered_price"]
        policy.update_sequence_state(d, info["accepted"], info.get("revenue_step",0.0))
        if done: break
    return rev

def eval_network(policy, G, device, label=""):
    cache=build_graph_feature_cache(G,compute_static_features(G))
    ei=_edge_index(G,device)
    revs=[eval_episode(policy,G,cache,ei,K_EVAL,s,device) for s in SEEDS]
    m=float(np.mean(revs))
    print(f"  {label}: {[f'{r:.1f}' for r in revs]} mean={m:.1f}")
    return m

def _gate_a(pb): return "STRONG" if pb>=530.4 else ("PARTIAL" if pb>=420.0 else "FAIL")
def _gate_b(pb,ff,rice): 
    if pb>=530.4 and ff>=440.0 and rice>=190.0: return "STRONG"
    if pb>=420.0 and ff>=440.0 and rice>=190.0: return "PARTIAL"
    return "FAIL"

def main():
    t0=time.time(); device=torch.device("cpu")
    # Load networks
    print("[eval] Loading networks...")
    polblogs=load_polblogs()
    ff1000=generate_forest_fire(1000,0.37,0.32,seed=0)
    rice=load_rice_facebook()
    modf=generate_modular_forest_fire([250,250],0.37,0.32,0.05,seed=0)  # Modular-FF ~500n (2x250)
    ff2000=generate_forest_fire(2000,0.37,0.32,seed=1)
    networks={"polblogs":polblogs,"FF_1000":ff1000,"Rice_FB":rice,"Modular_FF":modf,"FF_2000":ff2000}
    print(f"  Networks: {', '.join(f'{k}(n={G.number_of_nodes()})' for k,G in networks.items())}")
    # Load frozen LSTM baseline from polblogs_eval.json
    frozen_lstm=None
    pb_eval_path=os.path.join(LOG_DIR,"polblogs_eval.json")
    if os.path.exists(pb_eval_path):
        pb_data=json.load(open(pb_eval_path))
        frozen_lstm=pb_data.get("protocol_b",{}).get("lstm_mean",None)
    # Load arm policies
    results={}
    for ckpt_name, ckpt_path in [("arm_a", BA_CKPT), ("arm_b", MIX_CKPT)]:
        if not os.path.exists(ckpt_path):
            print(f"[eval] {ckpt_path} not found — skip {ckpt_name}")
            continue
        sha=_sha8(ckpt_path)
        print(f"\n[eval] {ckpt_name} sha8={sha}")
        policy=_load_policy(ckpt_path, device)
        arm_res={}
        for net_name, G in networks.items():
            m=eval_network(policy, G, device, f"{ckpt_name}/{net_name}")
            arm_res[net_name]=m
        results[ckpt_name]={"sha8":sha,"means":arm_res}
    # Read frozen Greedy from polblogs_eval
    greedy_pb=530.38  # from frozen polblogs_eval.json
    # Table
    print(f"\n{'─'*72}")
    print(f"{'network':<14} | {'frozen_LSTM':>11} | {'arm_A_BA':>10} | {'arm_B_mix':>10} | {'Greedy':>8}")
    print(f"{'─'*72}")
    for net in ["polblogs","FF_1000","Rice_FB","Modular_FF","FF_2000"]:
        fl=frozen_lstm if net=="polblogs" else "N/A"
        ra=results.get("arm_a",{}).get("means",{}).get(net,"—")
        rb=results.get("arm_b",{}).get("means",{}).get(net,"—")
        gr=greedy_pb if net=="polblogs" else "N/A"
        print(f"{net:<14} | {str(fl) if fl else 'N/A':>11} | {ra if isinstance(ra,str) else f'{ra:.1f}':>10} | {rb if isinstance(rb,str) else f'{rb:.1f}':>10} | {gr if isinstance(gr,str) else f'{gr:.1f}':>8}")
    print(f"{'─'*72}")
    # Gate verdicts
    pb_a=results.get("arm_a",{}).get("means",{}).get("polblogs",-1)
    pb_b=results.get("arm_b",{}).get("means",{}).get("polblogs",-1)
    ff_b=results.get("arm_b",{}).get("means",{}).get("FF_1000",-1)
    ri_b=results.get("arm_b",{}).get("means",{}).get("Rice_FB",-1)
    gate_a_v=_gate_a(pb_a) if pb_a>=0 else "PENDING"
    gate_b_v=_gate_b(pb_b,ff_b,ri_b) if all(x>=0 for x in [pb_b,ff_b,ri_b]) else "PENDING"
    print(f"\nGATE A: {gate_a_v}  (arm_a polblogs={pb_a:.1f} vs 530.4/420.0)")
    print(f"GATE B: {gate_b_v}  (arm_b polblogs={pb_b:.1f} FF1000={ff_b:.1f} Rice={ri_b:.1f})")
    # Save
    out={"seeds":SEEDS,"k_eval":K_EVAL,"networks":{k:G.number_of_nodes() for k,G in networks.items()},
         "results":results,"gate_a":gate_a_v,"gate_b":gate_b_v,"wall_s":time.time()-t0}
    os.makedirs(LOG_DIR,exist_ok=True)
    json.dump(out,open(os.path.join(LOG_DIR,"topology_arms_eval.json"),"w"),indent=2)
    print(f"\nSaved → results/logs/topology_arms_eval.json ({time.time()-t0:.0f}s)")

if __name__=="__main__":
    main()
