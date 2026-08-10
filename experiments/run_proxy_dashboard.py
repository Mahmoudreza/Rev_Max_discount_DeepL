#!/usr/bin/env python3
"""Probe 3a: every 25 epochs check Arm A ckpt on BA proxy. Appends to /tmp/proxy_dashboard.log."""
import sys, os, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src.env.ba_generators import generate_ba
from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast
from src.evaluation.budget_baselines import greedy_discount_budget

C=0.3; B_MAX=12.0; WEIGHT_HIGH=2.0
CKPT_ARMA="results/checkpoints/rev_gnn_lstm_ba.pt"
DASHBOARD="/tmp/proxy_dashboard.log"
POLL_INTERVAL=60   # check every 60 seconds
K=20; SEEDS=[42,7]

def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def load_policy(ckpt,device):
    enc=GraphSAGEEncoder(in_dim=21,hidden_dim=64,n_layers=2)
    lstm=EpisodeLSTM(graph_dim=64,lstm_hidden=64,n_layers=1)
    pol=SequentialJointPolicy(enc,lstm,gnn_dim=64,context_dim=64).to(device)
    sd=torch.load(ckpt,map_location=device,weights_only=True)
    if "policy_state_dict" in sd: sd=sd["policy_state_dict"]
    elif "model_state_dict" in sd: sd=sd["model_state_dict"]
    pol.load_state_dict(sd,strict=True); pol.eval(); return pol

def _edge_index(G,device):
    edges=list(G.edges())
    if not edges: return torch.zeros((2,0),dtype=torch.long,device=device)
    nmap={v:i for i,v in enumerate(G.nodes())}
    src=[nmap[u] for u,_ in edges]+[nmap[v] for _,v in edges]
    dst=[nmap[v] for _,v in edges]+[nmap[u] for u,_ in edges]
    return torch.tensor([src,dst],dtype=torch.long,device=device)

def _feat(cache,env,k):
    base=compute_node_features_fast(cache,env.S,env.offered,env.t,k,env)
    return np.concatenate([base,np.full((cache["n"],1),env.B/B_MAX,dtype=np.float32)],axis=1)

def _avail(env,n,device):
    m=torch.zeros(n,dtype=torch.bool,device=device)
    for i in env.available_nodes: m[i]=True
    return m

def eval_ckpt(pol,G,cache,ei,k,seeds,device):
    revs=[]
    for seed in seeds:
        cfg=BudgetEnvConfig(budget_B=k*C,production_cost=C,seed=seed,weight_high=WEIGHT_HIGH)
        env=BudgetRevenueEnv(G,cfg); env.reset()
        pol.reset_episode(device); rev=0.0
        for _ in range(G.number_of_nodes()):
            if not env.available_nodes or env._check_bankrupt(): break
            x=torch.FloatTensor(_feat(cache,env,k)).to(device)
            av=_avail(env,G.number_of_nodes(),device)
            if not av.any(): break
            sc,h,ctx,_=pol.forward(x,ei,av)
            ni=int(sc.argmax().item())
            d=float(pol.get_discount_distribution(torch.cat([h[ni],ctx])).mean.item())
            obs,rw,done,info=env.step(ni,d)
            if info["accepted"]: rev+=info["offered_price"]
            if done: break
        revs.append(rev)
    return float(np.mean(revs))

def main():
    device=torch.device("cpu")
    G=generate_ba(600,10,seed=999)
    cache=build_graph_feature_cache(G,compute_static_features(G))
    ei=_edge_index(G,device)
    gr=greedy_discount_budget(G,B=K*C,c=C,n_trials=2,weight_high=WEIGHT_HIGH)
    greedy_mean=gr["revenue"]["mean"]
    print(f"Dashboard: proxy greedy={greedy_mean:.2f}; polling {CKPT_ARMA} every {POLL_INTERVAL}s")
    print(f"Output → {DASHBOARD}")
    last_sha=""
    epoch_est=0
    with open(DASHBOARD,"a") as f:
        f.write(f"# Dashboard started {time.strftime('%Y-%m-%d %H:%M:%S')}; greedy={greedy_mean:.2f}\n")
        f.write("# epoch_est | proxy_rev | pct_greedy | sha8\n")
    while True:
        if os.path.exists(CKPT_ARMA):
            try:
                sha=_sha8(CKPT_ARMA)
                if sha!=last_sha:
                    pol=load_policy(CKPT_ARMA,device)
                    rev=eval_ckpt(pol,G,cache,ei,K,SEEDS,device)
                    pct=100*rev/greedy_mean if greedy_mean>0 else 0
                    epoch_est+=25  # rough estimate: ckpt updated every ~25 epochs when saved
                    line=f"ep~{epoch_est:4d} | {rev:8.2f} | {pct:6.1f}% | {sha}"
                    print(line,flush=True)
                    with open(DASHBOARD,"a") as f: f.write(line+"\n")
                    last_sha=sha
            except Exception as e:
                print(f"[warn] {e}",flush=True)
        time.sleep(POLL_INTERVAL)

if __name__=="__main__":
    main()
