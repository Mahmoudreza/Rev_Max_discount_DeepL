#!/usr/bin/env python3
"""BA proxy probes 1+2+3b. Probe 3a dashboard is a separate script."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import networkx as nx
from src.env.ba_generators import generate_ba
from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast
from src.evaluation.budget_baselines import greedy_discount_budget

C=0.3; B_MAX=12.0; WEIGHT_HIGH=2.0
CKPT_BASE="results/checkpoints/rev_gnn_lstm.pt"
CKPT_ARMA="results/checkpoints/rev_gnn_lstm_ba.pt"

def _sha8(p): import hashlib; return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def load_policy(ckpt_path, in_dim=20, device=torch.device("cpu")):
    enc=GraphSAGEEncoder(in_dim=in_dim,hidden_dim=64,n_layers=2)
    lstm=EpisodeLSTM(graph_dim=64,lstm_hidden=64,n_layers=1)
    pol=SequentialJointPolicy(enc,lstm,gnn_dim=64,context_dim=64).to(device)
    sd=torch.load(ckpt_path,map_location=device,weights_only=True)
    if "policy_state_dict" in sd: sd=sd["policy_state_dict"]
    elif "model_state_dict" in sd: sd=sd["model_state_dict"]
    pol.load_state_dict(sd,strict=True); pol.eval(); return pol

def _edge_index(G, device):
    edges=list(G.edges())
    if not edges: return torch.zeros((2,0),dtype=torch.long,device=device)
    nmap={v:i for i,v in enumerate(G.nodes())}
    src=[nmap[u] for u,_ in edges]+[nmap[v] for _,v in edges]
    dst=[nmap[v] for _,v in edges]+[nmap[u] for u,_ in edges]
    return torch.tensor([src,dst],dtype=torch.long,device=device)

def _feat(cache, env, k):
    base=compute_node_features_fast(cache,env.S,env.offered,env.t,k,env)
    return np.concatenate([base,np.full((cache["n"],1),env.B/B_MAX,dtype=np.float32)],axis=1)

def _avail(env, n, device):
    m=torch.zeros(n,dtype=torch.bool,device=device)
    for i in env.available_nodes: m[i]=True
    return m

def eval_policy(policy, G, cache, ei, k, seeds, device, in_dim=21):
    revs=[]
    policy.eval()
    for seed in seeds:
        cfg=BudgetEnvConfig(budget_B=k*C,production_cost=C,seed=seed,weight_high=WEIGHT_HIGH)
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
            if done: break
        revs.append(rev)
    return float(np.mean(revs)), revs

def main():
    device=torch.device("cpu")
    # ── Build proxy graph ─────────────────────────────────────────────────────
    print("=== PROBE 1: BA n=600, m=10, seed=999 (held-out) ===")
    G_proxy=generate_ba(600,10,seed=999)
    n=G_proxy.number_of_nodes(); m=G_proxy.number_of_edges()
    degs=sorted([d for _,d in G_proxy.degree()],reverse=True)
    max_deg=degs[0]; med_deg=degs[len(degs)//2]
    print(f"n={n} edges={m} max_deg={max_deg} med_deg={med_deg} max/med={max_deg/med_deg:.1f}")

    # Greedy on proxy
    K=20; seeds=[42,123,7]
    gr=greedy_discount_budget(G_proxy,B=K*C,c=C,n_trials=3,weight_high=WEIGHT_HIGH)
    greedy_mean=gr["revenue"]["mean"]; greedy_all=gr["revenue"]["all"]
    print(f"Greedy k={K}: mean={greedy_mean:.2f} all={[f'{x:.1f}' for x in greedy_all]}")

    # Frozen LSTM on proxy (20-dim)
    sha=_sha8(CKPT_BASE); print(f"Frozen ckpt sha8={sha}")
    pol_frozen=load_policy(CKPT_BASE,in_dim=20,device=device)
    cache=build_graph_feature_cache(G_proxy,compute_static_features(G_proxy))
    cache20={k:v for k,v in cache.items()}  # same cache, 20-dim features
    ei=_edge_index(G_proxy,device)
    # frozen policy uses 20-dim features (no budget col)
    class FrozenWrapper:
        def __init__(self,pol): self.pol=pol
        def eval(self): self.pol.eval()
        def reset_episode(self,dev): self.pol.reset_episode(dev)
        def forward(self,x,ei,av): return self.pol.forward(x[:,:-1],ei,av)  # strip budget col
        def get_discount_distribution(self,v): return self.pol.get_discount_distribution(v)
        def update_sequence_state(self,*a): return self.pol.update_sequence_state(*a) if hasattr(self.pol,'update_sequence_state') else None
    fw=FrozenWrapper(pol_frozen)
    frozen_mean,frozen_all=eval_policy(fw,G_proxy,cache,ei,K,seeds,device,in_dim=20)
    print(f"Frozen LSTM k={K}: mean={frozen_mean:.2f} all={[f'{x:.1f}' for x in frozen_all]}")
    ratio1=frozen_mean/greedy_mean if greedy_mean>0 else 0
    print(f"Frozen/Greedy ratio: {ratio1:.3f}")
    if ratio1>0.92:
        print("PROBE 1 INVALID: frozen LSTM ~matches Greedy on proxy → proxy not capturing BA skew gap")
        print("Stopping. 48h gates remain the only readout.")
        return
    print("PROBE 1 VALID: frozen LSTM clearly below Greedy on proxy ✓")

    # ── Probe 2: Arm A intermediate checkpoint ────────────────────────────────
    print("\n=== PROBE 2: Arm A partial checkpoint (Phase-1) ===")
    if not os.path.exists(CKPT_ARMA):
        print("CKPT_ARMA not found yet (Phase 1 still running) — skipping Probe 2")
    else:
        sha_a=_sha8(CKPT_ARMA)
        print(f"Arm A ckpt sha8={sha_a}")
        pol_a=load_policy(CKPT_ARMA,in_dim=21,device=device)
        a_mean,a_all=eval_policy(pol_a,G_proxy,cache,ei,K,seeds,device)
        ratio_a=a_mean/greedy_mean if greedy_mean>0 else 0
        print(f"Arm A (partial) k={K}: mean={a_mean:.2f} all={[f'{x:.1f}' for x in a_all]}")
        print(f"Arm A / Greedy proxy: {ratio_a:.3f} ({'ON TRACK' if ratio_a>=0.85 else 'EARLY WARNING' if ratio_a<0.60 else 'MONITOR'})")
        # Polblogs eval (2 seeds only)
        try:
            from src.env.polblogs_loader import load_polblogs_lcc
            G_pb=load_polblogs_lcc()
            cache_pb=build_graph_feature_cache(G_pb,compute_static_features(G_pb))
            ei_pb=_edge_index(G_pb,device)
            pb_mean,pb_all=eval_policy(pol_a,G_pb,cache_pb,ei_pb,K,[42,123],device)
            gr_pb=greedy_discount_budget(G_pb,B=K*C,c=C,n_trials=2,weight_high=WEIGHT_HIGH)
            print(f"Arm A polblogs k={K}: mean={pb_mean:.2f} greedy={gr_pb['revenue']['mean']:.2f}")
        except Exception as e:
            print(f"Polblogs eval failed: {e}")

    # ── Probe 3b: Feature saturation check ───────────────────────────────────
    print("\n=== PROBE 3b: Feature saturation — hub vs median nodes ===")
    degs_dict=dict(G_proxy.degree())
    sorted_nodes=sorted(degs_dict, key=lambda x:degs_dict[x], reverse=True)
    hub_nodes=sorted_nodes[:10]; med_start=len(sorted_nodes)//2
    med_nodes=sorted_nodes[med_start:med_start+10]
    # get feature matrix at episode start
    cfg0=BudgetEnvConfig(budget_B=K*C,production_cost=C,seed=42,weight_high=WEIGHT_HIGH)
    env0=BudgetRevenueEnv(G_proxy,cfg0); env0.reset()
    feat_mat=np.array(_feat(cache,env0,K),dtype=np.float32)  # (n,21)
    nmap={v:i for i,v in enumerate(G_proxy.nodes())}
    hub_idx=[nmap[v] for v in hub_nodes]
    med_idx=[nmap[v] for v in med_nodes]
    hub_feats=feat_mat[hub_idx]   # (10,21)
    med_feats=feat_mat[med_idx]   # (10,21)
    print(f"{'dim':>4} {'hub_min':>8} {'hub_max':>8} {'hub_mean':>9} | {'med_min':>8} {'med_max':>8} {'med_mean':>9} | flag")
    for d in range(21):
        hmin,hmax,hmean=hub_feats[:,d].min(),hub_feats[:,d].max(),hub_feats[:,d].mean()
        mmin,mmax,mmean=med_feats[:,d].min(),med_feats[:,d].max(),med_feats[:,d].mean()
        # flag if hub range is >5x median range or if hub values are very large
        flag=""
        if hmean>0 and mmean>0 and (hmean/mmean>5 or hmean/mmean<0.2): flag="**SATURATE**"
        elif hmax>10 or hmin<-5: flag="*large*"
        print(f"{d:>4} {hmin:>8.3f} {hmax:>8.3f} {hmean:>9.3f} | {mmin:>8.3f} {mmax:>8.3f} {mmean:>9.3f} | {flag}")

    hub_degs=[degs_dict[v] for v in hub_nodes]; med_degs=[degs_dict[v] for v in med_nodes]
    print(f"Hub degrees: {sorted(hub_degs,reverse=True)}")
    print(f"Med degrees: {sorted(med_degs,reverse=True)}")
    print("=== DONE ===")

if __name__=="__main__":
    main()
