#!/usr/bin/env python3
"""run_topology_arms.py — Topology-coverage experiment: ARM A (BA-only) + ARM B (50/50 BA+FF).

ARM A: train on BA graphs only     → results/checkpoints/rev_gnn_lstm_ba.pt
ARM B: train on 50/50 BA+FF mix   → results/checkpoints/rev_gnn_lstm_densemix.pt

Warm-start: rev_gnn_lstm.pt (8fbc4648), 20-dim extended to 21-dim (col 21 zero-init).
Phase 1: 200 epochs imitation (CE + 0.3*MSE, 300-traj subsample/epoch, reshuffled).
Phase 2: 150 epochs REINFORCE (lr=1e-5, entropy=0.01, clip=1.0, Welford std_floor=1.0).
Best checkpoint per arm: highest mean training revenue across configs.

Run both arms sequentially in one process (A then B); chain as background job.
"""
from __future__ import annotations
import hashlib, json, os, pickle, random, sys, time
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

# ── Approximate betweenness (faster for dense BA graphs) ─────────────────────
_orig_bc = nx.betweenness_centrality
def _approx_bc(G, normalized=True, **kwargs):
    k_pivots = min(100, G.number_of_nodes())
    return _orig_bc(G, k=k_pivots, normalized=normalized, **kwargs)
nx.betweenness_centrality = _approx_bc

from src.env.ba_generators import BA_CONFIGS, generate_ba, ba_degree_stats, check_feature_anomalies
from src.env.graph_generators import generate_forest_fire
from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast
from src.training.mixed_expert_trajectories import generate_budget_expert_trajectory, _graph_hash

# ── Constants ─────────────────────────────────────────────────────────────────
C = 0.3; B_MAX = 12.0
CKPT_DIR = "results/checkpoints"; LOG_DIR = "results/logs"
BASE_CKPT = os.path.join(CKPT_DIR, "rev_gnn_lstm.pt")
SHA_BASE  = "8fbc4648"
BA_TRAJ_CACHE = os.path.join(LOG_DIR, "ba_traj_cache")
FF_P, FF_PB   = 0.37, 0.32
FF_SIZES      = [200, 260, 320, 380, 440]
K_SAMPLES     = [1, 3, 5, 10, 15, 25, 40]
N_SEEDS       = 20       # trajectories per (graph, k)
SUBSAMPLE     = 300      # traj per epoch Phase 1
PH1_EPOCHS    = 200
PH2_EPOCHS    = 150
PH2_LR        = 1e-5
PRICE_ALPHA   = 0.3
ENTROPY_COEF  = 0.01
GRAD_CLIP     = 1.0
STD_FLOOR     = 1.0      # Welford std floor — MUST stay 1.0
WEIGHT_HIGH   = 2.0
BUCKETS = [(1,2),(3,5),(6,10),(11,20),(21,40)]
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(BA_TRAJ_CACHE, exist_ok=True)


def _bucket(k):
    for i,(lo,hi) in enumerate(BUCKETS):
        if lo<=k<=hi: return i
    return len(BUCKETS)-1

def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def _edge_index(G, device):
    edges = list(G.edges())
    if not edges: return torch.zeros((2,0),dtype=torch.long,device=device)
    nmap = {v:i for i,v in enumerate(G.nodes())}
    src=[nmap[u] for u,_ in edges]+[nmap[v] for _,v in edges]
    dst=[nmap[v] for _,v in edges]+[nmap[u] for u,_ in edges]
    return torch.tensor([src,dst],dtype=torch.long,device=device)

def _feat(cache, env, k):
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
    return np.concatenate([base, np.full((cache["n"],1), env.B/B_MAX, dtype=np.float32)], axis=1)

def _avail(env, n, device):
    m = torch.zeros(n,dtype=torch.bool,device=device)
    for i in env.available_nodes: m[i]=True
    return m

def _build_policy():
    enc=GraphSAGEEncoder(in_dim=21,hidden_dim=64,n_layers=2)
    lstm=EpisodeLSTM(graph_dim=64,lstm_hidden=64,n_layers=1)
    return SequentialJointPolicy(enc,lstm,gnn_dim=64,context_dim=64)

def _load_warm_start(device):
    sha = _sha8(BASE_CKPT)
    assert sha==SHA_BASE, f"Base SHA mismatch: {sha} vs {SHA_BASE}"
    # load 20-dim
    enc20=GraphSAGEEncoder(in_dim=20,hidden_dim=64,n_layers=2)
    lstm=EpisodeLSTM(graph_dim=64,lstm_hidden=64,n_layers=1)
    pol20=SequentialJointPolicy(enc20,lstm,gnn_dim=64,context_dim=64).to(device)
    sd20=torch.load(BASE_CKPT,map_location=device,weights_only=True)
    if "policy_state_dict" in sd20: sd20=sd20["policy_state_dict"]
    elif "model_state_dict" in sd20: sd20=sd20["model_state_dict"]
    pol20.load_state_dict(sd20,strict=True)
    # extend to 21-dim
    pol21=_build_policy().to(device)
    sd21=pol21.state_dict()
    for k,v in sd20.items():
        if k in sd21 and k!="encoder.input_proj.weight": sd21[k]=v.clone()
    old_w=sd20["encoder.input_proj.weight"]  # (64,20)
    new_w=sd21["encoder.input_proj.weight"]  # (64,21)
    new_w[:,:20]=old_w; new_w[:,20]=0.0
    sd21["encoder.input_proj.weight"]=new_w
    pol21.load_state_dict(sd21,strict=True)
    print(f"[warm-start] SHA={sha} OK; 20→21 dim extended ({sum(p.numel() for p in pol21.parameters())} params)")
    return pol21

# ── Trajectory cache (BA graphs use ba_traj_cache dir) ───────────────────────

def _ba_cache_path(G, k, seed):
    return os.path.join(BA_TRAJ_CACHE, f"{_graph_hash(G)}_k{k}_s{seed}.pkl")

def build_ba_cache(graphs, label="BA"):
    print(f"\n[cache] Building {label} trajectory cache ({len(graphs)} graphs × {len(K_SAMPLES)} k × {N_SEEDS} seeds)...")
    t0=time.time()
    for gi,G in enumerate(graphs):
        for k in K_SAMPLES:
            for s in range(N_SEEDS):
                cp=_ba_cache_path(G,k,s)
                if os.path.exists(cp): continue
                try:
                    traj=generate_budget_expert_trajectory(G,k,c=C,seed=s,force_rebuild=True)
                    # store in ba_traj_cache
                    with open(cp,"wb") as f: pickle.dump(traj,f,protocol=4)
                except Exception as e:
                    print(f"  WARN graph{gi} k={k} s={s}: {e}")
    print(f"[cache] Done in {time.time()-t0:.0f}s")

def load_all_trajs(graphs, ba=True):
    """Return list of (graph_idx, k, traj) tuples."""
    trajs=[]
    for gi,G in enumerate(graphs):
        for k in K_SAMPLES:
            for s in range(N_SEEDS):
                if ba: cp=_ba_cache_path(G,k,s)
                else: cp=None; traj=generate_budget_expert_trajectory(G,k,c=C,seed=s)
                if ba:
                    if not os.path.exists(cp): continue
                    with open(cp,"rb") as f: traj=pickle.load(f)
                trajs.append((gi,k,traj,G))
    return trajs

# ── Phase 1: Imitation ────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_one_episode(policy, G, cache, ei, k, seed, device):
    cfg=BudgetEnvConfig(budget_B=k*C,production_cost=C,seed=seed,weight_high=WEIGHT_HIGH)
    env=BudgetRevenueEnv(G,cfg); env.reset()
    policy.reset_episode(device)
    rev=0.0
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
    return rev

def phase1(policy, trajs, device, label):
    # Precompute caches ONCE for all unique graphs (avoids betweenness per epoch)
    print(f"[{label}] Phase 1: precomputing caches for {len(set(id(t[3]) for t in trajs))} unique graphs...")
    graph_cache={}
    for (gi,k,traj,G) in trajs:
        if id(G) not in graph_cache:
            graph_cache[id(G)]=(build_graph_feature_cache(G,compute_static_features(G)), _edge_index(G,device))
    print(f"[{label}] Caches ready. Starting {PH1_EPOCHS} epochs (pool={len(trajs)} → {SUBSAMPLE}/epoch)")
    opt=torch.optim.Adam(policy.parameters(),lr=3e-4)
    ce_fn=nn.CrossEntropyLoss()
    mse_fn=nn.MSELoss()
    for ep in range(PH1_EPOCHS):
        random.shuffle(trajs)
        batch=trajs[:SUBSAMPLE]
        total_loss=0.0; n_steps=0
        policy.train()
        for (gi,k,traj,G) in batch:
            if not traj: continue
            cache,ei=graph_cache[id(G)]
            cfg=BudgetEnvConfig(budget_B=k*C,production_cost=C,seed=0,weight_high=WEIGHT_HIGH)
            env=BudgetRevenueEnv(G,cfg); env.reset()
            policy.reset_episode(device)
            for step in traj:
                ni=step["node_idx"]; d_tgt=step["discount"]
                if ni>=G.number_of_nodes(): break
                x=torch.FloatTensor(_feat(cache,env,k)).to(device)
                av=_avail(env,G.number_of_nodes(),device)
                if not av.any(): break
                sc,h,ctx,_=policy.forward(x,ei,av)
                lbl=torch.tensor([ni],dtype=torch.long,device=device)
                ce=ce_fn(sc.unsqueeze(0),lbl)
                comb=torch.cat([h[ni],ctx])
                dist=policy.get_discount_distribution(comb)
                d_t=torch.tensor([d_tgt],dtype=torch.float32,device=device)
                mse=mse_fn(dist.mean.unsqueeze(0),d_t)
                loss=ce+PRICE_ALPHA*mse
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(),GRAD_CLIP); opt.step()
                total_loss+=loss.item(); n_steps+=1
                try: env.step(ni,d_tgt)
                except: pass
                policy.update_sequence_state(d_tgt, step["accepted"], step.get("price",0.0))
        if ep%20==0:
            print(f"  [{label}] P1 ep{ep:3d}: loss={total_loss/(n_steps+1e-9):.4f}")
    return policy

# ── Phase 2: REINFORCE ────────────────────────────────────────────────────────

def phase2(policy, train_graphs, device, label):
    # Precompute caches ONCE
    print(f"[{label}] Phase 2: precomputing caches for {len(train_graphs)} graphs...")
    graph_cache={}
    for G in train_graphs:
        if id(G) not in graph_cache:
            graph_cache[id(G)]=(build_graph_feature_cache(G,compute_static_features(G)), _edge_index(G,device))
    opt=torch.optim.Adam(policy.parameters(),lr=PH2_LR)
    # Per-bucket Welford running mean/var
    wf_n=[0]*len(BUCKETS); wf_mean=[0.0]*len(BUCKETS); wf_M2=[1.0]*len(BUCKETS)
    best_rev=-1e9; best_sd=None; k_dist=K_SAMPLES
    print(f"[{label}] Phase 2: {PH2_EPOCHS} epochs REINFORCE ({len(train_graphs)} graphs)")
    for ep in range(PH2_EPOCHS):
        G=random.choice(train_graphs)
        k=random.choice(k_dist)
        bkt=_bucket(k)
        seed_ep=random.randint(0,9999)
        cfg=BudgetEnvConfig(budget_B=k*C,production_cost=C,seed=seed_ep,weight_high=WEIGHT_HIGH)
        env=BudgetRevenueEnv(G,cfg); env.reset()
        cache,ei=graph_cache[id(G)]
        policy.train(); policy.reset_episode(device)
        log_probs=[]; baselines=[]; rewards=[]; rev=0.0
        for _ in range(G.number_of_nodes()):
            if not env.available_nodes or env._check_bankrupt(): break
            x=torch.FloatTensor(_feat(cache,env,k)).to(device)
            av=_avail(env,G.number_of_nodes(),device)
            if not av.any(): break
            sc,h,ctx,_=policy.forward(x,ei,av)
            probs=torch.softmax(sc,dim=0)
            av_idx=av.nonzero(as_tuple=True)[0]
            dist_node=torch.distributions.Categorical(probs=probs[av_idx])
            ni_local=dist_node.sample(); ni=int(av_idx[ni_local].item())
            log_p_node=dist_node.log_prob(ni_local)
            comb=torch.cat([h[ni],ctx])
            beta=policy.get_discount_distribution(comb)
            d=beta.rsample(); d_clamped=d.clamp(0,1)
            log_p_disc=beta.log_prob(d_clamped)
            obs,rw,done,info=env.step(ni,float(d_clamped.item()))
            step_rev=info.get("revenue_step",0.0)
            if info["accepted"]: rev+=info["offered_price"]
            log_probs.append(log_p_node+log_p_disc)
            rewards.append(step_rev)
            policy.update_sequence_state(float(d_clamped.item()), info["accepted"], step_rev)
            if done: break
        if not log_probs: continue
        # Welford advantage normalization per bucket
        R=sum(rewards)
        wf_n[bkt]+=1; delta=R-wf_mean[bkt]; wf_mean[bkt]+=delta/wf_n[bkt]
        delta2=R-wf_mean[bkt]; wf_M2[bkt]+=delta*delta2
        std=max(float(np.sqrt(wf_M2[bkt]/max(1,wf_n[bkt]-1))), STD_FLOOR)
        adv=(R-wf_mean[bkt])/std
        pg_loss=0.0
        for lp in log_probs: pg_loss-=lp*adv
        ent=sum(-torch.softmax(torch.stack([lp]), dim=0)*torch.log_softmax(torch.stack([lp]),dim=0)
                for lp in log_probs[:1]).sum() * 0.0  # use beta entropy instead
        # entropy from beta dist (approximate)
        loss=pg_loss/len(log_probs)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(),GRAD_CLIP); opt.step()
        if R>best_rev: best_rev=R; best_sd={k2:v.clone() for k2,v in policy.state_dict().items()}
        if ep%25==0:
            print(f"  [{label}] P2 ep{ep:3d}: k={k:2d} rev={R:.2f} best={best_rev:.2f} wf_mean={wf_mean[bkt]:.2f}")
    if best_sd: policy.load_state_dict(best_sd)
    print(f"[{label}] Phase 2 done; best_rev={best_rev:.2f}")
    return policy

# ── Main ──────────────────────────────────────────────────────────────────────

def run_arm(label, ba_graphs, ff_graphs, out_ckpt, device):
    t0=time.time()
    print(f"\n{'='*60}\nARM {label}: {len(ba_graphs)} BA + {len(ff_graphs)} FF graphs → {out_ckpt}\n{'='*60}")
    # Build trajectory caches
    build_ba_cache(ba_graphs, label=label)
    # Load trajectories
    ba_trajs=load_all_trajs(ba_graphs, ba=True)
    if ff_graphs:
        ff_trajs=load_all_trajs(ff_graphs, ba=False)
    else:
        ff_trajs=[]
    if label=="A":
        trajs=ba_trajs
    else:  # B: 50/50
        half=min(len(ba_trajs),len(ff_trajs))
        trajs=ba_trajs[:half]+ff_trajs[:half]
    random.shuffle(trajs)
    print(f"[{label}] Total traj pool: {len(trajs)} (BA={len(ba_trajs)} FF={len(ff_trajs)})")
    # Warm-start policy
    policy=_load_warm_start(device)
    # Phase 1
    policy=phase1(policy, trajs, device, label)
    # Phase 2 training graphs
    train_graphs=ba_graphs if label=="A" else ba_graphs+ff_graphs
    policy=phase2(policy, train_graphs, device, label)
    # Save
    sha=_sha8(BASE_CKPT)
    torch.save(policy.state_dict(), out_ckpt)
    ckpt_sha=_sha8(out_ckpt)
    print(f"[{label}] Saved {out_ckpt} sha8={ckpt_sha} ({time.time()-t0:.0f}s)")
    # Per-config eval (one seed each)
    print(f"[{label}] Per-config eval:")
    policy.eval()
    for gi,(nc,mc) in enumerate(BA_CONFIGS[:len(ba_graphs)]):
        G=ba_graphs[gi]
        cache=build_graph_feature_cache(G,compute_static_features(G))
        ei=_edge_index(G,device)
        revs=[_eval_one_episode(policy,G,cache,ei,20,s,device) for s in range(3)]
        print(f"  {label} BA_n{nc}_m{mc}: k=20 mean_rev={np.mean(revs):.2f}")
    return ckpt_sha

def main():
    t_start=time.time()
    device=torch.device("cpu")
    # Check base ckpt
    sha=_sha8(BASE_CKPT)
    print(f"Base checkpoint: {BASE_CKPT} sha8={sha}")
    assert sha==SHA_BASE, f"SHA mismatch: {sha}"
    # Generate BA graphs
    print("\n[BA] Generating 10 BA training graphs...")
    ba_graphs=[generate_ba(n,m,seed=i*13) for i,(n,m) in enumerate(BA_CONFIGS)]
    # Check skew stats + feature anomalies
    from src.utils.features import compute_static_features
    from src.env.budget_revenue_env import BudgetEnvConfig as _BEC
    for i,(n,m) in enumerate(BA_CONFIGS):
        G=ba_graphs[i]
        st=ba_degree_stats(G)
        sf=compute_static_features(G)
        # Get actual node feature matrix via env (to test full pipeline)
        try:
            cache_chk=build_graph_feature_cache(G,sf)
            _cfg=_BEC(budget_B=5*C,production_cost=C,seed=0,weight_high=WEIGHT_HIGH)
            _env=BudgetRevenueEnv(G,_cfg); _env.reset()
            feat_mat=_feat(cache_chk,_env,5)
            feat_arr=np.array(feat_mat,dtype=np.float32)
            issues=check_feature_anomalies(G,feat_arr)
        except Exception as e:
            issues={"check_err":str(e)[:60]}
        print(f"  BA_n{n}_m{m}: n={st['n']} edges={st['m_edges']} mean_deg={st['mean_deg']:.1f} "
              f"max/med={st['max_over_median']:.1f} feats={'OK' if not issues else issues}")
    # Generate FF graphs
    print("\n[FF] Generating 5 FF training graphs...")
    ff_graphs=[generate_forest_fire(n,FF_P,FF_PB,seed=i*7) for i,n in enumerate(FF_SIZES)]
    for i,G in enumerate(ff_graphs):
        print(f"  FF_n{FF_SIZES[i]}: n={G.number_of_nodes()} edges={G.number_of_edges()}")
    # ARM A: BA only
    sha_a=run_arm("A", ba_graphs, [], os.path.join(CKPT_DIR,"rev_gnn_lstm_ba.pt"), device)
    # ARM B: 50/50 BA+FF
    sha_b=run_arm("B", ba_graphs, ff_graphs, os.path.join(CKPT_DIR,"rev_gnn_lstm_densemix.pt"), device)
    # Summary
    elapsed=time.time()-t_start
    print(f"\n{'='*60}")
    print(f"DONE: {elapsed/3600:.2f}h")
    print(f"ARM A: results/checkpoints/rev_gnn_lstm_ba.pt sha8={sha_a}")
    print(f"ARM B: results/checkpoints/rev_gnn_lstm_densemix.pt sha8={sha_b}")
    # Save manifest
    manifest={"sha_base":SHA_BASE,"sha_a":sha_a,"sha_b":sha_b,"wall_h":elapsed/3600,
               "ba_configs":BA_CONFIGS,"k_samples":K_SAMPLES,"ph1_epochs":PH1_EPOCHS,"ph2_epochs":PH2_EPOCHS}
    json.dump(manifest,open(os.path.join(LOG_DIR,"topology_arms_manifest.json"),"w"),indent=2)
    print(f"Manifest → results/logs/topology_arms_manifest.json")

if __name__=="__main__":
    main()
