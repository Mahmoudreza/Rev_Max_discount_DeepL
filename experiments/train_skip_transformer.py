#!/usr/bin/env python3
"""train_skip_transformer.py — Budget Transformer + skip connection from raw features.

WHY: Linear probe shows the GNN encoder discards pricing-relevant
information (encoder h R^2=0.15 vs raw-feats R^2=0.97 for v_hat).
The fix is a direct path from raw features to the decision heads.

CHANGE vs train_budget_transformer.py:
  z_i = [ h_i , c_t , s_i ]   (64 + 64 + 4 = 132)
  s_i = feats21[:, SKIP_IDX]  = [v_hat_i, x_i, deg_norm, B_t/B_MAX]
  Both selection head and pricing head use 132-dim input.
  SKIP_IDX = [17, 16, 0, 20]  (verified by sanity_check — run before training)

Checkpoints:
  results/checkpoints/transf_skip_s{SEED}_p1_ep{EP:04d}.pt
  results/checkpoints/transf_skip_s{SEED}_p2_ep{EP:04d}.pt
  README.md appended every 20th epoch and for best.

Protocol: monotone + flat + Uniform(0,2), n_mc=200, c=0.3.

Usage (3 seeds):
  for S in 0 1 2; do
    nohup venv/bin/python3 -u experiments/train_skip_transformer.py \\
      --seed $S --device cuda:$S \\
      > /tmp/skip_s${S}.log 2>&1 &
    echo "Seed $S PID=$!"
  done
"""
from __future__ import annotations
import argparse, hashlib, math, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.utils.helpers import set_seed, load_config_with_base
from src.utils.features import (compute_static_features, build_graph_feature_cache,
                                 compute_node_features_fast)
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.episode_transformer import EpisodeTransformerSliding
from src.training.mixed_expert_trajectories import generate_budget_expert_trajectory

_ROOT     = str(Path(__file__).parent.parent)
CFG_TFM   = os.path.join(_ROOT, "configs/experiments/rev_gnn_transformer_300ep.yaml")
BASE_CKPT = os.path.join(_ROOT, "results/checkpoints/rev_gnn_transformer.pt")
BASE_SHA  = "c24215b8"
CKPT_DIR  = os.path.join(_ROOT, "results/checkpoints")
README    = os.path.join(CKPT_DIR, "README.md")

C           = 0.3
B_MAX       = 40 * C
FF_P, FF_PB = 0.37, 0.32
TRAIN_SIZES = [200, 260, 320, 380, 440]
W_HIGH      = 2.0
N_MC        = 200
N_MC_P1     = 10
PH1_EPOCHS  = 300
PH1_LR      = 1e-4
PRICE_ALPHA = 0.3
K_P1        = [1, 3, 5, 10, 15, 25, 40]
N_SEEDS_P1  = 10
PH2_EPOCHS  = 200
PH2_LR      = 5e-5
ENTROPY     = 0.01
GRAD_CLIP   = 1.0
STD_FLOOR   = 1.0

# ── Skip connection ────────────────────────────────────────────────────────────
# Feature column indices in the 21-dim vector (0-indexed):
#   0-9  : static (deg_rank, cc, bc, pr, kc, ec, tc, cl, ecc, and)
#   10-15: dynamic WSDM (seed_flag, round_ratio, hop1, log_deg, clust_rep, group)
#   16   : current_influence (x_i)
#   17   : current_valuation (v_hat_i)  ← already one of the 21 inputs
#   18   : was_offered
#   19   : steps_remaining
#   20   : B_t/B_MAX  (budget column added by _features())
SKIP_IDX  = [17, 16, 0, 20]   # [v_hat_i, x_i, deg_norm, B_t/B_MAX]
SKIP_DIM  = 4
SKIP_NAMES = ["v_hat(17)", "x_i(16)", "deg(0)", "Bt/Bmax(20)"]

BUCKETS = [(1,2),(3,5),(6,10),(11,20),(21,40)]
def _bucket(k):
    for i,(lo,hi) in enumerate(BUCKETS):
        if lo<=k<=hi: return i
    return len(BUCKETS)-1


# ── Skip policy ────────────────────────────────────────────────────────────────

class TransformerSkipPolicy(nn.Module):
    """TransformerJointPolicy + 4-dim raw-feature skip connection.

    z_i = [h_i (64) ‖ c_t (64) ‖ s_i (4)] = 132-dim
    Selection head: Linear(132,132)→ReLU→Linear(132,1)
    Pricing head:   Linear(132,2) → Beta(α,β)
    Param count vs base: +882 (negligible).
    """
    SKIP_IDX = SKIP_IDX

    def __init__(self, encoder, seq_module, gnn_dim=64, context_dim=64,
                 skip_dim=SKIP_DIM):
        super().__init__()
        self.encoder     = encoder
        self.seq_module  = seq_module
        self.gnn_dim     = gnn_dim
        self.context_dim = context_dim
        self.skip_dim    = skip_dim
        combined_dim     = gnn_dim + context_dim + skip_dim   # 132
        self.scorer = nn.Sequential(
            nn.Linear(combined_dim, combined_dim), nn.ReLU(),
            nn.Linear(combined_dim, 1))
        self.pricing_head = nn.Linear(combined_dim, 2)
        self._last_discount: float = 0.0
        self._last_accepted: bool  = False
        self._last_revenue:  float = 0.0
        self._hidden = None

    def reset_episode(self, device):
        self.seq_module.reset_episode(device)
        self._last_discount = 0.0; self._last_accepted = False
        self._last_revenue  = 0.0; self._hidden = None

    def update_sequence_state(self, discount, accepted, revenue):
        self._last_discount = discount; self._last_accepted = accepted
        self._last_revenue  = revenue
        self.seq_module.update_sequence_state(discount, accepted, revenue)

    def forward(self, x, edge_index, mask, skip_feats):
        """
        x:          (n, 21) node features
        skip_feats: (n, 4)  raw skip vector [v_hat, x_i, deg, B_t/B_MAX]
        Returns: (scores (n,), h (n,64), context (64,), None)
        """
        h = self.encoder(x, edge_index)               # (n, 64)
        graph_emb = h.mean(dim=0)                     # (64,)
        context, self._hidden = self.seq_module.step(
            graph_emb, self._last_discount,
            self._last_accepted, self._last_revenue, hidden=None)
        ctx_exp  = context.unsqueeze(0).expand(h.shape[0], -1)   # (n, 64)
        combined = torch.cat([h, ctx_exp, skip_feats], dim=-1)   # (n, 132)
        scores   = self.scorer(combined).squeeze(-1)              # (n,)
        scores   = scores.masked_fill(~mask, float("-inf"))
        return scores, h, context, None

    def get_discount_distribution(self, combined_132):
        """combined_132: (132,) = [h_i ‖ ctx ‖ s_i]"""
        raw = self.pricing_head(combined_132)
        return Beta(F.softplus(raw[0])+1.0, F.softplus(raw[1])+1.0)


# ── Welford ───────────────────────────────────────────────────────────────────

class WelfordBucket:
    def __init__(self):
        self.n=0; self.m1=0.0; self.m2=0.0
    def update(self,x):
        self.n+=1; d=x-self.m1; self.m1+=d/self.n; self.m2+=d*(x-self.m1)
    @property
    def mean(self): return self.m1
    @property
    def std(self): return max(math.sqrt(self.m2/max(self.n-1,1)), STD_FLOOR)
    def normalise(self,x): return (x-self.mean)/self.std


# ── Feature helpers ───────────────────────────────────────────────────────────

def _features(cache, env, k):
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
    bcol = np.full((cache["n"],1), env.B/B_MAX, dtype=np.float32)
    return np.concatenate([base, bcol], axis=1)   # (n, 21)

def _skip(feats21_np, device):
    """Extract (n,4) skip vector from numpy (n,21) feats."""
    s = feats21_np[:, SKIP_IDX].copy()
    return torch.tensor(s, dtype=torch.float32, device=device)

def _edge_index(G, device):
    edges=list(G.edges())
    if not edges: return torch.zeros((2,0),dtype=torch.long,device=device)
    m={v:i for i,v in enumerate(G.nodes())}
    s=[m[u] for u,_ in edges]+[m[v] for _,v in edges]
    d=[m[v] for _,v in edges]+[m[u] for u,_ in edges]
    return torch.tensor([s,d],dtype=torch.long,device=device)

def _avail(env, n, device):
    mask=torch.zeros(n,dtype=torch.bool,device=device)
    for idx in env.available_nodes: mask[idx]=True
    return mask


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check():
    """Print feature indices and one sample row to confirm SKIP_IDX is correct."""
    print("\n=== SANITY CHECK: skip feature indices ===")
    print(f"SKIP_IDX = {SKIP_IDX}  →  {SKIP_NAMES}")
    G = generate_forest_fire(100, FF_P, FF_PB, seed=0)
    cache = build_graph_feature_cache(G, compute_static_features(G))
    cfg = BudgetEnvConfig(budget_B=10*C, production_cost=C, seed=0,
                          weight_high=W_HIGH, n_mc_samples=50)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    f21 = _features(cache, env, k=10)
    node0 = f21[0]
    print(f"  feats21[0] shape={f21.shape}  node0={node0.tolist()[:5]}…")
    print(f"  skip[0]:  {node0[SKIP_IDX[0]]:.4f}={SKIP_NAMES[0]}  "
          f"{node0[SKIP_IDX[1]]:.4f}={SKIP_NAMES[1]}  "
          f"{node0[SKIP_IDX[2]]:.4f}={SKIP_NAMES[2]}  "
          f"{node0[SKIP_IDX[3]]:.4f}={SKIP_NAMES[3]}")
    # Cross-check v_hat (col 17) against MC estimate
    nodes = list(G.nodes())
    try:
        v_hat_direct = env._estimate_valuation(nodes[0])
        print(f"  v_hat direct MC={v_hat_direct:.4f}  feats21[0,17]={node0[17]:.4f}  "
              f"match={'YES' if abs(v_hat_direct - node0[17]) < 0.05 else 'MISMATCH'}")
    except Exception as e:
        print(f"  v_hat direct: {e}")
    print("=== END SANITY CHECK ===\n", flush=True)


# ── Model init ────────────────────────────────────────────────────────────────

def _build_skip_policy(device):
    sha = hashlib.sha256(open(BASE_CKPT,"rb").read()).hexdigest()
    assert sha.startswith(BASE_SHA), f"ABORT base sha={sha[:8]}"
    cfg_t = load_config_with_base(CFG_TFM)
    in_dim = int(cfg_t.features.dim)   # 20
    H, NL, DO = (int(cfg_t.encoder.hidden_dim),
                 int(cfg_t.encoder.n_layers), float(cfg_t.encoder.dropout))
    enc = GraphSAGEEncoder(in_dim+1, H, NL, DO)  # 21-dim input
    tfm = EpisodeTransformerSliding.from_config(cfg_t.transformer)
    pol = TransformerSkipPolicy(enc, tfm, gnn_dim=H,
                                context_dim=tfm.context_dim,
                                skip_dim=SKIP_DIM).to(device)
    # Load base ckpt, extend input_proj by 1 (zero-init), skip heads (new dims)
    sd_old = torch.load(BASE_CKPT, map_location=device, weights_only=True)
    if isinstance(sd_old, dict) and "state_dict" in sd_old: sd_old = sd_old["state_dict"]
    sd_new = pol.state_dict()
    n_loaded = 0
    for k, v in sd_old.items():
        if k in sd_new and k != "encoder.input_proj.weight":
            if sd_new[k].shape == v.shape:
                sd_new[k] = v.clone(); n_loaded += 1
    old_w = sd_old["encoder.input_proj.weight"]  # (H, 20)
    new_w = sd_new["encoder.input_proj.weight"]  # (H, 21)
    new_w[:, :in_dim] = old_w; new_w[:, in_dim] = 0.0
    sd_new["encoder.input_proj.weight"] = new_w
    pol.load_state_dict(sd_new, strict=True)
    n_params = sum(p.numel() for p in pol.parameters())
    print(f"[init] skip policy built  params={n_params:,}  "
          f"(heads: 132-dim; +882 vs base)  n_loaded={n_loaded}", flush=True)
    return pol


# ── Traj cache ────────────────────────────────────────────────────────────────

def _build_traj_cache(graphs, seed):
    import pickle
    path = os.path.join(_ROOT, f"results/logs/traj_cache_budget_s{seed}.pkl")
    key  = str(("budget_expert","monotone","flat",W_HIGH,K_P1,N_SEEDS_P1,
                 [g.number_of_nodes() for g in graphs]))
    if os.path.exists(path):
        try:
            sk, tc = pickle.load(open(path,"rb"))
            if sk==key and len(tc)>0:
                print(f"[traj] HIT {len(tc)}"); return tc
        except Exception: pass
    print("[traj] MISS — building…", flush=True)
    tc={}
    for gi,G in enumerate(graphs):
        for k in K_P1:
            for s in range(N_SEEDS_P1):
                try:
                    t=generate_budget_expert_trajectory(G,k,c=C,seed=s)
                    if t: tc[(gi,k,s)]=t
                except Exception: pass
    pickle.dump((key,tc),open(path,"wb"))
    print(f"[traj] {len(tc)} saved"); return tc


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def phase1(pol, graphs, traj_cache, device, seed, save_prefix):
    opt = torch.optim.Adam(pol.parameters(), lr=PH1_LR, weight_decay=1e-5)
    graph_caches = [build_graph_feature_cache(G,compute_static_features(G)) for G in graphs]
    graph_eis    = [_edge_index(G,device) for G in graphs]
    graph_nodes  = [list(G.nodes()) for G in graphs]
    print(f"[P1-s{seed}] {PH1_EPOCHS} ep", flush=True)

    for ep in range(1, PH1_EPOCHS+1):
        epoch_loss=0.0; epoch_steps=0
        for gi,G in enumerate(graphs):
            cache=graph_caches[gi]; ei_t=graph_eis[gi]
            nodes=graph_nodes[gi]; n=len(nodes)
            for k in K_P1:
                B0=k*C
                for s in range(N_SEEDS_P1):
                    traj=traj_cache.get((gi,k,s))
                    if not traj: continue
                    pol.reset_episode(device)
                    cfg=BudgetEnvConfig(budget_B=B0,production_cost=C,seed=s,
                                       n_mc_samples=N_MC_P1)
                    env=BudgetRevenueEnv(G,cfg); env.reset()
                    step_losses=[]
                    for step in traj:
                        ei=step["node_idx"]; ed=step["discount"]
                        if ei not in env.available_nodes: break
                        f21=_features(cache,env,k)
                        x  =torch.tensor(f21,dtype=torch.float32,device=device)
                        sk =_skip(f21,device)
                        av =_avail(env,n,device)
                        if not av.any(): break
                        scores,h,ctx,_=pol.forward(x,ei_t,av,sk)
                        safe=scores.clone(); safe[~av]=-1e9
                        ce = -torch.log_softmax(safe,dim=-1)[ei]
                        z_i = torch.cat([h[ei],ctx,sk[ei]])
                        pd  = pol.get_discount_distribution(z_i)
                        mse = (pd.mean - torch.tensor(ed,dtype=torch.float32,device=device))**2
                        step_losses.append(ce + PRICE_ALPHA*mse)
                        pol.update_sequence_state(ed,step["accepted"],step["price"])
                        _,_,done,_=env.step(ei,ed)
                        if done: break
                    if step_losses:
                        loss=torch.stack(step_losses).mean()
                        opt.zero_grad(); loss.backward()
                        nn.utils.clip_grad_norm_(pol.parameters(),1.0)
                        opt.step()
                        epoch_loss+=loss.item(); epoch_steps+=1

        avg=epoch_loss/max(epoch_steps,1)
        sp=save_prefix.replace("_ep.pt",f"_p1_ep{ep:04d}.pt")
        torch.save(pol.state_dict(),sp)
        if ep%20==0: _append_readme(sp)
        print(f"[P1-s{seed}] ep={ep}/{PH1_EPOCHS}  loss={avg:.4f}  saved {os.path.basename(sp)}",flush=True)
        old_ep=ep-21
        old_sp=save_prefix.replace("_ep.pt",f"_p1_ep{old_ep:04d}.pt")
        if old_ep>0 and old_ep%20!=0 and os.path.exists(old_sp): os.remove(old_sp)
    return pol


# ── Rollout ────────────────────────────────────────────────────────────────────

def _rollout(pol, G, cache, ei_t, B0, k, s, device):
    set_seed(s)
    cfg=BudgetEnvConfig(budget_B=B0,production_cost=C,seed=s,
                        weight_high=W_HIGH,n_mc_samples=N_MC)
    env=BudgetRevenueEnv(G,cfg); env.reset()
    nodes=list(G.nodes()); n=len(nodes)
    pol.reset_episode(device); revenue=0.0; lps=[]; ents=[]
    while env.available_nodes and not env._check_bankrupt():
        f21=_features(cache,env,k)
        x  =torch.tensor(f21,dtype=torch.float32,device=device)
        sk =_skip(f21,device)
        av =_avail(env,n,device)
        if not av.any(): break
        with torch.enable_grad():
            scores,h,ctx,_=pol.forward(x,ei_t,av,sk)
            safe=scores.clone(); safe[~av]=-1e9
            probs=torch.softmax(safe,dim=-1)
            dist=torch.distributions.Categorical(probs=probs[av])
            si=dist.sample()
            glob=av.nonzero(as_tuple=True)[0][si]
            lps.append(dist.log_prob(si)); ents.append(dist.entropy())
            z_i=torch.cat([h[int(glob)],ctx,sk[int(glob)]])
            disc=float(pol.get_discount_distribution(z_i).mean.clamp(1e-4,1-1e-4).detach())
        _,r,done,_=env.step(env.node_to_idx[nodes[int(glob)]],disc)
        revenue+=r; pol.update_sequence_state(disc,r>0,r)
        if done: break
    return revenue, lps, ents


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def phase2(pol, graphs, device, seed, save_prefix):
    assert STD_FLOOR==1.0
    opt=torch.optim.Adam(pol.parameters(),lr=PH2_LR,weight_decay=1e-5)
    welfords=[WelfordBucket() for _ in BUCKETS]
    rng=np.random.default_rng(seed=seed+123)
    best_min=-1e9; best_sd=None; best_ep=0
    caches={gi:build_graph_feature_cache(G,compute_static_features(G))
            for gi,G in enumerate(graphs)}
    eis={gi:_edge_index(G,device) for gi,G in enumerate(graphs)}
    print(f"\n[P2-s{seed}] {PH2_EPOCHS} ep  STD_FLOOR={STD_FLOOR}", flush=True)

    for ep in range(1, PH2_EPOCHS+1):
        adv_lp_ent=[]; bucket_revs=[[] for _ in BUCKETS]
        for gi,G in enumerate(graphs):
            k=int(np.exp(rng.uniform(math.log(1),math.log(40))))
            k=max(1,min(40,k)); B0=k*C; s=int(rng.integers(0,1000))
            rev,lps,ents=_rollout(pol,G,caches[gi],eis[gi],B0,k,s,device)
            bid=_bucket(k); adv=welfords[bid].normalise(rev)
            adv_lp_ent.append((adv,lps,ents)); bucket_revs[bid].append(rev)
        for bid,revs in enumerate(bucket_revs):
            for r in revs: welfords[bid].update(r)
        if not adv_lp_ent: continue
        pol_loss=-sum(adv*sum(lp for lp in lps)
                      for adv,lps,ents in adv_lp_ent)/len(adv_lp_ent)
        ent_loss=-ENTROPY*sum(e for _,_,ents in adv_lp_ent
                              for e in ents)/max(len(adv_lp_ent),1)
        loss=pol_loss+ent_loss
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(),GRAD_CLIP); opt.step()
        bucket_nadv=[]
        for bid,revs in enumerate(bucket_revs):
            if revs and welfords[bid].n>=2:
                vals=[(r-welfords[bid].mean)/welfords[bid].std for r in revs]
                bucket_nadv.append(float(np.mean(vals)))
        if bucket_nadv:
            min_adv=min(bucket_nadv)
            if min_adv>best_min:
                best_min=min_adv
                best_sd={k:v.clone() for k,v in pol.state_dict().items()}
                best_ep=ep
        sp=save_prefix.replace("_ep.pt",f"_p2_ep{ep:04d}.pt")
        torch.save(pol.state_dict(),sp)
        if ep%20==0:
            bra=[f"{np.mean(r):.1f}" if r else "—" for r in bucket_revs]
            print(f"[P2-s{seed}] ep={ep}/{PH2_EPOCHS}  "
                  f"min_adv={bucket_nadv and min(bucket_nadv) or 0:.3f}  "
                  f"bucket_rev={bra}", flush=True)
            _append_readme(sp)
        old_ep=ep-21
        old_sp=save_prefix.replace("_ep.pt",f"_p2_ep{old_ep:04d}.pt")
        if old_ep>0 and old_ep%20!=0 and os.path.exists(old_sp): os.remove(old_sp)
    if best_sd:
        pol.load_state_dict(best_sd)
        print(f"[P2-s{seed}] best ep={best_ep}  min_adv={best_min:.3f}", flush=True)
    return pol, best_ep


# ── README ────────────────────────────────────────────────────────────────────

def _sha8(p):
    try: return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]
    except: return "????????"

def _append_readme(path):
    sha=_sha8(path)
    line=f"| `{os.path.basename(path)}` | skip-transformer | sha={sha} |\n"
    try:
        with open(README,"a") as f: f.write(line)
    except: pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_p1",action="store_true")
    ap.add_argument("--p1_ckpt",default="")
    args=ap.parse_args()
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    sanity_check()   # ALWAYS runs — verifies SKIP_IDX before training

    graphs=[generate_forest_fire(n,FF_P,FF_PB,seed=args.seed*100+i)
            for i,n in enumerate(TRAIN_SIZES)]
    print(f"=== Skip-Transformer  seed={args.seed}  device={device} ===")
    print(f"SKIP_IDX={SKIP_IDX}  combined_dim=132  B_MAX={B_MAX}  C={C}")
    print(f"P1={PH1_EPOCHS} ep  P2={PH2_EPOCHS} ep  STD_FLOOR={STD_FLOOR}", flush=True)

    save_prefix=os.path.join(CKPT_DIR,f"transf_skip_s{args.seed}_ep.pt")

    if args.skip_p1 and args.p1_ckpt and os.path.exists(args.p1_ckpt):
        cfg_t=load_config_with_base(CFG_TFM)
        in_dim=int(cfg_t.features.dim)+1  # 21
        H,NL,DO=(int(cfg_t.encoder.hidden_dim),int(cfg_t.encoder.n_layers),
                 float(cfg_t.encoder.dropout))
        enc=GraphSAGEEncoder(in_dim,H,NL,DO)
        tfm=EpisodeTransformerSliding.from_config(cfg_t.transformer)
        pol=TransformerSkipPolicy(enc,tfm,gnn_dim=H,
                                  context_dim=tfm.context_dim,skip_dim=SKIP_DIM).to(device)
        pol.load_state_dict(torch.load(args.p1_ckpt,map_location=device,weights_only=True))
        print(f"[main] loaded P1 from {args.p1_ckpt}  sha={_sha8(args.p1_ckpt)}",flush=True)
    else:
        pol=_build_skip_policy(device)
        traj_cache=_build_traj_cache(graphs,args.seed)
        pol=phase1(pol,graphs,traj_cache,device,args.seed,save_prefix)
        p1f=save_prefix.replace("_ep.pt",f"_p1_ep{PH1_EPOCHS:04d}.pt")
        torch.save(pol.state_dict(),p1f); _append_readme(p1f)
        print(f"[main] P1 done → {p1f}  sha={_sha8(p1f)}", flush=True)

    pol,best_ep=phase2(pol,graphs,device,args.seed,save_prefix)
    best_path=save_prefix.replace("_ep.pt","_best.pt")
    torch.save(pol.state_dict(),best_path); _append_readme(best_path)
    sha=_sha8(best_path)
    print(f"\n[main] best → {best_path}  sha={sha}", flush=True)

    import subprocess
    subprocess.run(["git","add","-f",best_path,README],cwd=_ROOT)
    r=subprocess.run(["git","commit","-m",
                       f"transf_skip_s{args.seed}_best sha={sha}"],cwd=_ROOT)
    h=subprocess.run(["git","rev-parse","--short","HEAD"],
                      capture_output=True,text=True,cwd=_ROOT).stdout.strip()
    print(h,flush=True)


if __name__=="__main__":
    main()
