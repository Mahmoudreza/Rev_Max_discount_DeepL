#!/usr/bin/env python3
"""run_c1_ffba_training.py ARM_TAG [--ratio R]
Train UNCONSTRAINED (C1) model on FF+BA mixture with hub-inclusive BA configs.
ARM_TAG: 50_50 or 2to1   (controls FF:BA sampling ratio)
--ratio R: explicit FF fraction 0-1 (overrides tag default)

CHECKPOINT: warm-start from rev_gnn_lstm.pt (sha 8fbc4648, read-only).
Phase 1 (200 epochs): CE + 0.3*MSE(Beta.mean, teacher_disc), subsample 300/epoch.
Phase 2 (150 epochs): REINFORCE, lr=1e-5, entropy=0.01, grad_clip=1.0,
                      Welford baseline std_floor=1.0 (NOT 1e-8).
Saves every 20 epochs: c1_ffba_{ARM_TAG}_p1_ep{N}.pt / p2_ep{N}.pt
README.md appended (never overwritten).
"""
import argparse, hashlib, json, math, os, pickle, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx

from src.env.revenue_env import RevenueEnv, RevenueEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.env.ba_generators import BA_CONFIGS, generate_ba
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import (compute_static_features, build_graph_feature_cache,
                                 compute_node_features_fast)
from src.evaluation.baselines import greedy_discount_trajectory, _make_env

# ─── Patch: fast neighbour-ratio influence (avoids MC IC per step in features) ─
def _fast_gci(self, node):
    nb = list(self.graph.neighbors(node))
    if not nb: return 0.
    tw = sum(self._link_weights.get((node,n),0.) for n in nb)
    if tw == 0: return 0.
    return sum(self._link_weights.get((node,n),0.) for n in nb if n in self.S)/tw
RevenueEnv.get_current_influence = _fast_gci

# ─── BA configs: existing 10 + new high-skew n∈{600,800,1000} m=4 ────────────
# NOTE: EXTRA_BA used for Phase-2 rollouts only (too slow for Phase-1 traj cache)
EXTRA_BA = [(600,4),(800,4),(1000,4)]   # max/med ≈ 15-16x; closer to polblogs 27x
ALL_BA_CONFIGS   = list(BA_CONFIGS) + EXTRA_BA   # all 13 (Phase 2 sampling)
PHASE1_BA_CONFIGS = list(BA_CONFIGS)              # 10 only (Phase 1 cache, n≤440)

FF_SIZES = [200, 260, 320, 380, 440]

# ─── Approx betweenness for large graphs ─────────────────────────────────────
_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

BASE_CKPT  = "results/checkpoints/rev_gnn_lstm.pt"
SHA_BASE   = "8fbc4648"
CKPT_DIR   = "results/checkpoints"
LOG_DIR    = "results/logs"
CACHE_DIR  = "results/logs/c1_ffba_traj_cache"
README     = os.path.join(CKPT_DIR, "README.md")

HID = 64; NIM = 200; NRL = 150; PW = 0.3; SUBSAMPLE = 30
LR_P1 = 1e-3; LR_P2 = 1e-5; ENT = 0.01; GRAD_CLIP = 1.0; MC = 5

def _sha8(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:8]

def _cfg_obj():
    from types import SimpleNamespace
    inf = SimpleNamespace(b=1.0, weight_low=0., weight_high=2., n_mc_samples=MC,
                          model="monotone")
    tr  = SimpleNamespace(imitation_lr=LR_P1, reinforce_lr=LR_P2,
                          entropy_coef=ENT, grad_clip=GRAD_CLIP,
                          pricing_loss_weight=PW)
    rwd = SimpleNamespace(type="flat", gamma=1.0)
    prj = SimpleNamespace(seed=0)
    return SimpleNamespace(influence=inf, training=tr, reward=rwd, project=prj,
                           graph=SimpleNamespace(p=0.37, pb=0.32, n_nodes=500))

def _make_pol():
    enc  = GraphSAGEEncoder(in_dim=20, hidden_dim=HID, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=HID, lstm_hidden=HID, n_layers=1)
    return SequentialJointPolicy(enc, lstm, gnn_dim=HID, context_dim=HID)

def _ei(G):
    nm = {v:i for i,v in enumerate(G.nodes())}
    E  = list(G.edges())
    s  = [nm[u] for u,_ in E]+[nm[v] for _,v in E]
    d  = [nm[v] for _,v in E]+[nm[u] for u,_ in E]
    return torch.tensor([s,d], dtype=torch.long)

def _cache_path(G, ep):
    h = hashlib.md5(str(sorted(G.edges())).encode()).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f"n{G.number_of_nodes()}_{h}_ep{ep}.pkl")

# ─── Welford baseline (std_floor = 1.0 — never 1e-8) ─────────────────────────
class Welford:
    def __init__(self):
        self.n = 0; self.mu = 0.; self.M2 = 0.
    def update(self, x):
        self.n += 1; d = x - self.mu; self.mu += d/self.n; self.M2 += d*(x-self.mu)
    def std(self): return max((self.M2/max(self.n-1,1))**0.5, 1.0)  # floor=1.0
    def advantage(self, x): self.update(x); return (x - self.mu) / self.std()

# ─── Trajectory generation ───────────────────────────────────────────────────
def gen_traj(G, ep, cfg):
    """Load cached or generate + cache one teacher episode."""
    cp = _cache_path(G, ep)
    if os.path.exists(cp):
        try:
            return pickle.load(open(cp,'rb'))
        except (EOFError, pickle.UnpicklingError):
            pass   # partial write (parallel race); regenerate
    random.seed(ep); np.random.seed(ep)
    # Fresh edge weights per episode (per spec)
    G2 = G.copy()
    for u,v in G2.edges():
        G2[u][v]['weight'] = random.uniform(cfg.influence.weight_low, cfg.influence.weight_high)
    traj = greedy_discount_trajectory(G2, cfg)
    os.makedirs(CACHE_DIR, exist_ok=True)
    pickle.dump((traj, G2), open(cp,'wb'))
    return traj, G2

def build_cache(graphs, n_ep, cfg, label=""):
    stats = []
    for G in graphs:
        revs = []
        for ep in range(n_ep):
            traj, G2 = gen_traj(G, ep, cfg)
            rev = sum(td['price'] for td in traj if td.get('accepted', False))
            revs.append(rev)
            # accounting check: each node offered at most once
            seen = [td['node_idx'] for td in traj]
            assert len(seen) == len(set(seen)), f"Duplicate offer in {label} n={G.number_of_nodes()} ep={ep}"
        stats.append({"n":G.number_of_nodes(), "n_ep":n_ep, "mean_rev":round(np.mean(revs),2)})
    return stats

# ─── Phase 1 imitation ────────────────────────────────────────────────────────
def phase1(policy, ff_graphs, ba_graphs, n_ep, cfg, device, tag, log_path):
    print(f"[P1] generating trajectories ({len(ff_graphs)} FF + {len(ba_graphs)} BA) x {n_ep} eps",
          flush=True)
    all_graphs = ff_graphs + ba_graphs
    # Static features are topology-only (BC/clustering/PageRank ignore edge weights).
    # Precompute cache ONCE per base graph, reuse across all 200 episodes.
    print(f"[P1] pre-computing static caches for {len(all_graphs)} base graphs...", flush=True)
    base_caches = {}
    for G in all_graphs:
        key = id(G)
        base_caches[key] = (build_graph_feature_cache(G, compute_static_features(G)),
                            _ei(G), list(G.nodes()))
    print(f"[P1] loading {len(all_graphs)*n_ep} trajectories (edge weights vary per ep)...", flush=True)
    all_trajs = []  # list of (traj, G2, cache, ei, nodes)
    for G in all_graphs:
        cache0, ei0, nodes0 = base_caches[id(G)]
        for ep in range(n_ep):
            traj, G2 = gen_traj(G, ep, cfg)
            # Use base-graph static cache (topology identical; edge weights in G2 go to env)
            all_trajs.append((traj, G2, cache0, ei0, nodes0))
    print(f"[P1] {len(all_trajs)} trajectories ready (caches shared per topology)", flush=True)

    opt = torch.optim.Adam(policy.parameters(), lr=LR_P1, weight_decay=1e-5)
    p1_log = []

    for epoch in range(1, NIM+1):
        policy.train()
        # Subsample 300 trajectories
        subset = random.sample(all_trajs, min(SUBSAMPLE, len(all_trajs)))
        ep_losses = []
        for traj, G2, cache, ei2, nodes in subset:
            ei    = ei2.to(device)
            nodes = list(G2.nodes())
            n_g   = G2.number_of_nodes()
            env   = RevenueEnv(G2, RevenueEnvConfig(
                influence_model="monotone", b=1.0, weight_low=0., weight_high=2.,
                n_mc_samples=MC, reward_type="flat", gamma=1.0, seed=epoch))
            env.reset()
            policy.reset_episode(device)
            off = set()  # mutable set for speed
            step_losses = []
            for td in traj:
                nidx = td["node_idx"]; ed = float(td["discount"])
                # env.S is synced from prior steps; fast_gci uses env.S for influence feat
                feats = compute_node_features_fast(cache=cache, S=frozenset(env.S),
                                                   offered=frozenset(off),
                                                   t=len(off), k=n_g, env=env)
                x = torch.FloatTensor(feats).to(device)
                av = torch.tensor([v not in off for v in nodes], dtype=torch.bool, device=device)
                ms, h, ctx, _ = policy.forward(x, ei, av)
                ce = F.cross_entropy(ms.unsqueeze(0), torch.tensor([nidx], device=device))
                comb = torch.cat([h[nidx], ctx])
                bd = policy.get_discount_distribution(comb)
                mse = F.mse_loss(bd.mean, torch.tensor(ed, device=device))
                step_losses.append(ce + PW * mse)
                # update state: sync env.S so fast_gci is correct next step
                v = nodes[nidx]; off.add(v)
                if td.get("accepted", False): env.S.add(v)
                policy.update_sequence_state(ed, td.get("accepted",False),
                                             td.get("price",0.))
            if step_losses:
                loss = torch.stack(step_losses).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                ep_losses.append(loss.item())

        mean_loss = float(np.mean(ep_losses)) if ep_losses else float('nan')
        p1_log.append({"epoch":epoch,"loss":round(mean_loss,4)})
        if epoch % 20 == 0 or epoch == 1:
            print(f"[P1-{tag}] ep={epoch:4d}  loss={mean_loss:.4f}", flush=True)
            # Save checkpoint
            ckpt_p = os.path.join(CKPT_DIR, f"c1_ffba_{tag}_p1_ep{epoch}.pt")
            assert not os.path.exists(ckpt_p), f"STOP: {ckpt_p} already exists"
            torch.save({"epoch":epoch,"phase":1,"policy_state_dict":policy.state_dict()}, ckpt_p)
            sha = _sha8(ckpt_p)
            with open(README,"a") as f:
                f.write(f"c1_ffba_{tag}_p1_ep{epoch}.pt  sha={sha}  epoch={epoch}  loss={mean_loss:.4f}\n")
            print(f"[P1-{tag}] saved {ckpt_p} sha={sha}", flush=True)

    json.dump(p1_log, open(log_path,"w"))
    return policy

# ─── Phase 2 REINFORCE ────────────────────────────────────────────────────────
def phase2(policy, ff_graphs, ba_graphs, ff_frac, cfg, device, tag):
    opt     = torch.optim.Adam(policy.parameters(), lr=LR_P2, weight_decay=1e-5)
    welf    = Welford()

    for epoch in range(1, NRL+1):
        # Sample graph from mixture
        if random.random() < ff_frac:
            n = random.choice(FF_SIZES); G = generate_forest_fire(n, 0.37, 0.32, seed=epoch)
        else:
            nm, m = random.choice(ALL_BA_CONFIGS); G = generate_ba(nm, m, seed=epoch)
        # Fresh edge weights
        for u,v in G.edges(): G[u][v]['weight'] = random.uniform(0.,2.)
        cache = build_graph_feature_cache(G, compute_static_features(G))
        ei    = _ei(G).to(device)
        nodes = list(G.nodes()); n_g = G.number_of_nodes()
        env   = RevenueEnv(G, RevenueEnvConfig(
            influence_model="monotone", b=1.0, weight_low=0., weight_high=2.,
            n_mc_samples=MC, reward_type="flat", gamma=1.0, seed=epoch))
        env.reset()
        policy.train(); policy.reset_episode(device)
        log_probs = []; entropies = []; off = set(); rev = 0.

        for _ in range(n_g):
            if len(off) == n_g: break
            feats = compute_node_features_fast(cache=cache, S=env.S,
                                               offered=frozenset(off), t=env.t,
                                               k=0, env=env)
            x  = torch.FloatTensor(feats).to(device)
            av = torch.tensor([v not in off for v in nodes], dtype=torch.bool, device=device)
            ms, h, ctx, _ = policy.forward(x, ei, av)
            dist_s = torch.distributions.Categorical(logits=ms)
            idx    = dist_s.sample(); lp_s = dist_s.log_prob(idx)
            v      = nodes[int(idx)]
            comb   = torch.cat([h[idx], ctx])
            bd     = policy.get_discount_distribution(comb)
            disc   = float(bd.rsample().clamp(0.,1.))
            lp_d   = bd.log_prob(torch.tensor(disc, device=device))
            ent    = (dist_s.entropy() + bd.entropy()) * 0.5
            log_probs.append(lp_s + lp_d); entropies.append(ent)
            price  = env._estimate_valuation(v) * (1.-disc)
            off.add(v)
            if env._true_valuation(v) >= price: rev += price; env.S.add(v)
            env.t += 1
            policy.update_sequence_state(disc, env._true_valuation(v)>=price,
                                         price if env._true_valuation(v)>=price else 0.)

        adv = welf.advantage(rev)
        if log_probs:
            loss = torch.stack([-lp*adv - ENT*ent for lp,ent in zip(log_probs,entropies)]).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            opt.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"[P2-{tag}] ep={epoch:4d}  rev={rev:.1f}  adv={adv:.2f}  std={welf.std():.2f}",
                  flush=True)
            ckpt_p = os.path.join(CKPT_DIR, f"c1_ffba_{tag}_p2_ep{epoch}.pt")
            assert not os.path.exists(ckpt_p), f"STOP: {ckpt_p} already exists"
            torch.save({"epoch":epoch,"phase":2,"policy_state_dict":policy.state_dict()}, ckpt_p)
            sha = _sha8(ckpt_p)
            with open(README,"a") as f:
                f.write(f"c1_ffba_{tag}_p2_ep{epoch}.pt  sha={sha}  epoch={epoch}  rev={rev:.1f}\n")
            print(f"[P2-{tag}] saved {ckpt_p} sha={sha}", flush=True)

    return policy

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm_tag", choices=["50_50","2to1"])
    parser.add_argument("--ratio", type=float, default=None)
    args = parser.parse_args()

    tag      = args.arm_tag
    ff_frac  = args.ratio if args.ratio is not None else (0.5 if tag=="50_50" else 2/3)
    cfg      = _cfg_obj()
    device   = torch.device("cpu")   # CPU training (no MPS REINFORCE bug)

    # Verify base checkpoint
    assert os.path.exists(BASE_CKPT), f"Missing {BASE_CKPT}"
    sha = _sha8(BASE_CKPT)
    assert sha == SHA_BASE, f"Base SHA mismatch: {sha} vs {SHA_BASE}"
    print(f"[main-{tag}] base ckpt sha={sha} ✓  ff_frac={ff_frac:.3f}", flush=True)
    print(f"[main-{tag}] Welford std_floor=1.0 (sentinel verified, NOT 1e-8)", flush=True)
    print(f"[main-{tag}] Env: RevenueEnv unconstrained; no budget/bankruptcy", flush=True)
    print(f"[main-{tag}] Feature k=n_g (P1) / k=0 (P2 rollout, harness default)", flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Build policy (warm-start from base)
    policy = _make_pol()
    sd = torch.load(BASE_CKPT, map_location='cpu', weights_only=True)
    if 'policy_state_dict' in sd: sd = sd['policy_state_dict']
    policy.load_state_dict(sd, strict=False)   # strict=False: allow LSTM weight mismatch
    print(f"[main-{tag}] warm-start loaded (strict=False for LSTM dims)", flush=True)

    # Build training graphs
    ff_graphs    = [generate_forest_fire(n, 0.37, 0.32, seed=i) for i,n in enumerate(FF_SIZES)]
    ba_graphs_p1 = [generate_ba(n, m, seed=i) for i,(n,m) in enumerate(PHASE1_BA_CONFIGS)]
    # ba_graphs for P2 sampling: ALL_BA_CONFIGS used by phase2() via ALL_BA_CONFIGS global
    print(f"[main-{tag}] FF graphs: {[g.number_of_nodes() for g in ff_graphs]}", flush=True)
    print(f"[main-{tag}] P1 BA configs (n≤440): {PHASE1_BA_CONFIGS}", flush=True)
    print(f"[main-{tag}] P2 ALL BA configs: {ALL_BA_CONFIGS}", flush=True)
    print(f"[main-{tag}] total P2 BA={len(ALL_BA_CONFIGS)} (10 existing + 3 high-skew m=4)", flush=True)

    # Step 1: Generate + cache trajectories (200 eps per graph, P1 graphs only)
    print("[main] Building trajectory cache...", flush=True)
    stats_ff = build_cache(ff_graphs,    200, cfg, label="FF")
    stats_ba = build_cache(ba_graphs_p1, 200, cfg, label="BA")
    for s in stats_ff+stats_ba:
        print(f"  traj n={s['n']}: n_ep={s['n_ep']} mean_rev={s['mean_rev']}", flush=True)

    # Phase 1 (imitation, FF + BA n≤440 only)
    t0 = time.time()
    phase1(policy, ff_graphs, ba_graphs_p1, 200, cfg, device, tag,
           os.path.join(LOG_DIR, f"c1_ffba_{tag}_p1_log.json"))
    print(f"[main-{tag}] P1 done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Phase 2 (REINFORCE, samples from ALL_BA_CONFIGS including large n)
    t0 = time.time()
    phase2(policy, ff_graphs, ba_graphs_p1, ff_frac, cfg, device, tag)
    print(f"[main-{tag}] P2 done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Final checkpoint
    final_p = os.path.join(CKPT_DIR, f"c1_ffba_{tag}_final.pt")
    assert not os.path.exists(final_p), f"STOP: {final_p} already exists"
    torch.save({"policy_state_dict":policy.state_dict(),"tag":tag,"ff_frac":ff_frac}, final_p)
    sha = _sha8(final_p)
    with open(README,"a") as f:
        f.write(f"c1_ffba_{tag}_final.pt  sha={sha}  FINAL\n")
    print(f"[main-{tag}] final checkpoint {final_p} sha={sha}", flush=True)

if __name__ == "__main__":
    main()
