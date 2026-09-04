"""
diag_armc_trace.py — confirm which checkpoint is the profit-objective Arm C.
Run: venv/bin/python3 experiments/diag_armc_trace.py --ckpt PATH [--net Modular_FF] [--k 15]
"""
import sys, os, hashlib, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire, generate_modular_forest_fire, load_rice_facebook
from src.env.polblogs_loader import load_polblogs
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.helpers import set_seed
from _arm_b_utils import make_ei, N_MC, W_HIGH, C, _avail_mask

def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def _feat(cache, env, n, in_dim):
    from src.utils.features import compute_node_features_fast
    x = compute_node_features_fast(cache, env.S, env.offered, env.t, k=50, env=env)
    if in_dim == 21:
        x = np.concatenate([x, np.ones((n,1), np.float32)], axis=1)
    return x

def try_load(ckpt, in_dim, device):
    enc  = GraphSAGEEncoder(in_dim=in_dim, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd = torch.load(ckpt, map_location="cpu")
    if "policy_state_dict" in sd: sd = sd["policy_state_dict"]
    elif "model_state_dict" in sd: sd = sd["model_state_dict"]
    pol.load_state_dict(sd, strict=True)
    return pol.eval().to(device)

def run_trace(pol, graph, in_dim, B, seed, device, n_price=10):
    from _arm_b_utils import make_ei
    n = graph.number_of_nodes()
    ei, cache = make_ei(graph, device)
    set_seed(seed)
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed,
                          weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(graph, cfg); env.reset()
    pol.reset_episode(device)
    offers=0; accepted=0; prices=[]
    with torch.no_grad():
        while env.available_nodes and not env._check_bankrupt():
            x = torch.FloatTensor(_feat(cache, env, n, in_dim)).to(device)
            av = _avail_mask(env, n, device)
            if not av.any(): break
            sc, h, ctx, _ = pol.forward(x, ei, av)
            ni = int(sc.argmax().item())
            node = env.nodes[ni]
            d = float(pol.get_discount_distribution(torch.cat([h[ni],ctx])).mean.item())
            v_hat = float(env._estimate_valuation(node))
            price = v_hat * (1.0 - d)
            _, r, done, info = env.step(ni, d)
            acc = bool(info.get("accepted", r > 0))
            offers += 1
            if acc:
                accepted += 1
                if len(prices) < n_price: prices.append((price, d, v_hat))
            pol.update_sequence_state(d, acc, info.get("revenue_step", 0.0))
            if done: break
    rev = float(env.total_revenue); ns = len(env.S)
    profit = env.B - B
    return offers, accepted, prices, rev, ns, profit, env.B

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--net", default="Modular_FF")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cpu")
    sha = _sha8(args.ckpt)
    print(f"Checkpoint: {os.path.basename(args.ckpt)}  sha={sha}")
    # Try in_dim=20 first, then 21
    for in_dim in [20, 21]:
        try:
            pol = try_load(args.ckpt, in_dim, device)
            print(f"  Loaded: in_dim={in_dim}")
            break
        except RuntimeError as e:
            print(f"  in_dim={in_dim} FAIL: {str(e)[:80]}")
            pol = None
    if pol is None: return
    # Load graph
    if args.net == "Modular_FF": graph = generate_modular_forest_fire([250,250],0.37,0.32,0.05,seed=0)
    elif args.net == "FF_1000":  graph = generate_forest_fire(1000,0.37,0.32,seed=0)
    elif args.net == "polblogs": graph = load_polblogs()
    else: graph = generate_forest_fire(1000,0.37,0.32,seed=0)
    B = args.k * C
    off, acc, prices, rev, ns, prof, fb = run_trace(pol, graph, in_dim, B, args.seed, device)
    print(f"\nModular_FF  k={args.k}  seed={args.seed}  B={B:.1f}")
    print(f"  offers={off}  accepted={acc}  rev={rev:.3f}  |S|={ns}  profit={prof:.3f}  B_T={fb:.3f}")
    print(f"  First {len(prices)} accepted prices (price, discount, v_hat):")
    for i,(p,d,v) in enumerate(prices): print(f"    {i}: price={p:.4f}  disc={d:.4f}  v_hat={v:.4f}")

if __name__ == "__main__":
    main()
