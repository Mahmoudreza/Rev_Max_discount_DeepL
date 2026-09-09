"""
train_skip_policy.py — Phase 1 (imitation) + Phase 2 (profit RL) for SkipPolicy.

Skip semantics: "waiting" — skipped buyer stays available; step advances.
Objective    : Phase 2 reward = profit = R - c|S_T| = B_T - B_0.
Expert (P1)  : arm_b (0b549f93) for selection/discount; skip when v_hat < c.
Skip cap     : 50 consecutive skips → terminate episode.

Usage (3 seeds in parallel, one GPU each):
  venv/bin/python3 -u experiments/train_skip_policy.py --seed 0 --gpu 0 &
  venv/bin/python3 -u experiments/train_skip_policy.py --seed 1 --gpu 1 &
  venv/bin/python3 -u experiments/train_skip_policy.py --seed 2 --gpu 2 &

Checkpoints (every epoch):
  results/checkpoints/skip_s{SEED}_p1_ep{EP:04d}.pt
  results/checkpoints/skip_s{SEED}_p2_ep{EP:04d}.pt
README.md   : appended every 20 epochs and for the best.
"""
import sys, os, argparse, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire, generate_modular_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy_skip import SequentialJointPolicySkip, SKIP_CAP
from src.utils.helpers import set_seed
from _arm_b_utils import make_ei, N_MC, W_HIGH, C, _feat_unconstrained, _avail_mask, load_arm_b

assert W_HIGH == 2.0, f"Expected W_HIGH=2.0 but got {W_HIGH}"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
P1_EPOCHS   = 200
P2_EPOCHS   = 150
P1_LR       = 3e-4
P2_LR       = 1e-4
P2_ENTROPY  = 0.001  # entropy bonus (small — arm_b init already has good diversity)
P2_BASELINE = 0.1    # EMA momentum for profit baseline
GRAD_CLIP   = 1.0
K_VALUES    = [5, 10, 15, 20, 30, 40]
TRAIN_SEEDS = list(range(5))   # inner rollout seeds per (graph, k)
IN_DIM      = 21               # same feature set as arm_b (includes budget dummy)
N_MC_TRAIN  = 5                # MC samples during training (eval uses N_MC=200); ~40× speedup
MAX_EP_STEPS = 200             # max steps per training episode (BPTT truncation + consistent timing)
_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR    = os.path.join(_ROOT, "results", "checkpoints")
README_PATH = os.path.join(CKPT_DIR, "README.md")


def make_policy(device):
    enc  = GraphSAGEEncoder(in_dim=IN_DIM, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    return SequentialJointPolicySkip(enc, lstm, gnn_dim=64, context_dim=64).to(device)


def make_train_graphs():
    """5 training graphs (different sizes/seeds). Polblogs included via load_polblogs."""
    from src.env.polblogs_loader import load_polblogs
    return [
        generate_forest_fire(500,  0.37, 0.32, seed=200),
        generate_forest_fire(800,  0.37, 0.32, seed=201),
        generate_forest_fire(1000, 0.37, 0.32, seed=202),
        generate_modular_forest_fire([250, 250], 0.37, 0.32, 0.05, seed=200),
        load_polblogs(),
    ]


def save_ckpt(pol, phase, epoch, seed, extra=None):
    os.makedirs(CKPT_DIR, exist_ok=True)
    p = os.path.join(CKPT_DIR, f"skip_s{seed}_p{phase}_ep{epoch:04d}.pt")
    d = {"policy_state_dict": pol.state_dict(), "epoch": epoch, "phase": phase, "seed": seed}
    if extra: d.update(extra)
    torch.save(d, p)
    return p


def append_readme(msg):
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(README_PATH, "a") as f:
        f.write(msg + "\n")


# ── Phase 1 helpers ───────────────────────────────────────────────────────────

@torch.no_grad()
def _arm_b_action(arm_b, x, ei, av, n, env, device):
    """Return arm_b's chosen (ni, d, v_hat) without touching pol_skip's state."""
    sc, h, ctx, _ = arm_b.forward(x[:, :IN_DIM], ei, av)
    ni = int(sc.argmax().item())
    node = env.nodes[ni]
    v_hat = float(env._estimate_valuation(node))
    d = float(arm_b.get_discount_distribution(torch.cat([h[ni], ctx])).mean.item())
    return ni, d, v_hat


def p1_episode(pol_skip, arm_b, graph, ei, cache, B, seed, device):
    """One Phase-1 imitation episode. Returns scalar loss."""
    n = graph.number_of_nodes()
    set_seed(seed)
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed,
                          weight_high=W_HIGH, n_mc_samples=N_MC_TRAIN)
    env = BudgetRevenueEnv(graph, cfg); env.reset()
    pol_skip.reset_episode(device)
    arm_b.reset_episode(device)  # type: ignore

    ce_losses, disc_losses = [], []
    consec_skips = 0
    step_count = 0

    while env.available_nodes and not env._check_bankrupt() and step_count < MAX_EP_STEPS:
        x   = torch.FloatTensor(_feat_unconstrained(cache, env, n)).to(device)
        av  = _avail_mask(env, n, device)
        if not av.any(): break

        # --- Expert target (no grad) ---
        ni_exp, d_exp, v_hat_exp = _arm_b_action(arm_b, x, ei, av, n, env, device)

        # Skip if expert's top buyer is below cost
        target = n if v_hat_exp < C else ni_exp  # n = skip index

        # --- Skip policy forward ---
        sc_full, h, ctx, _ = pol_skip.forward(x, ei, av)   # (n+1,)
        log_p = F.log_softmax(sc_full, dim=0)
        ce_losses.append(-log_p[target])

        # Discount supervision only when expert picks a real buyer
        if target < n:
            d_pred = pol_skip.get_discount_distribution(
                torch.cat([h[target], ctx])).mean
            disc_losses.append(F.mse_loss(d_pred, torch.tensor(d_exp, device=device)))

        # --- Act in environment ---
        if target == n:
            pol_skip.update_sequence_state(0.0, False, 0.0)
            arm_b.update_sequence_state(0.0, False, 0.0)
            consec_skips += 1
            if consec_skips >= SKIP_CAP: break
        else:
            consec_skips = 0
            node = env.nodes[ni_exp]
            _, r, done, info = env.step(ni_exp, d_exp)
            acc = bool(info.get("accepted", r > 0))
            pol_skip.update_sequence_state(d_exp, acc, info.get("revenue_step", 0.0))
            arm_b.update_sequence_state(d_exp, acc, info.get("revenue_step", 0.0))
            if done: break
        step_count += 1

    loss = torch.stack(ce_losses).mean()
    if disc_losses:
        loss = loss + 0.5 * torch.stack(disc_losses).mean()
    return loss


# ── Phase 2 helpers ───────────────────────────────────────────────────────────

def p2_episode(pol_skip, graph, ei, cache, B, seed, device):
    """One Phase-2 REINFORCE episode.
    Returns (log_probs, entropies, profit, skip_cap_hit, per_step_G).
    per_step_G[t] = Σ_{τ≥t} r_τ  where r_τ = price_τ-C (if accepted), else 0.
    """
    n = graph.number_of_nodes()
    set_seed(seed)
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed,
                          weight_high=W_HIGH, n_mc_samples=N_MC_TRAIN)
    env = BudgetRevenueEnv(graph, cfg); env.reset()
    pol_skip.reset_episode(device)

    log_probs, entropies, step_rewards = [], [], []
    consec_skips = 0; skip_cap_hit = False
    step_count = 0

    while env.available_nodes and not env._check_bankrupt() and step_count < MAX_EP_STEPS:
        x   = torch.FloatTensor(_feat_unconstrained(cache, env, n)).to(device)
        av  = _avail_mask(env, n, device)
        if not av.any(): break

        sc_full, h, ctx, _ = pol_skip.forward(x, ei, av)  # (n+1,)
        dist = Categorical(logits=sc_full)
        action = dist.sample().item()
        log_probs.append(dist.log_prob(torch.tensor(action, device=device)))
        entropies.append(dist.entropy())

        step_reward = 0.0
        if action == n:  # skip
            pol_skip.update_sequence_state(0.0, False, 0.0)
            consec_skips += 1
            if consec_skips >= SKIP_CAP:
                skip_cap_hit = True; break
        else:
            consec_skips = 0
            node = env.nodes[action]
            d = float(pol_skip.get_discount_distribution(
                torch.cat([h[action], ctx])).mean.item())
            v_hat = float(env._estimate_valuation(node))
            price = v_hat * (1.0 - d)
            _, r, done, info = env.step(action, d)
            acc = bool(info.get("accepted", r > 0))
            pol_skip.update_sequence_state(d, acc, info.get("revenue_step", 0.0))
            if acc: step_reward = price - C   # marginal profit (can be negative)
            if done: break
        step_rewards.append(step_reward)
        step_count += 1

    profit = float(env.B) - B   # = R - c|S_T|
    lp = torch.stack(log_probs) if log_probs else torch.zeros(1, device=device)
    en = torch.stack(entropies) if entropies else torch.zeros(1, device=device)
    # Per-step returns (suffix sums)
    G = []; cum = 0.0
    for rw in reversed(step_rewards):
        cum += rw; G.append(cum)
    G.reverse()
    return lp, en, profit, skip_cap_hit, G if len(G) == len(step_rewards) else None


# ── Main training loop ────────────────────────────────────────────────────────

def train(args, pol_skip_init=None):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"seed={args.seed}  device={device}  P1={P1_EPOCHS}ep  P2={P2_EPOCHS}ep")

    torch.manual_seed(args.seed * 1000)
    np.random.seed(args.seed * 1000)

    pol_skip = pol_skip_init if pol_skip_init is not None else make_policy(device)
    arm_b    = load_arm_b(device)
    arm_b.eval()
    for p in arm_b.parameters(): p.requires_grad_(False)

    print("Building training graphs...", flush=True)
    graphs = make_train_graphs()
    eis    = []
    caches = []
    for g in graphs:
        ei, cache = make_ei(g, device)
        eis.append(ei); caches.append(cache)
    print(f"  {len(graphs)} graphs: n={[g.number_of_nodes() for g in graphs]}", flush=True)

    # ── arm_b init for Phase 2 ──────────────────────────────────────────────
    if args.arm_b_init:
        arm_b_sd = {k: v for k, v in arm_b.state_dict().items()}
        missing, unexpected = pol_skip.load_state_dict(arm_b_sd, strict=False)
        print(f"arm_b_init: loaded {len(arm_b_sd)-len(missing)} keys "
              f"(missing={len(missing)} = skip_head, unexpected={len(unexpected)})", flush=True)
        # Freeze everything except skip_head so arm_b behaviour is preserved
        for name, p in pol_skip.named_parameters():
            p.requires_grad_("skip_head" in name)
        n_trainable = sum(p.numel() for p in pol_skip.parameters() if p.requires_grad)
        print(f"  trainable params: {n_trainable} (skip_head only)", flush=True)
        args.skip_p1 = True   # no imitation phase

    best_p1_loss = float("inf")
    best_p2_profit = -float("inf")

    # ── PHASE 1 ──────────────────────────────────────────────────────────────
    if not args.skip_p1:
        opt1 = torch.optim.Adam(pol_skip.parameters(), lr=P1_LR)
        t0_p1 = time.time()
        for ep in range(1, P1_EPOCHS + 1):
            pol_skip.train()
            ep_losses = []
            for gi, (g, ei, cache) in enumerate(zip(graphs, eis, caches)):
                for k in K_VALUES:
                    B = k * C
                    for ts in TRAIN_SEEDS:
                        loss = p1_episode(pol_skip, arm_b, g, ei, cache, B, ts + ep * 100, device)
                        opt1.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(pol_skip.parameters(), GRAD_CLIP)
                        opt1.step()
                        ep_losses.append(loss.item())
            mean_loss = np.mean(ep_losses)
            save_ckpt(pol_skip, 1, ep, args.seed, {"loss": mean_loss})
            is_best = mean_loss < best_p1_loss
            if is_best: best_p1_loss = mean_loss
            if ep % 20 == 0 or is_best:
                elapsed = time.time() - t0_p1
                msg = (f"[skip s{args.seed} P1 ep{ep:3d}/{P1_EPOCHS}] "
                       f"loss={mean_loss:.4f}  elapsed={elapsed:.0f}s"
                       + (" BEST" if is_best else ""))
                print(msg, flush=True)
                if ep % 20 == 0 or is_best:
                    append_readme(msg)
        print(f"Phase 1 done. best_loss={best_p1_loss:.4f}", flush=True)

    # ── PHASE 2 ──────────────────────────────────────────────────────────────
    opt2 = torch.optim.Adam(filter(lambda p: p.requires_grad, pol_skip.parameters()), lr=P2_LR)
    # Initial baseline: arm_b earns ≈ 0 to -B; start at -0.5*B so profit=0 (skip all) is above it
    baseline = {(gi, k): -0.5 * k * C
                for gi in range(len(graphs)) for k in K_VALUES}
    ckpt_prefix = "skip_arminit" if args.arm_b_init else "skip"
    best_p2_profit = -float("inf")
    t0_p2 = time.time()

    for ep in range(1, P2_EPOCHS + 1):
        pol_skip.train()
        ep_profits, ep_caps = [], 0
        ep_losses = []

        for gi, (g, ei, cache) in enumerate(zip(graphs, eis, caches)):
            for k in K_VALUES:
                B = k * C
                bkey = (gi, k)
                b_val = baseline[bkey]

                for ts in TRAIN_SEEDS:
                    lp, en, profit, cap_hit, per_step_G = p2_episode(
                        pol_skip, g, ei, cache, B, ts + ep * 200, device)
                    ep_profits.append(profit)
                    if cap_hit: ep_caps += 1

                    # Per-step returns (lower variance than episode reward)
                    if per_step_G is not None and len(per_step_G) == len(lp):
                        G = torch.tensor(per_step_G, dtype=torch.float32, device=device)
                        adv = G - b_val
                        loss = -(lp * adv).sum() - P2_ENTROPY * en.sum()
                    else:
                        advantage = profit - b_val
                        loss = -(lp.sum() * advantage) - P2_ENTROPY * en.sum()
                    opt2.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(pol_skip.parameters(), GRAD_CLIP)
                    opt2.step()
                    ep_losses.append(loss.item())

                    baseline[bkey] = (1 - P2_BASELINE) * b_val + P2_BASELINE * profit

        mean_profit = np.mean(ep_profits)
        is_best = mean_profit > best_p2_profit
        if is_best: best_p2_profit = mean_profit
        save_ckpt(pol_skip, 2, ep, args.seed, {"profit": mean_profit, "skip_caps": ep_caps})
        if ep % 20 == 0 or is_best:
            elapsed = time.time() - t0_p2
            msg = (f"[skip s{args.seed} P2 ep{ep:3d}/{P2_EPOCHS}] "
                   f"profit={mean_profit:.3f}  caps={ep_caps}  elapsed={elapsed:.0f}s"
                   + (" BEST" if is_best else ""))
            print(msg, flush=True)
            append_readme(msg)

    print(f"Phase 2 done. best_profit={best_p2_profit:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",      type=int,  default=0)
    ap.add_argument("--gpu",       type=int,  default=0)
    ap.add_argument("--skip_p1",   action="store_true",
                    help="Skip Phase 1; start Phase 2 from scratch or from --resume_p1")
    ap.add_argument("--resume_p1", type=str,  default=None,
                    help="Path to a P1 checkpoint; loads weights then runs Phase 2 only")
    ap.add_argument("--arm_b_init", action="store_true",
                    help="Init Phase 2 from arm_b weights (skip_head trainable only); skips Phase 1")
    args = ap.parse_args()

    pol_init = None
    if args.resume_p1:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        pol_init = make_policy(device)
        sd = torch.load(args.resume_p1, map_location="cpu")
        pol_init.load_state_dict(sd["policy_state_dict"])
        print(f"Loaded P1 ckpt: {args.resume_p1}", flush=True)
        args.skip_p1 = True

    train(args, pol_skip_init=pol_init)


if __name__ == "__main__":
    main()
