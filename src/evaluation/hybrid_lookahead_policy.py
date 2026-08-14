"""src/evaluation/hybrid_lookahead_policy.py — Hybrid Policy+Planner lookahead.

Wraps a trained SequentialJointPolicy. At each step:
  1. Run policy.forward() to get per-node scores and context.
  2. Build candidate set: top-5 available nodes × discount grid
     {d*, 0.0, 0.2, 0.5} (dedup; ≤20 pairs). Policy's own (node*, d*)
     always included.
  3. Score each candidate with Cal-DP v2 one-step lookahead:
       score = price + J[b_idx(b_next)][t_next]
     where J[b_idx][t] = expected future revenue from offer-step t with
     budget b_idx*delta. J is built by the same backward induction as
     _plan_dp_v2 but returned as a full table.
  4. Act with argmax score; ties → policy's own action.
  5. Update LSTM state with the executed action's outcome.

APPROXIMATION NOTE: J was built over the degree-sorted ordering sigma used
by Cal-DP v2. The hybrid indexes J by (offer_count, budget) — a remaining-
set mismatch relative to the DP's sigma ordering. This is an acceptable
approximation for an inference-time re-ranking heuristic; it does not
affect training or correctness of accounting.

Inference-time only — no training, no gradient computation anywhere here.

New file: does NOT modify dp_calibrated_v2.py, sequential_joint_policy.py,
or any other existing file.
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import torch

# ── J-table builder (thin replication of _plan_dp_v2 recurrence) ──────────────

def build_J_table(
    V: np.ndarray,
    A: np.ndarray,
    P: np.ndarray,
    class_boundaries: np.ndarray,
    n_total: int,
    B: float,
    c: float,
    tiers: Tuple[float, ...] = (1.0, 0.8, 0.5, 0.2, 0.0),
    delta: float = 0.05,
) -> np.ndarray:
    """Build the full DP value table J[b_idx, k_rel].

    Args:
        V:                 (n_classes, n_buckets) mean est_val table.
        A:                 (n_classes, n_buckets, 5) acceptance-rate table.
        P:                 (n_total, n_buckets) influence-bucket probs by position.
        class_boundaries:  (n_classes+1,) degree quantile boundaries.
        n_total:           Number of nodes / offer steps.
        B:                 Initial budget.
        c:                 Production cost per offer.
        tiers:             Discount tiers used by DP v2 (must match calibration).
        delta:             Budget discretisation step.

    Returns:
        J: np.ndarray of shape (b_steps+2, n_total+1).
           J[b_idx, k_rel] = expected future revenue from offer-step k_rel
           onwards with budget b_idx*delta remaining.
    """
    n_classes   = V.shape[0]
    n_buckets   = V.shape[1]
    tiers_list  = list(tiers)
    b_steps     = max(1, int(B / delta) + 1)

    # J is (b_steps+2) × (n_total+1); extra row/col for safe clipping
    J = np.zeros((b_steps + 2, n_total + 1), dtype=np.float64)

    # Degree-class helper (mirrors _deg_class in dp_calibrated_v2.py)
    def _deg_class(deg: int) -> int:
        for i in range(n_classes - 1, 0, -1):
            if deg >= class_boundaries[i]:
                return i
        return 0

    # We approximate class_of_pos using the P matrix's dominant bucket
    # (no degree info available here; use P[k] shape to infer cls via V lookup)
    # Simpler: replicate the same approach as _plan_dp_v2 — compute avg_val from V+P
    # without needing exact class_of_pos (use mean across classes weighted by P)

    for k_rel in range(n_total - 1, -1, -1):
        # Expected valuation at position k_rel (class-marginalised)
        # avg_val = E[V[cls] · P[k_rel]] averaged over classes
        # Since P[k_rel] is (n_buckets,) and V is (n_classes, n_buckets),
        # take mean over classes: avg_val = (V.mean(axis=0) @ P[k_rel])
        p_k   = P[k_rel] if k_rel < len(P) else P[-1]  # (n_buckets,)
        v_avg = V.mean(axis=0)                           # (n_buckets,) mean over classes
        avg_val = float(np.dot(v_avg, p_k))
        if avg_val <= 1e-9:
            avg_val = float(V.max())

        for b_idx in range(b_steps + 1):
            b_curr   = b_idx * delta
            best_rev = J[min(b_idx, b_steps), min(k_rel + 1, n_total)]

            # A approximation: use class-mean A
            A_mean  = A.mean(axis=0)  # (n_buckets, 5)
            ib_mode = int(np.argmax(p_k))

            for t_idx, t_disc in enumerate(tiers_list):
                price = avg_val * (1.0 - t_disc)
                if b_curr - c + price < -1e-9:
                    continue
                new_b_raw = b_curr - c + price
                new_b_idx = min(int(new_b_raw / delta), b_steps)
                p_acc     = float(A_mean[ib_mode, t_idx])
                ev_cand   = p_acc * price + J[new_b_idx, min(k_rel + 1, n_total)]
                if ev_cand > best_rev:
                    best_rev = ev_cand

            J[b_idx, k_rel] = best_rev

    return J


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dedup(vals: List[float], tol: float = 1e-6) -> List[float]:
    """Return deduplicated list preserving order."""
    seen, out = [], []
    for v in vals:
        if not any(abs(v - s) < tol for s in seen):
            seen.append(v)
            out.append(v)
    return out


def _top5_available(
    scores: torch.Tensor,
    available_mask: torch.Tensor,
    must_include: int,
) -> List[int]:
    """Return up to 5 available node indices by descending score.

    Always includes must_include if it is available.
    """
    avail_indices = available_mask.nonzero(as_tuple=True)[0].tolist()
    # Sort by score descending
    avail_scores  = [(i, float(scores[i].item())) for i in avail_indices]
    avail_scores.sort(key=lambda x: x[1], reverse=True)
    top5 = [i for i, _ in avail_scores[:5]]
    if must_include not in top5 and must_include in avail_indices:
        top5.append(must_include)
    return top5


# ── HybridLookaheadPolicy ─────────────────────────────────────────────────────

class HybridLookaheadPolicy:
    """Inference-time wrapper combining SequentialJointPolicy + DP v2 lookahead.

    The wrapped policy is used as-is (no training, eval mode assumed).

    Args:
        policy:     Trained SequentialJointPolicy (eval mode, on device).
        J:          Value table from build_J_table, shape (b_steps+2, n_total+1).
        grid_delta: Budget discretisation step (must match J's delta).
        b_max:      Max budget (B for the episode), used for clipping.
        cost_c:     Production cost per offer step.
    """

    def __init__(
        self,
        policy,
        J: np.ndarray,
        grid_delta: float,
        b_max: float,
        cost_c: float,
    ) -> None:
        self.pi    = policy
        self.J     = J
        self.delta = grid_delta
        self.bmax  = b_max
        self.c     = cost_c

    def _bidx(self, b: float) -> int:
        """Clamp budget to [0, b_max] and convert to grid index."""
        return int(min(max(b, 0.0), self.bmax) / self.delta)

    @torch.no_grad()
    def select_action(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        available_mask: torch.Tensor,
        env,
    ) -> Tuple[int, float]:
        """Select (node_idx, discount) using policy-proposes, planner-disposes.

        Args:
            x:              Node features (n, feat_dim).
            edge_index:     Edge index (2, |E|).
            available_mask: Boolean mask (n,), True = available.
            env:            BudgetRevenueEnv instance (for est_val and state).

        Returns:
            (node_idx, discount) — the chosen action.
        """
        # ── Step 1: get policy scores and its preferred action ────────────────
        masked_scores, h, context, _ = self.pi.forward(x, edge_index, available_mask)
        node_p = int(masked_scores.argmax().item())
        # Policy discount: greedy Beta mean
        combined_p = torch.cat([h[node_p], context], dim=0)
        d_p = float(self.pi.get_discount_distribution(combined_p).mean.item())

        # ── Step 2: build candidate set ───────────────────────────────────────
        cand_nodes  = _top5_available(masked_scores, available_mask, must_include=node_p)
        disc_opts   = _dedup([d_p, 0.0, 0.2, 0.5])

        # ── Step 3: score each candidate with lookahead ───────────────────────
        t_next       = len(env.offered) + 1
        j_t          = min(t_next, self.J.shape[1] - 1)
        current_b    = float(env.B)

        best_pair:   Optional[Tuple[int, float]] = None
        best_score:  float = -np.inf
        policy_score: Optional[float] = None  # for tie-breaking

        for v in cand_nodes:
            node_obj = env.nodes[v]
            ev = float(env._estimate_valuation(node_obj))
            for d in disc_opts:
                price     = ev * (1.0 - d)
                b_after   = current_b - self.c + price
                if b_after < -1e-9:
                    continue  # infeasible — drop (SKIP candidate)
                b_idx_nxt = self._bidx(b_after)
                score     = price + float(self.J[b_idx_nxt, j_t])
                is_policy = (v == node_p and abs(d - d_p) < 1e-6)
                # Tie-break: prefer policy's own action
                cmp_key   = (score, 1.0 if is_policy else 0.0)
                if cmp_key > (best_score, 1.0 if best_pair == (node_p, d_p) else 0.0):
                    best_pair  = (v, d)
                    best_score = score
                if is_policy:
                    policy_score = score

        # Fall back to policy action if no candidate survived feasibility
        if best_pair is None:
            best_pair = (node_p, d_p)

        return best_pair
