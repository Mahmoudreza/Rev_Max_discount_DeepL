"""src/models/encoders/episode_transformer.py — ALiBi Sliding-Window Transformer.

Fixes the three issues in the WSDM-ported EpisodeTransformer:
  1. Learned absolute positions cap episode length → replaced with ALiBi.
  2. O(T²) from-scratch recomputation every step → sliding window of 256 steps.
  3. Missing step-by-step interface (needed by TransformerJointPolicy) → added.

Architecture:
  Token:      [graph_emb ‖ last_discount ‖ last_accepted ‖ last_revenue], 67-dim
  Input proj: Linear(67, 64)
  Encoder:    2-layer Pre-LN TransformerEncoder, 4 heads, ffn=128
  Position:   ALiBi (Press et al. 2021) — bias = -slope_h × (i−j) for causal pairs.
              No learned embeddings → works at any episode length.
  Context:    Output at the last (most recent) token position.
  Window:     Attention capped at last `window` (default 256) tokens.
              Old tokens are detached (limit BPTT depth). New token keeps gradients.

The ALiBi + causal combined mask (shape H×T×T):
  mask[h, i, j] = -slope_h × (i−j)   if j ≤ i  (past/present)
                  −∞                   if j > i  (future, causal block)

Passed as `attn_mask` to TransformerEncoder with batch_size=1 → shape H×T×T.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Must match sequence_models.py TOKEN_DIM
_TOKEN_DIM: int = 67   # 64 (graph_dim) + 3 (discount, accepted, revenue)


def _alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute ALiBi attention slopes (Press et al. 2021, eq. 2).

    slope_h = 2^(−(8/H * h))  for h = 1..H when H is a power of 2.
    For non-power-of-2 H: use every other slope from the nearest 2× power.
    """
    def _pow2_slopes(n: int):
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3.0)))
        return [start * (start ** i) for i in range(n)]

    if (n_heads & (n_heads - 1)) == 0:   # power of 2
        slopes = _pow2_slopes(n_heads)
    else:
        n2 = 2 ** math.ceil(math.log2(n_heads))
        slopes = _pow2_slopes(n2)[0::2][:n_heads]
    return torch.tensor(slopes, dtype=torch.float32)  # (n_heads,)


def _make_alibi_causal_mask(
    T: int,
    slopes: torch.Tensor,   # (n_heads,)
    device: torch.device,
) -> torch.Tensor:
    """Build combined ALiBi + causal attention mask of shape (n_heads, T, T).

    mask[h, i, j] = -slope_h * (i-j)   if j ≤ i  (attend to past/present)
                    -1e9                 if j > i  (block future)
    """
    pos = torch.arange(T, device=device, dtype=torch.float32)  # (T,)
    # rel[i,j] = i - j → positive for past, negative for future
    rel = (pos.unsqueeze(1) - pos.unsqueeze(0))  # (T, T)
    # ALiBi bias (only for causal positions, 0 on diagonal/past)
    alibi = -slopes.view(-1, 1, 1) * rel.clamp(min=0).unsqueeze(0)  # (H, T, T)
    # Causal block: upper triangle → -1e9
    causal = torch.full((T, T), -1e9, device=device)
    causal = causal.triu(diagonal=1)  # (T, T): 0 on lower/diagonal, -1e9 on upper
    return alibi + causal.unsqueeze(0)  # (H, T, T)


class EpisodeTransformerSliding(nn.Module):
    """ALiBi Transformer sequence model with sliding-window causal attention.

    Drop-in replacement for EpisodeLSTM. Provides identical step() interface.

    Attributes:
        context_dim: Output dimension (= graph_dim). Used by policy heads.
        window:      Attention window size (default 256).

    Episode lifecycle:
        1. Call reset_episode(device) at start of each episode.
        2. At each step t, call step(graph_emb, last_d, last_acc, last_rev)
           to get context_t (used for scoring and pricing at step t).
        3. Call update_sequence_state(d_t, acc_t, rev_t) after env.step().
           (This method is a no-op here; state is maintained by step() itself.)
    """

    def __init__(
        self,
        graph_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        window: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if graph_dim % n_heads != 0:
            raise ValueError(f"graph_dim={graph_dim} must be divisible by n_heads={n_heads}")

        self.graph_dim   = graph_dim
        self.context_dim = graph_dim
        self.n_heads     = n_heads
        self.window      = window

        # Project 67-dim token → graph_dim
        self.token_proj = nn.Linear(_TOKEN_DIM, graph_dim)

        # Pre-LN TransformerEncoder (norm_first=True = Pre-LN, more stable)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=graph_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm    = nn.LayerNorm(graph_dim)

        # ALiBi slopes (no learned parameters → size-invariant by construction)
        slopes = _alibi_slopes(n_heads)
        self.register_buffer("alibi_slopes", slopes)  # (n_heads,)

        # Episode-level state (reset each episode, not nn.Parameters)
        self._token_buf: list = []   # list of detached (graph_dim,) tensors
        self._device: Optional[torch.device] = None

    # ── Episode management ─────────────────────────────────────────────────────

    def reset_episode(self, device: torch.device) -> None:
        """Clear token buffer and reset device. Call at start of each episode."""
        self._token_buf = []
        self._device    = device

    def update_sequence_state(
        self,
        discount:  float,
        accepted:  bool,
        revenue:   float,
    ) -> None:
        """Interface compatibility with EpisodeLSTM. No-op: state held in buffer."""
        pass   # buffer already updated inside step()

    # ── Core step function ─────────────────────────────────────────────────────

    def step(
        self,
        graph_embedding: torch.Tensor,          # (graph_dim,) — from GNN mean-pool
        last_discount:   float,
        last_accepted:   bool,
        last_revenue:    float,
        hidden:          None = None,           # unused, for interface compatibility
    ) -> Tuple[torch.Tensor, None]:
        """Compute context for the current step using causal attention over history.

        Builds the token for this step from [graph_emb ‖ last_action_info],
        appends to the sliding window buffer, runs ALiBi causal attention,
        and returns the output at the last (current) position.

        Old tokens in the buffer are detached from the computation graph to
        limit BPTT depth and memory. The current step's token retains gradients.

        Args:
            graph_embedding: Mean-pooled GNN output at current step, shape (graph_dim,).
            last_discount:   Discount offered at step t−1 (0.0 at t=0).
            last_accepted:   Whether step t−1 offer was accepted (False at t=0).
            last_revenue:    Revenue collected at step t−1 (0.0 at t=0).
            hidden:          Ignored. Present for interface compatibility.

        Returns:
            context: Context vector c_t, shape (graph_dim,).
            None:    Placeholder for hidden-state (EpisodeLSTM returns (h, c)).
        """
        device = graph_embedding.device

        # Build 67-dim raw token for this step
        extras = torch.tensor(
            [last_discount, float(last_accepted), last_revenue],
            dtype=torch.float32, device=device,
        )
        token_raw = torch.cat([graph_embedding, extras], dim=0)  # (67,)

        # Project to graph_dim (with gradient)
        token_emb = self.token_proj(token_raw)  # (graph_dim,)

        # ── 1-step shortcut (no history yet) ──────────────────────────────────
        if len(self._token_buf) == 0:
            self._token_buf.append(token_emb.detach())
            return self.out_norm(token_emb), None

        # ── Assemble window: detached old tokens + live new token ─────────────
        # Detach all old tokens: limits BPTT to current step.
        # Gradient flows through: transformer weights (all steps), current token_proj/GNN.
        old = self._token_buf[-(self.window - 1):]   # at most window-1 old tokens, detached
        old_stacked  = torch.stack(old, dim=0).detach()       # (T-1, D)
        new_token    = token_emb.unsqueeze(0)                  # (1, D) with grad
        x = torch.cat([old_stacked, new_token], dim=0)        # (T, D)

        T = x.shape[0]
        x = x.unsqueeze(0)   # (1, T, D) — batch_first=True

        # ── ALiBi + causal mask: (n_heads, T, T) ──────────────────────────────
        mask = _make_alibi_causal_mask(T, self.alibi_slopes, device)
        # PyTorch expects (B*H, T, T) for batch=1 → same as (H, T, T)

        # ── Transformer forward ────────────────────────────────────────────────
        out = self.transformer(x, mask=mask)   # (1, T, D)
        context = self.out_norm(out[0, -1, :]) # last position → (D,)

        # Append current token (detached) to buffer for next step
        self._token_buf.append(token_emb.detach())

        return context, None

    # ── Convenience constructor ────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg) -> "EpisodeTransformerSliding":
        """Build from OmegaConf config node cfg.transformer."""
        return cls(
            graph_dim=int(getattr(cfg, "d_model", 64)),
            n_heads=int(getattr(cfg, "n_heads", 4)),
            n_layers=int(getattr(cfg, "n_layers", 2)),
            ff_dim=int(getattr(cfg, "ffn_dim", 128)),
            window=int(getattr(cfg, "attention_window", 256)),
            dropout=float(getattr(cfg, "dropout", 0.0)),
        )
