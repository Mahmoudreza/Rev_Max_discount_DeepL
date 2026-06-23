"""
src/models/encoders/sequence_models.py

Episode-level sequence models: LSTM and Transformer.
These wrap the GNN encoder to add temporal memory across steps.

Motivation:
  The GNN at step t captures WHO is connected to whom (spatial structure).
  But it has no memory of the trajectory: whether past buyers rejected,
  what discounts were offered, how prices were received.
  LSTM/Transformer over the step sequence captures this temporal structure.

Architecture:
  Step t:
    1. GNN encoder → node embeddings H_t ∈ R^(n×64)
    2. Global mean pool → graph state g_t ∈ R^64
    3. Sequence model processes [g_0, ..., g_t] → context vector c_t
    4. For each node v:
         score_v    = scoring_head([H_t[v] ‖ c_t])
         discount_v = pricing_head([H_t[v*] ‖ c_t])

The "token" fed to the sequence model at each step t is:
  token_t = [g_t ‖ last_discount ‖ last_accepted ‖ last_revenue]
             (64   +    1         +      1         +     1       ) = 67-dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


TOKEN_DIM = 67   # 64 (graph state) + 1 (discount) + 1 (accepted) + 1 (revenue)


class EpisodeLSTM(nn.Module):
    """LSTM over the sequence of graph states across episode steps.

    At each step t, the LSTM receives a token encoding the current graph state
    and the outcome of the last action. Its hidden state c_t is concatenated
    with each node's GNN embedding before the scoring and pricing heads.

    Args:
        graph_dim: GNN output dimension (default 64, matching GraphSAGE).
        lstm_hidden: LSTM hidden state dimension (default 64).
        n_layers: Number of LSTM layers (default 1).
        dropout: LSTM dropout (default 0.0).
    """

    def __init__(
        self,
        graph_dim: int = 64,
        lstm_hidden: int = 64,
        n_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.graph_dim = graph_dim
        self.lstm_hidden = lstm_hidden

        # Project token to LSTM input dim
        self.token_proj = nn.Linear(TOKEN_DIM, graph_dim)

        self.lstm = nn.LSTM(
            input_size=graph_dim,
            hidden_size=lstm_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Hidden state dimensionality for downstream heads
        self.context_dim = lstm_hidden

    def init_hidden(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state to zeros at episode start.

        Args:
            device: Target device.

        Returns:
            (h_0, c_0) zero tensors, each shape (n_layers, 1, lstm_hidden).
        """
        h = torch.zeros(self.lstm.num_layers, 1, self.lstm_hidden, device=device)
        c = torch.zeros(self.lstm.num_layers, 1, self.lstm_hidden, device=device)
        return h, c

    def step(
        self,
        graph_embedding: torch.Tensor,
        last_discount: float,
        last_accepted: bool,
        last_revenue: float,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process one episode step and update hidden state.

        Args:
            graph_embedding: Mean-pooled GNN output at current step, shape (64,).
            last_discount: Discount offered at t-1 (0.0 at t=0).
            last_accepted: Whether t-1 offer was accepted (False at t=0).
            last_revenue: Revenue collected at t-1 (0.0 at t=0).
            hidden: (h, c) LSTM hidden state tuple.

        Returns:
            context: Context vector c_t, shape (lstm_hidden,).
            new_hidden: Updated (h, c) tuple.
        """
        device = graph_embedding.device

        # Build token: [graph_state ‖ last_discount ‖ last_accepted ‖ last_revenue]
        extras = torch.tensor(
            [last_discount, float(last_accepted), last_revenue],
            dtype=torch.float32,
            device=device,
        )
        token = torch.cat([graph_embedding, extras], dim=0)   # (67,)
        token = self.token_proj(token).unsqueeze(0).unsqueeze(0)   # (1, 1, 64)

        output, new_hidden = self.lstm(token, hidden)
        context = output.squeeze(0).squeeze(0)   # (lstm_hidden,)
        return context, new_hidden


class EpisodeTransformer(nn.Module):
    """Transformer over the full step history of an episode.

    At each step t, applies multi-head self-attention over all past tokens
    [token_0, ..., token_{t-1}] to produce a context vector c_t.
    Unlike the LSTM, this can attend to any past step directly.

    Args:
        graph_dim: GNN output dimension (default 64).
        n_heads: Number of attention heads (default 4).
        n_layers: Number of Transformer encoder layers (default 2).
        ff_dim: Feed-forward hidden dimension (default 128).
        max_seq_len: Maximum episode length (default 2000 nodes).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        graph_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        max_seq_len: int = 2000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.graph_dim = graph_dim
        self.context_dim = graph_dim

        # Project token to model dim
        self.token_proj = nn.Linear(TOKEN_DIM, graph_dim)

        # Learnable positional encoding (one vector per step)
        self.pos_embedding = nn.Embedding(max_seq_len, graph_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=graph_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Learned [CLS] token — acts as the context summary
        self.cls_token = nn.Parameter(torch.randn(1, 1, graph_dim))

    def forward(
        self,
        token_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Attend over the full step history and return context.

        Args:
            token_sequence: Stacked tokens [token_0, ..., token_{t-1}],
                            shape (t, TOKEN_DIM).

        Returns:
            context: Context vector c_t, shape (graph_dim,).
        """
        t = token_sequence.shape[0]
        device = token_sequence.device

        # Project tokens to model dim: (t, graph_dim)
        x = self.token_proj(token_sequence)

        # Add positional encodings
        positions = torch.arange(t, device=device)
        x = x + self.pos_embedding(positions)   # (t, graph_dim)

        # Prepend [CLS] token
        cls = self.cls_token.expand(1, 1, self.graph_dim)   # (1, 1, graph_dim)
        x = torch.cat([cls.squeeze(0), x], dim=0)           # (t+1, graph_dim)
        x = x.unsqueeze(0)                                   # (1, t+1, graph_dim)

        # Self-attention over full history
        out = self.transformer(x)   # (1, t+1, graph_dim)

        # Use [CLS] output as context
        context = out[0, 0, :]   # (graph_dim,)
        return context

    def build_token(
        self,
        graph_embedding: torch.Tensor,
        last_discount: float,
        last_accepted: bool,
        last_revenue: float,
    ) -> torch.Tensor:
        """Build a single token for the current step.

        Args:
            graph_embedding: Mean-pooled GNN output, shape (64,).
            last_discount: Discount at previous step (0.0 at t=0).
            last_accepted: Whether previous offer was accepted.
            last_revenue: Revenue from previous step.

        Returns:
            token: shape (TOKEN_DIM,) = (67,).
        """
        device = graph_embedding.device
        extras = torch.tensor(
            [last_discount, float(last_accepted), last_revenue],
            dtype=torch.float32,
            device=device,
        )
        return torch.cat([graph_embedding, extras], dim=0)
