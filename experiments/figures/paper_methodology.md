# RevMax-AAAI2027: Methodology Section Draft

## 3. Revenue Maximization in Social Networks

### 3.1 Problem Formulation

We study the seller's revenue maximization problem in a social network $G = (V, E)$. 
The seller visits buyers sequentially; each buyer $i \in V$ receives a personalized offer 
$(p_i, d_i)$ where $p_i$ is the base price and $d_i \in [0, 1]$ is a discount.

**Buyer Valuation** (Babaei et al. 2013): Buyer $i$'s willingness-to-pay depends on how 
many of their social contacts have already purchased the item:

$$v_i(S) = f\!\left(\frac{\sum_{j \in S \cap \mathcal{N}(i)} w_{ij}}{\sum_{k \in \mathcal{N}(i)} w_{ik}}\right)$$

where $w_{ij} \sim \text{Uniform}(0, 2)$ are link weights and $f$ is the Rayleigh-based 
valuation function $f(x) = \frac{2x}{b^2} e^{-x^2/b^2}$ (non-monotone) or its clipped 
monotone variant.

**MDP Formulation**: We cast the problem as a finite-horizon MDP $(S, A, R, T)$:
- **State** $s_t$: 20-dimensional GNN node embeddings encoding graph structure, 
  current seed set $S$, offered set, and time step.
- **Action** $a_t = (i_t, d_t)$: joint discrete (buyer selection) + continuous (discount).
- **Reward** $r_t = p_{i_t} \cdot (1 - d_{i_t}) \cdot \mathbf{1}[\text{accept}]$ (flat)
  or $\gamma^t \cdot r_t$ (NPV, Idea 2).

### 3.2 Joint Policy Architecture (Rev-GNN)

Our architecture combines a GNN encoder with two output heads:

$$h = \text{GraphSAGE}(G, x) \in \mathbb{R}^{n \times 64}$$

$$\text{score}(i) = \text{MLP}_{\theta_1}(h_i) \quad \Rightarrow \quad i^* = \arg\max_{i \notin \text{offered}} \text{score}(i)$$

$$d^* = \sigma\!\left(\text{MLP}_{\theta_2}(h_{i^*})\right) \in [0, 1]$$

**Training** follows a two-phase curriculum:
1. **Phase 1 (Imitation)**: MSE loss against Greedy-Discount expert trajectories.
2. **Phase 2 (REINFORCE)**: Policy gradient with running-mean baseline.

### 3.3 Sequence-Aware Extensions (Phase 5B)

For long selling sequences, we extend the policy with:
- **Rev-GNN-LSTM**: LSTM context $z_t = \text{LSTM}(z_{t-1}, \text{mean}(h^{(t)}))$
- **Rev-GNN-Transformer**: Transformer over the history of global states $\{s_0, \ldots, s_t\}$

### 3.4 Node Feature Encoding (20 Dimensions)

| Dims | Feature | Description |
|------|---------|-------------|
| 0–3  | Structural | Degree, clustering, betweenness, closeness |
| 4–7  | Ego-network | 2-hop degree, edge density, neighbor degree stats |
| 8–11 | Spectral | Fiedler value, approx. pagerank, katz centrality |
| 12–15| In-S context | Fraction of neighbors in S, max/mean w to S |
| 16   | Current influence | $x_i(S_t)$ normalized influence |
| 17   | Current valuation | $v_i(S_t)$ = $f(x_i)$ |
| 18   | Was-offered flag | 1 if $i \in \text{offered}$ |
| 19   | Time remaining | $(n - t) / n$ |

### 3.5 Baselines (Babaei et al. 2013)

| Strategy | Description |
|----------|-------------|
| IE-Strategy | Greedy hill-climbing seeds → free give-away → myopic pricing |
| µ-Discount | Discount determined by average degree µ |
| Greedy-Discount | 6 influence regions; greedy discount selection per step |
| σ-Discount | µ ± σ degree thresholds for super-influencers |

**Expected ordering**: Greedy-Discount > σ-Discount > µ-Discount > IE-Strategy,
with Greedy-Discount achieving ~21% higher revenue than IE-Strategy on BA graphs.

### 3.6 Idea 2: Net Present Value (NPV) Objective

To model time preference of money, the NPV reward modifies the return:
$$G_t^{\text{NPV}} = \sum_{k \geq 0} \gamma^{t+k} r_{t+k}, \quad \gamma = 0.9$$

This creates a natural incentive to acquire high-value buyers **early** in the sequence,
which differs from flat revenue maximization because discounting penalizes late revenue.
