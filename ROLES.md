# ROLES.md

| Teammate | Role | Responsibility |
|----------|------|----------------|
| Ramazan Berk ASLAN | A | Value-based — DQN → Double DQN → Dueling DQN |
| Resul Berk DURSUN | B | Policy-based — REINFORCE + GAE → A2C; DDPG |
| Alaaddin Enes SARI | C | Planning — Dyna-Q (charging/repositioning), world model, n_planning_steps ablation |
| Eray IRMAK | Joint | Offline RL (Ch. 20) + Multi-agent RL (Ch. 21) |

## Joint components (Eray IRMAK)
- **Offline RL (Ch. 20):** pooled mixed-quality dataset from all three teammates'
  policies (DQN + A2C + REINFORCE + Dyna-Q) plus baselines; demonstrated naive
  offline-DQN over-estimation failure; fixed with CQL; conservatism (alpha)
  ablation on two axes (Q-control and cost_per_order); CQL beats naive and BC;
  two-dataset coverage analysis; off-policy-evaluation bonus.
- **Multi-agent (Ch. 21):** parameter-shared IDQN on DroneDispatchMA-v0; 3 seeds
  with mean±std; non-stationarity analysis (cross-seed variance + action
  distributions); parameter-sharing ON/OFF ablation.
