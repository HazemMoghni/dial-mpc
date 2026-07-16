# discard/ — parked LCM-bridge & backup files (2026-07-10)

Files moved out of the package to strip dial-mpc down to the block1d **sync** (and async
deploy) modes. Nothing here is deleted — restore any file with the `mv` reversed.

| File here | Original location | What it is |
|---|---|---|
| `deploy/dial_lcm_bridge.py` | `dial_mpc/deploy/dial_lcm_bridge.py` | The LCM bridge (`dial-mpc-lcm-bridge`): DIAL controller driving c3-lcs's external Drake `block_sim` over LCM |
| `deploy/lcm_msgs/` | `dial_mpc/deploy/lcm_msgs/` | Vendored LCM message classes (dairlib lcm-gen output) the bridge imports |
| `examples/block1d_lcm.yaml` | `dial_mpc/examples/block1d_lcm.yaml` | Bridge run config (fairness-aligned solver constants + LCM channels) |
| `deploy/dial_plan.py.bak`, `deploy/dial_plan.py.bak-1782852394` | `dial_mpc/deploy/` | Pre-edit editor backups of dial_plan.py (the tracked file keeps the real edits) |

Companion edits commented out (grep for `discard/` in the repo):
- `setup.py:31` — the `dial-mpc-lcm-bridge` console-script line
- `dial_mpc/examples/__init__.py` — the `"block1d_lcm"` entry in `deploy_examples`

To fully restore the bridge: move the three LCM items back, uncomment both lines,
re-run `pip install -e .` (regenerates the console script). The c3-lcs side
(sweep harness `analysis/dial_block_experiments_log.py`, `analysis/run_block1d_sim.py`,
`analysis/experiment_params/dial_experiments_1d*.yaml`) was NOT touched — those are now
committed in c3-lcs (commit fd96841). Full context: ~/Repos/block1d_experiments.md §3.
