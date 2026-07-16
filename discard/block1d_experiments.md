# block1d Push Experiments — Files, Edits, and How to Run

DIAL-MPC solving the same 1D block-push task as MJPC and C3, in three modes:
**sync** (single process, internal MJX sim), **async** (two processes, internal MuJoCo sim,
shared memory), and **LCM bridge + sweep** (DIAL controller driving c3-lcs's external Drake
`block_sim` over LCM, with a sweep/logging harness).

All three verified running on this machine (r2d2; comparison pipeline verified 2026-07-09
with a 2x2 sweep that produced npy grids + heatmap PNGs). Status: sync is stable; the bridge
pushes the block in every test cell since the wire-velocity fix (the bridge had been feeding
DIAL the block's *spin* as its forward velocity), but the maps currently show high velocity
RMSE with ~0.9 contact ratio — DIAL is in the "repeated striking" regime, the exact kind of
behavioral distinction the comparison framework exists to expose. Whether some
(sigma, Ndiffuse) region reaches steady pushing is now a measurable question — run the two
sweeps below. The ee is undamped in both models (shared physics; changing it needs Grey's
sign-off).

"New"/"Edited" is relative to the upstream repos (LeCAR-Lab/dial-mpc, alinasarmiento/c3-lcs).
File references: repo-relative path, `path:line` where useful.

---

## 1. Sync (single process; sim waits for the controller)

### New files (dial-mpc)
| File | What it is |
|---|---|
| `dial_mpc/envs/block1d_env.py` | The DIAL env for the task: `Block1DEnvConfig` + `Block1DEnv`. Reward = MJPC's two nonzero residual terms (block-velocity tracking w=30, actuation w=0.001); defines `act2ctrl` (direct force mapping) used by the deploy modes. Registered as brax env `"block1d"`. |
| `dial_mpc/models/block1d/block1d.xml` | MJCF model: free-joint block + 1-DOF prismatic ee, ±6 N motor (gear 6). Trimmed copy of `mujoco_mpc/mjpc/tasks/block/task_push.xml` (+ its ee include); every physics value cited to its source in comments. |
| `dial_mpc/examples/block1d.yaml` | Sync run config. Solver knobs (Nsample 2048, Hsample 25, Hnode 5, Ndiffuse 4, …), env fields (dt/timestep 0.02), task fields (vel_target 0.3, weights, initial x's). Per-line provenance in comments. |

### Edited files (dial-mpc, shared registries)
| File | What changed |
|---|---|
| `dial_mpc/envs/__init__.py` | Added `Block1DEnvConfig` import + `"block1d"` entry in `_configs`. |
| `dial_mpc/examples/__init__.py` | Added `"block1d"` to the `examples` list. |

### Run
```bash
dial-mpc --example block1d
```
Runs `n_steps` (200) control steps (~min including JIT), prints mean reward, saves
`block1d/<timestamp>_states.npy` (`[step, qpos, qvel, ctrl]` rows),
`<timestamp>_predictions.npy`, and a brax HTML visualization, then serves it at
`http://127.0.0.1:5000`.

### Knobs
No CLI knobs — copy the yaml and pass `--config`:
```bash
cp ~/Repos/dial-mpc/dial_mpc/examples/block1d.yaml my.yaml   # edit Nsample, sigma_scale, ...
dial-mpc --config my.yaml
```
Initial positions: `block_init_x` / `ee_init_x` in the yaml (sync mode honors them).

---

## 2. Async (two processes, internal MuJoCo sim, shared memory)

### New files (dial-mpc)
| File | What it is |
|---|---|
| `dial_mpc/examples/block1d_deploy.yaml` | Async run config = `block1d.yaml` + the 8 `DialSimConfig` fields (`robot_name`, `sim_dt: 0.005`, `sync_mode: false`, …). `Ndiffuse: 1` for replan latency. `plot` must stay `false` (the live plotter hardcodes 4 joints; block1d has 1). |

### Edited files (dial-mpc)
| File | What changed |
|---|---|
| `dial_mpc/deploy/dial_plan.py` | Two small patches (rest untouched): **(a)** `pipeline_init` updated to current brax's `_reformat_contact(sys, data.contact)` signature (`:60-62`); **(b)** the control conversion prefers `env.act2ctrl` when the env defines it (`:221-227`) — block1d is a direct-force env, so the legged-robot PD `act2tau` math doesn't apply. Legged envs are unaffected (they don't define `act2ctrl`). |
| `dial_mpc/examples/__init__.py` | Added `"block1d_deploy"` to `deploy_examples`. |

`dial_sim.py`, `dial_core.py`, `dial_real.py`: **not edited**.

### Run (two terminals; sim FIRST — it creates the shared memory)
```bash
# Terminal 1 (opens a MuJoCo viewer window; needs a display)
dial-mpc-sim  --example block1d_deploy
# Terminal 2
dial-mpc-plan --example block1d_deploy
```
The sim waits (t=0) until the planner's first plan lands (JIT ≈ 30–50 s), then both run in
real time. Ctrl-C the planner first, then the sim (the sim unlinks the shared memory).

**If a run crashed** and the next `dial-mpc-sim` dies with `FileExistsError`:
```bash
rm -f /dev/shm/{acts,plan_time,refs,state,tau,time}_shm
```

### Knobs
Edit `block1d_deploy.yaml` (both processes read the same file; no CLI knobs on these two
scripts). Initial conditions: async **ignores** `block_init_x`/`ee_init_x` — both processes
start from the xml `"home"` keyframe, so IC changes mean editing the keyframe in
`dial_mpc/models/block1d/block1d.xml:96`. `record: true` dumps `[t, qpos, qvel, ctrl]` rows
to `.npy` on exit.

---

## 3. LCM bridge + sweep (DIAL vs the external Drake sim)

### New files (dial-mpc)
| File | What it is |
|---|---|
| `dial_mpc/deploy/dial_lcm_bridge.py` | The bridge (console script `dial-mpc-lcm-bridge`). LCM I/O thread decodes `ROBOT_STATE_SIMULATION` + `BLOCK_STATE_SIMULATION` into the latest state under one lock; a planner thread ports `dial_plan.py`'s loop (shift knots by elapsed sim time → `Ndiffuse × reverse_once` → spline to per-tick forces) and publishes an (Hsample+1)-point `lcmt_trajectory_block` (rows u/q/v, absolute sim-time `time_vec`) on **`INPUT_TRAJ`** each replan. It does **not** publish `ROBOT_INPUT` — `block_sim`'s own `SplineToRobotCommand` (wired when the sim params say `controller: MJPC`) interpolates the trajectory at sim time + 5 ms and publishes the single force there itself (currently pure feedforward: Ku=1, Kp=Kd=0 in `test_matrix/c3_options.yaml`). Skips publishing the JIT-warmup plan (its stale `time_vec` would be extrapolated by the sim's `FirstOrderHold` into a huge force impulse). On Ctrl-C saves `output_dir/lcm_bridge_block1d_<ts>/plans.npy` + `applied_inputs.npy`. |
| `dial_mpc/deploy/lcm_msgs/` (+ `dairlib/` pkg: `lcmt_robot_output.py`, `lcmt_object_state.py`, `lcmt_robot_input.py`, `lcmt_trajectory_block.py`) | LCM message classes vendored from `c3-lcs/analysis/dairlib/` (lcm-gen output; self-contained) so dial-mpc runs without a c3-lcs checkout on `PYTHONPATH`. |
| `dial_mpc/examples/block1d_lcm.yaml` | Bridge config: solver constants fairness-aligned with the MJPC 1d experiments (`Hsample: 20` = 0.4 s horizon, `Hnode: 4` = 5 spline knots, `timestep: 0.005` = mjpc agent_timestep, `Nsample: 512` documented as DIAL-specific) + `DialLcmConfig` fields (channels, `state_timeout_s`, `record`). `dt` must stay 0.02 (MBDPI hardcodes `ctrl_dt = 0.02`; the bridge asserts this). |

### Edited files (dial-mpc)
| File | What changed |
|---|---|
| `setup.py` | Added console script `dial-mpc-lcm-bridge=dial_mpc.deploy.dial_lcm_bridge:main`. |
| `dial_mpc/examples/__init__.py` | Added `"block1d_lcm"` to `deploy_examples`. |

### New files (c3-lcs) — no tracked c3-lcs file was edited
| File | What it is |
|---|---|
| `analysis/dial_block_experiments_log.py` | Sweep/log harness, sibling of `mjpc_block_experiments_log.py`, reusing `test_matrix_utils.py` unchanged. Per trial: patches `q_init`/`q_init_block` into `examples/block1d/parameters/test_matrix/block_sim_params.yaml` (the file `block_sim.cc:72` hardcodes; there is no sim CLI flag for ICs) and forces `controller: MJPC`; forwards swept controller knobs as bridge flags; launches `lcm-logger` (via `start_logging.py`) → bridge → `block_sim`; tears down after `--duration`. Sets `$C3` for its children so `start_logging.py`'s log path works without exporting it yourself. |
| `analysis/experiment_params/dial_experiments_1d.yaml` | Sweep definition, sibling of `mjpc_experiments_1d.yaml`: `sweep`/`sweep_params` lists, logger/controller/sim commands. `ee_x_offset` is the sim-side knob; any `DialConfig` field name (`Nsample`, `Ndiffuse`, `Ndiffuse_init`, `temp_sample`, `sigma_scale`, `horizon_diffuse_factor`, `traj_diffuse_factor`) is a controller-side knob, forwarded as `--<name with _ → ->`. |
| `analysis/run_block1d_sim.py` | Single-run sim launcher: CLI knobs (`--ee-x`, `--block-x`, `--realtime-rate`, `--visualize/--no-visualize`) with defaults from the base template; regenerates the test_matrix sim params, pins the private LCM bus (`LCM_DEFAULT_URL`), execs `block_sim`. |
| `user.bazelrc` (untracked, machine-local) | `build --override_repository=c3=/home/hazem/Repos/c3-ws` — repoints the WORKSPACE's dead `local_repository(name="c3", path="/home/grey/research/c3")` without editing the tracked `WORKSPACE`. |

### One-time build setup (already done on this machine)
`block_sim` needs `@c3//:libc3`, whose hardcoded path only exists on Grey's machine. The
reconstruction lives in a **git worktree** `~/Repos/c3-ws` (c3 `main` @ `5c08cb2`; your
`~/Repos/c3` checkout is untouched) with five shims so bzlmod-era c3 compiles in c3-lcs's
WORKSPACE/Drake-v1.28.0 world:
1. `WORKSPACE` file (one line, `workspace(name = "c3")`) — `local_repository` requires it;
2. dropped `load("@rules_{cc,java,python}//...")` lines (`lcmtypes/BUILD.bazel`, `bindings/pyc3/BUILD.bazel`) — native rules in Bazel 6;
3. `lcmtypes/BUILD.bazel` rewritten to Drake-1.28's `drake_lcm_cc_library` (target `lcmt_c3`, package `c3`) instead of the newer `@lcm//lcm-bazel` macros;
4. `MobyLcpSolver` → `MobyLCPSolver<double>` in `core/c3.cc` + `core/lcs.cc` (Drake 1.28 spelling/templating);
5. `ComputeSphereMeshDistance` body stubbed in `multibody/geom_geom_collider.cc` (newer QueryObject API; sphere-mesh contact is unused by block1d — no mesh geometry).

Rebuild after changes: `cd ~/Repos/c3-lcs && bazel build //examples/block1d:block_sim`
(bazelisk auto-uses 6.3.2 from `.bazelversion`; Drake is cached, rebuilds are fast).

### Run — manual (two terminals)
```bash
# Terminal 1 — from the c3-lcs repo root; knobs optional, defaults from the base template
cd ~/Repos/c3-lcs
python3 -m analysis.run_block1d_sim                       # defaults: ee -0.04, block 0.25
python3 -m analysis.run_block1d_sim --ee-x -0.3 --block-x 0.2 --no-visualize   # or chosen
# Terminal 2 — anywhere
dial-mpc-lcm-bridge --example block1d_lcm
```
`run_block1d_sim.py` regenerates the sim-params yaml from the base template (+ your flags:
`--ee-x`, `--block-x`, `--realtime-rate`, `--visualize/--no-visualize`), pins the **private
LCM bus**, and execs `block_sim`. Expect ~30 s of zero force (JIT; the sim holds zero until
a valid spline arrives), then forces flow. Meshcat viewer at `http://localhost:7000`
(forward the port over SSH, or `--no-visualize`). Watch channels with a subscriber on the
private URL: `ROBOT_STATE_SIMULATION` + `BLOCK_STATE_SIMULATION` (sim → bridge),
`INPUT_TRAJ` (bridge → sim), `ROBOT_INPUT` (sim's internal spline-tracker output).

**Why the private bus (this bit you): LCM's default URL (port 7667) is one shared multicast
group for the whole machine.** With two people running block sims (this box is shared with
Grey), each bridge hears BOTH sims: a foreign sim's clock ~305 s ahead fed our spline
tracker, which extrapolated it into huge/NaN forces (`sending u: nan`), and foreign
`INPUT_TRAJ` commands moved our block "on its own" (physics is fine: with a quiet bus the
block shows literally zero position/quaternion drift). Everything in this stack now runs on
`udpm://239.255.76.67:7669?ttl=0` — set in `block1d_lcm.yaml` (`lcm_url`), exported by
`run_block1d_sim.py` and the sweep harness (`LCM_DEFAULT_URL`). If you bypass the launcher,
launch the sim as `LCM_DEFAULT_URL='udpm://239.255.76.67:7669?ttl=0' ./bazel-bin/...`.

The bridge also now guards itself: it never publishes a non-finite plan (resets instead),
resets on a backwards sim clock (two-sims / restart signature), and pre-compiles both JIT
shapes during warmup so there is no mid-run recompile stall (the old
`planner overtime ~900 ms` + false `sim may be down` pair).

### Knobs — bridge CLI (defaults from `block1d_lcm.yaml`)
```
--Nsample --Ndiffuse --Ndiffuse-init --temp-sample --sigma-scale
--horizon-diffuse-factor --traj-diffuse-factor
--lcm-url --robot-state-channel --block-state-channel --traj-channel --record/--no-record
```
Initial positions (sim side): edit `q_init` (ee x) / `q_init_block[4]` (block x) in
`c3-lcs/examples/block1d/parameters/test_matrix/block_sim_params.yaml` before launching, or
let the sweep do it.

### Run — the behavioral-map workflow (all from the c3-lcs repo root)

The project's deliverable is the distance-vs-tuning heatmap, comparable cell-for-cell with
the MJPC/C3 maps. Two sweep definitions, one controller knob each (task parameter =
`ee_x_offset`, the initial ee–block separation; same distances as `mjpc_experiments_1d.yaml`):

The experiment knobs are **EE starting distance** (task difficulty), **sigma** (noise), and
**anneal** (Ndiffuse). The block origin stays fixed at 0.25, as in the mjpc sweeps.

```bash
# SINGLE RUN — one point on a graph (short keys ee / sigma / anneal):
python3 -m analysis.dial_block_experiments_log --sweep sigma  --point ee=0.5,sigma=1.0
python3 -m analysis.dial_block_experiments_log --sweep anneal --point ee=0.5,anneal=4
#   (all three knobs at once, logged under the sigma graph:)
#   ... --sweep sigma --point ee=0.5,sigma=1.0,anneal=4

# FULL GRAPH — fills the whole 2D matrix, origin -> end in steps:
python3 -m analysis.dial_block_experiments_log --sweep sigma    # distance x sigma
python3 -m analysis.dial_block_experiments_log --sweep anneal   # distance x anneal

# Logs -> npy grids + the three heatmap PNGs (RMSE / vel-std / contact ratio),
# the same script + metric code that scores mjpc:
python3 analysis/block_plots/plot_1d_new.py --method dial-sigma  --load_new True
python3 analysis/block_plots/plot_1d_new.py --method dial-anneal --load_new True
```

The matrix ranges live in the sweep yaml as `sweep_params: {ee_x_offset: [start, end, step],
sigma_scale: [start, end, step]}` — the same np.arange convention (end exclusive) that
`mjpc_experiments_1d.yaml` uses. Defaults: distance 0.05->1.00 step 0.05 (identical to the
mjpc sweep, for cell-for-cell comparability) x sigma 0.1->1.5 step 0.1 (300 trials, ~5 h) or
x anneal 1->8 (160 trials, ~2.8 h). Edit the ranges in the yaml to coarsen or extend.

Logs land in `analysis/logs/block1d/dial-{sigma,anneal}/<eex_XXX_{sigmaYYYY|annealZZ}>/`;
the plot step writes `analysis/block_plots/np_1d/dial-*_{cost,velvar,contact}.npy` and
`dial-*_1d_{rmse,velvar,contact}.png` — directly comparable with the `mjpc_*`/`c3_*`
siblings in the same folder. Metrics (from `mjpc_success_eval`, sim channels only, so
identical across controllers): block-velocity RMSE vs the shared 0.3 m/s target from first
block motion, velocity std, and contact ratio; a never-moved block shows as a gray cell.

Trial duration defaults to 60 s: bridge startup (imports + tracing + cached JIT) is ~30 s,
and the metrics start at first block motion, so startup never pollutes them.

**Fairness alignment with the MJPC 1d experiments** (`block1d_lcm.yaml` comments carry the
refs): horizon 0.4 s (`Hsample: 20` = mjpc `t_horizon`), 5 spline knots (`Hnode: 4` = mjpc
`sampling_spline_points`), rollout-model integration at 0.005 s (`timestep` = mjpc
`agent_timestep`), same sim template / block start 0.25 / ee-placement convention
(−offset−0.02) / target 0.3 m/s. `Nsample: 512` is documented as DIAL-specific (GPU
sampling budget; mjpc's 50 CPU rollouts are not a fairness target). One real bug fixed for
comparability: the wire's block velocity is **angular-first** (Drake floating-base order;
`ObjectStateSender`, confirmed by mjpc's Handler swap and the analysis reading forward
velocity at `velocity[-3]`) — the bridge previously fed the planner the block's spin as its
forward velocity.
