# `dial_plan_standalone.py` — A Complete Execution Walkthrough

**File:** `dial_mpc/deploy/dial_plan_standalone.py` (331 lines)
**Role:** a standalone DIAL-MPC *planner process* that closes a control loop over LCM
against an external Drake/C3 simulator (`c3-lcs` `block1d` example), instead of the
shared-memory sim (`dial_sim.py`) the stock deploy pipeline uses.

This document follows the file in **actual execution order**: import time first, then
`if __name__ == "__main__"` → `main()`, then object construction, then the control loop.
Whenever execution enters a function defined elsewhere, the walkthrough takes a sidetrack
through that function and returns to the call site. Everything stated as fact was read
from the sources listed below; items that are interpretation or potential issues are
explicitly marked **[interpretation]** or **[concern]**.

**Sources inspected** (repo-relative unless noted):

| Source | What it contributes |
|---|---|
| `dial_mpc/deploy/dial_plan_standalone.py` | the file being explained |
| `dial_mpc/core/dial_core.py`, `dial_mpc/core/dial_config.py` | `MBDPI`, `DialConfig`, the reference "sigma fix" |
| `dial_mpc/envs/base_env.py`, `dial_mpc/envs/block1d_env.py`, `dial_mpc/envs/__init__.py` | environment, reward, `act2ctrl`/`act2tau` |
| `dial_mpc/config/base_env_config.py` | `BaseEnvConfig` fields |
| `dial_mpc/examples/__init__.py`, `dial_mpc/examples/block1d_deploy.yaml` | example registry and the run's actual numbers |
| `dial_mpc/utils/io_utils.py` | config/model path helpers |
| `dial_mpc/models/block1d/block1d.xml` | the MuJoCo model (nq=8, nv=7, nu=1, gear=6) |
| `dial_mpc/deploy/dial_plan.py` | the shared-memory twin this file was derived from |
| installed `brax` 0.14.1 (`brax/mjx/pipeline.py`, `brax/mjx/base.py`) | what the local `pipeline_init` patch replicates/omits |
| installed `mujoco` 3.9.0 / MJX | `make_data` behavior, `Data._impl`, `contact` property |
| installed `lcm` Python module docstrings | exact `handle`/`handle_timeout`/queue semantics |
| installed `jax_cosmo.scipy.interpolate` | spline math, k=2, extrapolation behavior |
| `~/Repos/c3-lcs/analysis/dairlib/*.py` | LCM message definitions (wire layout) |
| `~/Repos/c3-lcs/examples/c3-import/block1d/` | the Drake sim and the `INPUT_TRAJ` consumer |

---

## 0. The big picture

Three processes, connected by LCM (UDP-multicast pub/sub middleware):

```
┌────────────────────────┐   ROBOT_STATE_SIMULATION (lcmt_robot_output)
│  block_sim (Drake/C3)  │──────────────────────────────────────────────┐
│  bazel-built C++ sim   │   BLOCK_STATE_SIMULATION (lcmt_object_state) │
│                        │──────────────────────────────────────────────┤
│  integrates physics,   │                                              ▼
│  tracks INPUT_TRAJ     │                          ┌─────────────────────────────────┐
│  with a PD+FF law      │   INPUT_TRAJ             │ dial_plan_standalone.py (this)  │
│                        │◀─────────────────────────│ JAX/MJX sampling MPC planner    │
└────────────────────────┘  (lcmt_timestamped_      │ replans a 0.3 s horizon per tick│
        ▲                    saved_traj)            └─────────────────────────────────┘
        │
┌───────┴────────────┐
│  block_visualizer  │  (passive, draws the sim state)
└────────────────────┘
```

The task (from `dial_mpc/models/block1d/block1d.xml` and `block1d_env.py`): a 0.5 kg box
("block") rests on a low-friction floor; a 0.1 kg spherical end-effector ("ee") on a
single prismatic x-joint can push it with a force limited to ±6 N. The objective is to
make the block translate at a target velocity of 0.3 m/s while spending little force.

The planner is **stateful in one thing only**: the current plan `Y` — a small matrix of
control "nodes" spanning 0.3 s into the future. Every loop pass it:

1. blocks until a *fresh* state pair arrives from the sim,
2. time-shifts `Y` by however much sim time elapsed since the last plan,
3. improves `Y` with one (or, on the first pass, ten + one) annealed MPPI/diffusion
   updates, each of which rolls out 2049 candidate plans in parallel on the GPU through
   an MJX (MuJoCo-in-JAX) model of the same physics,
4. converts `Y` to a 16-knot trajectory message that byte-imitates what the native C3
   controller would publish, and sends it on `INPUT_TRAJ`.

The Drake sim, meanwhile, never waits for the planner: it keeps integrating at its own
rate and tracks whatever trajectory message it saw last. That asynchrony is why the
"shift by elapsed time" and "fresh state" machinery exists.

---

## 1. Import time — what happens before any function runs

### 1.1 The run recipe (lines 1–7)

The header comment documents the intended launch sequence — three terminals:

```python
# Terminal 1
# cd ~/Repos/c3-lcs && ./bazel-bin/examples/c3-import/block1d/block_sim
# Terminal 2
# PYTHONPATH="$HOME/Repos/c3-lcs/analysis:$PYTHONPATH" \
# python3 -m dial_mpc.deploy.dial_plan_standalone --example block1d_deploy
# Viz:
# cd ~/Repos/c3-lcs && ./bazel-bin/examples/c3-import/block1d/block_visualizer
```

The `PYTHONPATH` prefix matters: the generated Python LCM bindings (`dairlib` package)
live in `~/Repos/c3-lcs/analysis/dairlib/`, not on the normal path. Without it the
`from dairlib import ...` at line 48 raises `ImportError` and the process dies before
`main()` is ever reached.

### 1.2 Imports (lines 9–45) — and their side effects

Python executes every top-level statement of every imported module, so several imports
here do real work:

- **Lines 9–24, stdlib & utilities.** `os`, `time`, `sys`, `importlib`, `argparse`,
  `yaml`, `functools`/`partial`; `numpy as np` (host-side arrays for LCM I/O); `tqdm`
  (progress bars — imported but never used in this file); `art` (ASCII banner) and
  `emoji` (the 🚀 print); `lcm` (the LCM Python binding, a C extension).
- **Lines 25–27, the JAX stack.** `jax` and `jax.numpy as jnp` (device arrays,
  tracing/JIT); `from mujoco import mjx` — MJX, the JAX re-implementation of MuJoCo:
  physics stepping becomes pure, differentiable, `vmap`-able array code that runs on GPU.
- **Lines 29–35, brax.** `brax.envs as brax_envs` (a registry + `PipelineEnv` machinery
  wrapping MJX), `State` (the env-level state pytree: `pipeline_state, obs, reward, done,
  metrics, info`), `MjxState` (the physics-level state, brax's wrapper around
  `mjx.Data`), `_reformat_contact` (private brax helper, see §3.4), `Transform`/`Motion`
  (spatial-algebra pytrees), and `InterpolatedUnivariateSpline` from **jax_cosmo** — a
  JIT-compatible spline used for all node↔dense-control conversions (§3.2, §5.3).
- **Lines 37–45, dial_mpc itself.** Two of these have load-bearing side effects:
  - `import dial_mpc.envs as dial_envs` executes `dial_mpc/envs/__init__.py`, which
    imports every env module including `block1d_env.py`. The **last line of
    `block1d_env.py`** is `brax_envs.register_environment("block1d", Block1DEnv)`
    (`block1d_env.py:148`) — this import is what makes
    `brax_envs.get_environment("block1d", ...)` in `main()` work at all.
  - `from dial_mpc.examples import deploy_examples` loads the allow-list of example
    names; `"block1d_deploy"` is in it (`examples/__init__.py:13-19`).
  - `from dial_mpc.core.dial_core import DialConfig, MBDPI` pulls in the planner core
    (§3.2). (`DialConfig` actually lives in `dial_mpc/core/dial_config.py`; `dial_core`
    re-exports it.)

### 1.3 LCM bindings and channel names (lines 47–58)

```python
from dairlib import (
    lcmt_robot_output,      # sim → planner: robot (EE) state
    lcmt_object_state,      # sim → planner: block (free body) state
    lcmt_timestamped_saved_traj,  # planner → sim: outer envelope
    lcmt_trajectory_block,        # planner → sim: one named trajectory
)

ROBOT_STATE_CHANNEL = "ROBOT_STATE_SIMULATION"
BLOCK_STATE_CHANNEL = "BLOCK_STATE_SIMULATION"
TRAJ_CHANNEL = "INPUT_TRAJ"
```

These are lcm-gen–generated classes; each is a plain Python object with typed `__slots__`
plus `encode()`/`decode()` that pack/unpack a big-endian binary blob. Confirmed layouts
(from `c3-lcs/analysis/dairlib/*.py`):

| Type | Fields (in wire order) |
|---|---|
| `lcmt_robot_output` | `utime:int64` (µs), `num_positions/num_velocities/num_efforts:int32`, `position_names[]`, `position[]:double`, `velocity_names[]`, `velocity[]:double`, `effort_names[]`, `effort[]:double`, `imu_accel[3]` |
| `lcmt_object_state` | `utime:int64`, `object_name:str`, `num_positions/num_velocities:int32`, `position_names[]`, `position[]:double`, `velocity_names[]`, `velocity[]:double` |
| `lcmt_trajectory_block` | `trajectory_name:str`, `num_points:int32`, `num_datatypes:int32`, `time_vec:double[num_points]`, `datapoints:double[num_datatypes][num_points]`, `datatypes:str[num_datatypes]` |
| `lcmt_saved_traj` | `metadata:lcmt_metadata`, `num_trajectories:int32`, `trajectories[]`, `trajectory_names[]` |
| `lcmt_timestamped_saved_traj` | `utime:int64`, `saved_traj:lcmt_saved_traj` |

Note `datapoints` is indexed **[row = datatype][column = knot]** — that's why the
publisher later builds a `(6, num_points)` matrix (§5.8).

The channel names are copied from
`c3-lcs/examples/c3-import/block1d/parameters/lcm_channels_simulation.yaml` (confirmed:
`robot_state_channel: ROBOT_STATE_SIMULATION`, `block_state_channel:
BLOCK_STATE_SIMULATION`, `traj_channel: INPUT_TRAJ`).

### 1.4 XLA flag (lines 61–64)

```python
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
```

Appends a GPU compiler flag (use Triton for general matrix multiplies) to the
environment **before** JAX initializes its backends; XLA reads `XLA_FLAGS` lazily at
first compile. Same two lines exist in `dial_core.py:31-33`; the comment claims ~30%
step-rate improvement on some GPUs. Harmless on CPU-only runs.

---

## 2. `if __name__ == "__main__":` → `main()` (lines 329–330, 268–326)

Running the module as a script (`python3 -m dial_mpc.deploy.dial_plan_standalone ...`)
sets `__name__ == "__main__"`, so line 330 calls `main()` with `args=None` (argparse then
reads `sys.argv`).

### 2.1 Banner and argument parsing (lines 269–292)

`art.tprint(...)` prints a big ASCII banner. The parser defines a **required, mutually
exclusive** group — exactly one of:

- `--config <path>`: path to a YAML config file,
- `--example <name>`: a named example resolved inside the installed package,
- `--list-examples`: print names and exit,

plus an independent `--custom-env <module>` which, if given, appends the current working
directory to `sys.path` and imports the named module for its registration side effects
(lines 294–296) — the same trick `block1d_env.py` uses, but for user code outside the
package. For the documented run, only `--example block1d_deploy` is passed.

### 2.2 Loading the config dict (lines 298–311)

`--list-examples` short-circuits (prints `deploy_examples` and returns). Otherwise, for
`--example`, the name is validated against `deploy_examples` and the YAML is loaded:

```python
config_dict = yaml.safe_load(open(get_example_path(args.example + ".yaml"), "r"))
```

**Sidetrack — `get_example_path` (`utils/io_utils.py:10-12`):** uses
`importlib.resources.path("dial_mpc.examples", name)` to find the YAML *inside the
installed package*, so it works regardless of the current directory. Returns a
`pathlib.Path`.

`yaml.safe_load` produces one flat Python `dict` mixing planner keys, env keys, and
(unused here) sim-process keys. For `block1d_deploy.yaml` the values that matter:

| Key | Value | Consumed by |
|---|---|---|
| `env_name` | `block1d` | both configs |
| `seed` | 0 | `DialConfig` |
| `Nsample` | 2048 | `DialConfig` |
| `Hsample` | 15 | `DialConfig` |
| `Hnode` | 5 | `DialConfig` |
| `Ndiffuse` | 1 | `DialConfig` |
| `Ndiffuse_init` | 10 | `DialConfig` |
| `temp_sample` | 0.05 | `DialConfig` |
| `horizon_diffuse_factor` | 0.9 | `DialConfig` |
| `traj_diffuse_factor` | 0.5 | `DialConfig` |
| `update_method` | `mppi` | `DialConfig` |
| `dt` | 0.02 | `Block1DEnvConfig` (control step) |
| `timestep` | 0.02 | `Block1DEnvConfig` (MJX physics step) |
| `action_scale` | 1.0 | `Block1DEnvConfig` |
| `vel_target` / `w_vel` / `w_ctrl` | 0.3 / 30.0 / 0.001 | `Block1DEnvConfig` (reward) |
| `block_init_x` / `ee_init_x` | 0.25 / −0.04 | `Block1DEnvConfig` (unused live — state comes from LCM) |

(`sigma_scale` is absent from the YAML, so the `DialConfig` default `1.0` applies.
`n_steps`, `robot_name`, `sim_dt`, `sync_mode` etc. are for the shared-memory sim twin
and are simply ignored by this process — `load_dataclass_from_dict` filters by field
name, see next.)

### 2.3 Building the two config dataclasses (lines 313–318)

```python
dial_config = load_dataclass_from_dict(DialConfig, config_dict)
env_config_type = dial_envs.get_config(dial_config.env_name)
env_config = load_dataclass_from_dict(env_config_type, config_dict, convert_list_to_array=True)
```

**Sidetrack — `load_dataclass_from_dict` (`io_utils.py:15-24`):** intersects the
dataclass's declared field names with the dict's keys and constructs the dataclass from
just those — this is how one flat YAML feeds several dataclasses without
`unexpected keyword` errors. With `convert_list_to_array=True` any YAML list value
becomes a `jnp.array` (needed for envs whose configs hold, e.g., per-joint gain lists;
block1d has none).

**Sidetrack — `DialConfig` (`core/dial_config.py:4-23`):** the planner's hyperparameters:
`seed`, `output_dir`, `n_steps` (unused in deploy), `env_name`, and the diffusion knobs —
`Nsample` (candidate plans per update), `Hsample` (dense control horizon; there are
`Hsample+1` control steps), `Hnode` (spline-node horizon; `Hnode+1` nodes), `Ndiffuse` /
`Ndiffuse_init` (annealing iterations per replan / on the first replan), `temp_sample`
(MPPI softmax temperature), `horizon_diffuse_factor` (per-node noise profile),
`traj_diffuse_factor` (per-iteration noise decay), `update_method` (only `"mppi"`
exists), `sigma_scale` (global noise scale).

**Sidetrack — `dial_envs.get_config` (`envs/__init__.py:34-35`):** dictionary lookup in
`_configs`, where `"block1d": Block1DEnvConfig` was registered at import
(`envs/__init__.py:26`). `Block1DEnvConfig` (`block1d_env.py:22-31`) extends
`BaseEnvConfig` (`config/base_env_config.py`) — which supplies `kp=30, kd=1` (unused
here), `dt=0.02`, `timestep=0.02`, `backend="mjx"`, `leg_control`, `action_scale` — with
the task fields `vel_target, w_vel, w_ctrl, block_init_x, ee_init_x`.

### 2.4 Creating the environment (line 319)

```python
env = brax_envs.get_environment(dial_config.env_name, config=env_config)
```

Brax looks up `"block1d"` in its registry (populated at import time, §1.2) and calls
`Block1DEnv(config=env_config)`.

**Sidetrack — `Block1DEnv.__init__` (`block1d_env.py:35-52`) → `BaseEnv.__init__`
(`base_env.py:15-29`):**

1. `BaseEnv.__init__` asserts `dt % timestep == 0`, computes
   `n_frames = int(dt/timestep)` = **1** (one physics step per control step — the
   planner's model integrates at 0.02 s), then calls `self.make_system(config)`.
2. **`Block1DEnv.make_system` (`block1d_env.py:54-61`):** loads
   `dial_mpc/models/block1d/block1d.xml` via `mujoco.MjModel.from_xml_path`, converts to
   a brax/MJX `System` with `brax.io.mjcf.load_model`, and overrides the XML's
   `<option timestep="0.005">` with the YAML `timestep` (0.02) via `tree_replace`.
3. Back in `BaseEnv.__init__`: `PipelineEnv.__init__(sys, "mjx", n_frames, debug)` sets
   up the MJX pipeline; then it caches `physical_joint_range = sys.jnt_range[1:]` (all
   joints *except* the block's free joint), `joint_torque_range =
   sys.actuator_ctrlrange` (= `[[-1, 1]]` from the XML), and `nq/nv`.
4. `Block1DEnv.__init__` then reads the `"home"` keyframe qpos and overrides indices 0
   and 7 with `block_init_x`/`ee_init_x`, storing `_init_q`; and caches
   `_actuator_gear = [6.0]` from `mj_model.actuator_gear[:, 0]`.

**The model itself (`block1d.xml`, verified by loading it):** `nq = 8`
(`qpos = [block x y z, quat w x y z, ee_x]`), `nv = 7`
(`qvel = [block vx vy vz, wx wy wz, ee_vx]`), `nu = 1` — a single
`<motor joint="ee_x" gear="6" ctrlrange="-1 1">`. So a normalized control `ctrl ∈ [-1,1]`
becomes a physical force `6·ctrl` N on the EE. `env.action_size` is therefore **1**.

### 2.5 Constructing the publisher and entering the loop (lines 321–326)

```python
mbd_publisher = MBDPublisher_standalone(env, env_config, dial_config)
try:
    mbd_publisher.main_loop()
except KeyboardInterrupt:
    pass
```

`main_loop` never returns; Ctrl-C is the intended exit and is swallowed for a clean
shutdown. Everything from here on is §3 (construction) and §5 (the loop).

---

## 3. `MBDPublisher_standalone.__init__` (lines 87–117)

```python
self.dial_config = dial_config
self.env = env
self.env_config = env_config
self.mbdpi = MBDPI(self.dial_config, self.env)
```

### 3.1 Sidetrack — `MBDPI.__init__` (`dial_core.py:51-89`)

`MBDPI` ("Model-Based Diffusion / Path Integral") is the sampling-MPC core shared with
the synchronous runner. Its constructor precomputes everything the per-tick update needs:

- `self.nu = env.action_size` → **1**.
- `self.update_fn = softmax_update` (the only registered `update_method`, `"mppi"`).
  `softmax_update(weights, Y0s, sigma, mu_0t)` (`dial_core.py:45-48`) is just the
  weighted average `mu = Σₙ weights[n]·Y0s[n]` returned with `sigma` unchanged.
- **`sigma_control` — the per-node noise profile** (`dial_core.py:66-70`):

  ```python
  self.sigma_control = args.horizon_diffuse_factor ** jnp.arange(args.Hnode + 1)[::-1]
  self.sigma_control *= sigma_scale
  ```

  Shape `(Hnode+1,) = (6,)`. With `horizon_diffuse_factor = 0.9`, the reversed exponent
  runs 5…0, giving `[0.9⁵ … 0.9⁰] = [0.590, 0.656, 0.729, 0.81, 0.9, 1.0]`: nodes **near
  the present get small noise** (don't thrash what you're about to execute), the horizon
  end gets full noise (explore where uncertainty is cheap). `sigma_scale = 1.0` here.
  (The local variables `sigma0/sigma1/A/B` at `dial_core.py:61-65` are computed and never
  used — leftover dead code. Similarly `MBDPI.reverse` at `dial_core.py:147-158`
  references a `self.sigmas` that is never constructed anywhere; that method is dead in
  this codebase and unused by this file.)

- **Time grids** (`dial_core.py:74-77`):

  ```python
  self.ctrl_dt = 0.02
  self.step_us   = jnp.linspace(0, 0.02 * Hsample, Hsample + 1)   # (16,): 0.00, 0.02, …, 0.30
  self.step_nodes = jnp.linspace(0, 0.02 * Hsample, Hnode + 1)    # (6,):  0.00, 0.06, …, 0.30
  self.node_dt = 0.02 * Hsample / Hnode                            # 0.06
  ```

  **[concern]** `MBDPI.ctrl_dt` is hard-coded to `0.02` rather than read from
  `env_config.dt`. In this run they coincide, but changing `dt` in the YAML without
  editing `dial_core.py` would silently desynchronize the planner's spline clock from
  the env's integration step.

- **Node↔control conversion.** A plan is stored compactly as `Hnode+1 = 6` node values
  per actuator; the env needs `Hsample+1 = 16` dense controls. `node2u`
  (`dial_core.py:91-95`) fits a **quadratic (`k=2`) interpolating spline** through the 6
  `(step_nodes, node)` points (jax_cosmo's `InterpolatedUnivariateSpline` — a pure-JAX,
  jittable reimplementation of SciPy's; extrapolation is always enabled) and evaluates it
  at the 16 `step_us` times. `u2node` (`dial_core.py:97-101`) is the inverse resampling.
  Both are wrapped in three layers:

  | Wrapper | Axes | Maps |
  |---|---|---|
  | `node2u_vmap` / `u2node_vmap` | `in_axes=1, out_axes=1` | per-actuator column: `(6, nu) → (16, nu)` |
  | `node2u_vvmap` / `u2node_vvmap` | `in_axes=0` over batch | `(N, 6, nu) → (N, 16, nu)` |

  `jax.vmap` vectorizes a function over one axis without a Python loop; the composition
  gives batched, per-column spline evaluation entirely inside XLA.

- **Rollout machinery** (`dial_core.py:80-81`):

  ```python
  self.rollout_us = jax.jit(functools.partial(rollout_us, self.env.step))
  self.rollout_us_vmap = jax.jit(jax.vmap(self.rollout_us, in_axes=(None, 0)))
  ```

  `rollout_us(step_env, state, us)` (`dial_core.py:36-42`) runs
  `jax.lax.scan(step, state, us)` — a compiled sequential loop that steps the env once
  per row of `us` and stacks `(state.reward, state.pipeline_state)` per step. The vmap
  version broadcasts one initial `state` (`in_axes=None`) across a **batch of control
  sequences** (`in_axes=0`): `(N, 16, nu) → rews (N, 16), pipeline_states (N, 16, …)`.
  This one line is the GPU fan-out that makes sampling MPC feasible.

### 3.2 Back in `__init__`: RNG, jitted helpers, the plan buffer (lines 98–105)

```python
self.rng = jax.random.PRNGKey(seed=self.dial_config.seed)
self.pipeline_init_jit = jax.jit(pipeline_init)
self.shift_vmap = jax.jit(jax.vmap(self.shift, in_axes=(1, None), out_axes=1))

self.Y = jnp.zeros([self.dial_config.Hnode + 1, self.mbdpi.nu])   # (6, 1)
self.ctrl_dt = env_config.dt                                       # 0.02
self.n_acts = self.dial_config.Hsample + 1                         # 16
```

- JAX PRNG is explicit and splittable: `PRNGKey(0)` is a `(2,)` uint32 key that will be
  `split` every planning pass so no randomness is reused.
- `pipeline_init` (module-level, §3.4) is jitted once here.
- `shift_vmap` vectorizes the **time-shift** `shift(x, shift_time)` (§5.3) over the
  actuator axis of `Y` (`in_axes=(1, None)`: map over columns of the first argument,
  broadcast the scalar). For `nu=1` the vmap is trivial but keeps the code
  env-agnostic.
- **`self.Y` is the persistent plan**: `(Hnode+1, nu) = (6, 1)`, node values in
  normalized action units (clipped to `[-1, 1]` during planning; ×6 N at the actuator).
  Row `i` is the control node at `step_nodes[i]` seconds **from the most recent plan
  time**. It starts at all zeros ("do nothing").
- `self.ctrl_dt` — note this one *is* `env_config.dt`, distinct from the hard-coded
  `mbdpi.ctrl_dt` (§3.1 concern). `n_acts = 16` is used only for the "long time
  unplanned" threshold (§5.3).

### 3.3 LCM plumbing (lines 107–117)

```python
self.lc = lcm.LCM()
self.latest_robot_state = None
self.latest_block_state = None
self.received_robot = False
self.received_block = False
sub_r = self.lc.subscribe(ROBOT_STATE_CHANNEL, self.handle_robot_state)
sub_b = self.lc.subscribe(BLOCK_STATE_CHANNEL, self.handle_block_state)
sub_r.set_queue_capacity(1)
sub_b.set_queue_capacity(1)
```

LCM facts (verified against the installed module's docstrings):

- `lcm.LCM()` joins the default UDP-multicast group; **no callback ever fires
  spontaneously** — messages are queued per subscription and dispatched only inside
  `handle()`/`handle_timeout()` calls on *this* thread. That single-threaded model is
  why the two `handle_*` callbacks below can set plain attributes without locks.
- `subscribe(channel, callback)` registers `callback(channel: str, data: bytes)` and
  returns an `LCMSubscription`.
- `set_queue_capacity(1)`: per LCM docs, *"Sets the maximum number of received but
  unhandled messages to queue… if messages start arriving faster than they are handled,
  they will be discarded after more than this number start piling up."* I.e. once one
  message is queued, **later arrivals are dropped** — so while the planner spends tens
  of milliseconds (or ~30 s during first-pass JIT) inside JAX, at most one (the oldest,
  hence stale) message per channel accumulates. The comment at line 115 says exactly
  this: "at most one stale message survives a planning pass." That stale survivor is
  then explicitly drained (§5.2).

The callbacks (lines 119–125) are two-liners:

```python
def handle_robot_state(self, channel, data):
    self.latest_robot_state = lcmt_robot_output.decode(data)
    self.received_robot = True
```

`decode` is the generated parser: it checks the 8-byte type fingerprint and unpacks the
big-endian fields into a fresh message object. The `received_*` flags mark "a new message
arrived since the flags were last cleared" — the freshness handshake used in §5.2.

### 3.4 Sidetrack — module-level `pipeline_init` (lines 67–85)

This is a **local, patched copy of `brax.mjx.pipeline.init`** used to build the initial
physics state directly from an externally supplied `(q, qd)`:

```python
def pipeline_init(sys, q, qd) -> MjxState:
    data = mjx.make_data(sys)
    data = data.replace(qpos=q, qvel=qd)
    q, qd = data.qpos, data.qvel
    x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
    cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
    offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
    offset = Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)
    contact = _reformat_contact(sys, data.contact)
    return MjxState(q=q, qd=qd, x=x, xd=xd, contact=contact, **data.__dict__)
```

- `mjx.make_data(sys)` allocates a zeroed `mjx.Data` pytree (the JAX analog of
  `mjMData`); `replace` writes the LCM-sourced generalized position/velocity into it.
- The `x` (per-body world transforms), `xd` (per-body world motion) blocks are copied
  verbatim from brax's `init`; `[1:]` skips the world body.
- **The local patch (comment, lines 82–84):** current brax's `_reformat_contact(sys,
  contact)` takes and returns a `Contact`; the upstream dial-mpc copy of this function
  passed the whole `data`. Verified against installed brax 0.14.1
  (`brax/mjx/pipeline.py:30-41`) — the local signature is the correct current one.
- A mujoco-3.9 subtlety verified live: `data.__dict__` does **not** contain a
  `contact` key (contact now lives behind `Data._impl`, exposed as a property), so
  passing `contact=contact` alongside `**data.__dict__` is not a duplicate-kwarg error —
  it's the only way the field gets set.
- **[concern] The one real behavioral difference vs brax:** brax's `init` calls
  `data = mjx.forward(sys, data)` before reading `xpos/xquat/cvel` — the forward pass
  computes kinematics from `qpos`. This local copy **omits `mjx.forward`**, so
  `data.xpos` etc. are still the zeros from `make_data` (verified live: `xpos` is all
  zeros), and the returned state's `x`, `xd`, and `contact` are placeholders that do
  **not** correspond to `(q, qd)`. This is harmless *for this pipeline* because (a) the
  very next `env.step` calls `mjx.step`, which recomputes the full forward pass from
  `qpos/qvel` before anything reads positions, and (b) `Block1DEnv`'s reward and obs
  read only `qpos/qvel` (`block1d_env.py:107-116, 135-143`). But any env whose reward
  or observation read `pipeline_state.x` / site positions of the *initial* state would
  silently get zeros. Skipping `forward` is presumably deliberate — it avoids ~one extra
  compiled physics call per replan — but it trades away state-consistency that brax
  guarantees. **[interpretation]**

---

## 4. Interlude — the LCM state interface, in detail

Three small methods form the entire sim-facing input surface. They run before and inside
every loop pass, so they're documented here once.

### 4.1 `wait_for_fresh_state` (lines 127–135)

```python
def wait_for_fresh_state(self):
    # discard anything queued while planning, then block for a new pair
    while self.lc.handle_timeout(0) > 0:
        pass
    self.received_robot = False
    self.received_block = False
    while not (self.received_robot and self.received_block):
        self.lc.handle()
    return self.get_state_snapshot()
```

- **`handle_timeout(timeout_millis)` is an LCM method** (C extension): it waits up to
  `timeout_millis` for an incoming message, dispatches **one** message to its
  subscription callback(s) if available, and returns an `int` — `0` if it timed out
  (nothing dispatched), `>0` if a message was handled (raises `ValueError`/`IOError` on
  bad input/socket error). With **timeout `0` it never waits**: it either dispatches one
  already-queued message and returns positive, or returns `0` immediately. So
  `while self.lc.handle_timeout(0) > 0: pass` is a **non-blocking drain**: it pops the
  (at most one per channel, thanks to queue capacity 1) stale message that arrived
  during the previous planning computation, lets the callbacks overwrite
  `latest_*_state`, and exits the instant the queues are empty. Nonblocking matters
  because the point is to *empty* the queue, not to wait for more.
- The flags are then cleared, and `handle()` — the **blocking** variant, "waits for and
  dispatches the next incoming message" — is called repeatedly until **both** a robot
  and a block message have arrived **after** the clear. This guarantees the snapshot
  reflects sim state published *after* the previous plan finished, i.e. the planner
  always plans from now, never from a queue backlog. Typically this loop runs exactly
  two `handle()` calls (one message per channel per sim publish tick).
- **[interpretation]** The robot and block messages are only "paired" by arrival, not by
  timestamp — if the sim publishes them at slightly different phases, `qpos` can mix a
  robot state and block state one sim-tick apart. With both published from the same
  Drake diagram at the same rate this is at most one publish period of skew.

### 4.2 `get_state_snapshot` (lines 137–146) — Drake wire → MuJoCo layout

```python
robot = self.latest_robot_state
block = self.latest_block_state
t = robot.utime * 1e-6
p = block.position
v = block.velocity
# wire is quat-first / angular-first (drake); mujoco is pos-first / linear-first
qpos = np.array([p[4], p[5], p[6], p[0], p[1], p[2], p[3], robot.position[0]])
qvel = np.array([v[3], v[4], v[5], v[0], v[1], v[2], robot.velocity[0]])
return t, qpos, qvel
```

- `t` — sim time in **seconds** (float64), from the robot message's `utime`
  (microseconds). The block's `utime` is ignored.
- Drake serializes a floating body as `position = [qw qx qy qz, x y z]` (quaternion
  first) and `velocity = [wx wy wz, vx vy vz]` (angular first). MuJoCo's free joint is
  the opposite convention: `qpos = [x y z, qw qx qy qz]`, `qvel = [vx vy vz, wx wy wz]`.
  The index shuffle performs exactly that swap, then appends the robot's single
  prismatic DOF:

  | MuJoCo slot | Source | Meaning |
  |---|---|---|
  | `qpos[0:3]` | `p[4:7]` | block world position |
  | `qpos[3:7]` | `p[0:4]` | block quaternion (wxyz in both conventions) |
  | `qpos[7]` | `robot.position[0]` | EE x |
  | `qvel[0:3]` | `v[3:6]` | block linear velocity |
  | `qvel[3:6]` | `v[0:3]` | block angular velocity |
  | `qvel[6]` | `robot.velocity[0]` | EE ẋ |

  These are `np.array` (host, float64) — they cross into JAX at the next
  `pipeline_state.replace`, which casts to float32 device arrays.

### 4.3 `init_mjx_state` (lines 153–158) and `update_mjx_state` (lines 160–167)

```python
def init_mjx_state(self, q, qd, t):
    state = self.env.reset(jax.random.PRNGKey(0))
    pipeline_state = self.pipeline_init_jit(self.env.sys, q, qd)
    obs = self.env._get_obs(pipeline_state, state.info)
    state = state.replace(pipeline_state=pipeline_state, obs=obs)
    return state
```

Called **once**, before the loop. `env.reset` builds a complete, well-formed brax
`State` — sidetrack:

**`Block1DEnv.reset` (`block1d_env.py:63-77`)** initializes a pipeline state from the
keyframe `_init_q` and zero velocity, and — the part that actually matters here —
creates the `info` dict: `{"rng": key, "vel_tar": jnp.array([0.3, 0.0, 0.0]), "step": 0}`.
`State` is a pytree dataclass `(pipeline_state, obs, reward, done, metrics, info)`; the
env's `step` requires `info["vel_tar"]` (reward target) and `info["step"]` (counter), so
`reset` is used purely as a **template factory**. The keyframe qpos it installs is then
immediately discarded: `pipeline_init_jit` (§3.4) rebuilds `pipeline_state` from the LCM
`(q, qd)`, `_get_obs` recomputes the 5-element debug observation
`[block_x, block_vx, ee_x, ee_vx, vel_err_x]` (`block1d_env.py:130-143`), and `replace`
swaps both into the template. The fixed `PRNGKey(0)` is fine — `Block1DEnv.reset` uses
no randomness.

```python
# @partial(jax.jit, static_argnums=(0,))
def update_mjx_state(self, state, q, qd, t):
    pipeline_state = state.pipeline_state.replace(qpos=q, qvel=qd)
    step = int(t / self.ctrl_dt)
    info = state.info
    info["step"] = step
    state = state.replace(pipeline_state=pipeline_state, info=info)
    return state
```

Called **every** loop pass: overwrite `qpos/qvel` in the existing pipeline state with the
fresh LCM snapshot and set `info["step"]` to the wall-sim step index `⌊t/0.02⌋`. Two
details worth noting:

- The `jax.jit` decorator is commented out — necessarily so: `int(t / self.ctrl_dt)`
  forces a concrete Python `int` (a traced value can't be `int()`-ed), and the in-place
  `dict` mutation of `info` is a Python side effect. Eager execution is trivially cheap
  here anyway. Some envs (go2 gait phase) read `info["step"]`; `Block1DEnv` only logs
  it, so for this task the field is inert. **[interpretation]**
- Like `pipeline_init`, the replaced state's cached `x/xd/contact/sensordata` now
  disagree with the new `qpos/qvel` — and again it doesn't matter, because `mjx.step`
  inside each rollout recomputes forward state from `qpos/qvel` before using it.

---

## 5. `main_loop` (lines 169–265) — the control loop, one pass at a time

### 5.1 The scan body: `reverse_scan` (lines 171–174)

```python
def reverse_scan(rng_Y0_state, factor):
    rng, Y0, state = rng_Y0_state
    rng, Y0, info = self.mbdpi.reverse_once(state, rng, Y0, factor)
    return (rng, Y0, state), info
```

A closure shaped for `jax.lax.scan(f, carry, xs)`: `f(carry, x) → (carry, y)`. The
**carry** is `(rng, Y, state)` — note `state` rides along unchanged (the same measured
state is used by every diffusion iteration within one replan). The **xs** are rows of
the `traj_diffuse_factors` matrix (one per-node noise vector per iteration), and the
stacked **ys** are the per-iteration `info` dicts. `lax.scan` compiles the loop once and
runs it entirely on-device — no Python per iteration.

### 5.2 Priming (lines 176–178)

```python
last_plan_time, qpos, qvel = self.wait_for_fresh_state()   # §4.1–4.2
state = self.init_mjx_state(qpos, qvel, last_plan_time)    # §4.3
```

The process blocks here until `block_sim` is up and publishing. From here on, `state` is
the *template* brax `State` whose `pipeline_state.qpos/qvel` get refreshed each pass.

### 5.3 Loop head: fresh state and plan shifting (lines 180–196)

```python
first_time = True
while True:
    t0 = time.time()
    plan_time, qpos, qvel = self.wait_for_fresh_state()
    state = self.update_mjx_state(state, qpos, qvel, plan_time)
    # shift Y
    shift_time = plan_time - last_plan_time
    if shift_time > self.ctrl_dt + 1e-3:
        print(f"[WRAN] sim overtime {(shift_time-self.ctrl_dt)*1000:.1f} ms")
    if shift_time > self.ctrl_dt * self.n_acts:
        print(f"[WARN] long time unplanned {shift_time*1000:.1f} ms, reset control")
        self.Y = self.Y * 0.0
    else:
        self.Y = self.shift_vmap(self.Y, shift_time)
```

`t0` stamps wall-clock for the end-of-pass overtime check. `shift_time` is the **sim
time** that elapsed while the previous pass was planning — i.e. how far into the old
plan the world has already moved.

- If more than one control period (+1 ms tolerance) elapsed, warn `[WRAN]` (sic — typo
  for WARN, both here and at line 265): the sim outran the planner's once-per-tick
  ideal. Purely diagnostic.
- If more than the *entire horizon* elapsed (`0.02 × 16 = 0.32 s` — e.g. after a long
  stall), the old plan says nothing useful about the present; reset `Y` to zeros.
- Otherwise, **shift the plan** so its time axis is re-anchored at `plan_time`:

  **Sidetrack — `shift` (lines 148–151):**

  ```python
  def shift(self, x, shift_time):
      spline = InterpolatedUnivariateSpline(self.mbdpi.step_nodes, x, k=2)
      x_new = spline(self.mbdpi.step_nodes + shift_time)
      return x_new
  ```

  `x` is one column of `Y` — the 6 node values of one actuator over `step_nodes =
  [0, 0.06, …, 0.30]`. Fit a quadratic spline through them and resample at
  `step_nodes + shift_time`: the value that *was* planned for absolute time
  `t_old + s` is now found at plan-relative time `s − shift_time`. Everything the sim
  already executed slides off the front; the far end (`0.30 + shift_time`) lies beyond
  the last node, where the jax_cosmo spline **extrapolates** its last quadratic piece
  (its `ext` is always 0/extrapolate — confirmed in the class docstring).
  **[interpretation]** Quadratic extrapolation of the tail is the standalone's
  replacement for `dial_core.shift`'s roll-and-zero (which assumes exactly one control
  period elapsed); it handles the asynchronous, non-integer shift this deploy loop
  actually experiences — at the price that the last node is a guess that the subsequent
  diffusion update must clean up. `shift_vmap` (§3.2) applies this per column; for
  `nu = 1` there's exactly one column.

### 5.4 First pass only: JIT warm-up + long anneal (lines 198–215)

```python
skip_publish = first_time
n_diffuse = self.dial_config.Ndiffuse
if first_time:
    print("Performing JIT on DIAL-MPC")
    n_diffuse = self.dial_config.Ndiffuse_init
    first_time = False
    traj_diffuse_factors = (
        self.mbdpi.sigma_control[None, :]
        * self.dial_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]
    )
    (self.rng, self.Y, _), info = jax.lax.scan(
        reverse_scan, (self.rng, self.Y, state), traj_diffuse_factors
    )
    n_diffuse = self.dial_config.Ndiffuse
```

On the first pass the planner does a **10-iteration** (`Ndiffuse_init`) annealed
optimization from the all-zero plan — both because the first plan has no warm start to
lean on, and because this call triggers the ~30 s XLA compilation of the whole
scan-of-rollouts graph. `skip_publish` was latched **before** the flag flips, so this
pass's result is never published (§5.7): by the time compilation ends, `plan_time` is
~30 s stale and executing that trajectory would be nonsense.

**The "sigma fix" (comment, lines 204–206).** The noise schedule fed to the scan is a
matrix, one row per diffusion iteration:

```
traj_diffuse_factors[i, h] = sigma_control[h] · traj_diffuse_factor^i
   shape (n_diffuse, Hnode+1);  here (10, 6), then (1, 6) in steady state
```

Row `i` is the per-node noise vector for iteration `i`: the node profile
`[0.59 … 1.0]` (§3.1), damped by `0.5^i` — iteration 0 explores at full σ, iteration 1
at half, etc. (classic annealed/diffusion-style coarse-to-fine). This exactly mirrors
the reference implementation in `dial_core.py:259-261` (`main`), where
`mbdpi.sigma_control * traj_diffuse_factor ** arange(n_diffuse)[:, None]` broadcasts
`(6,)` against `(n_diffuse, 1)`. The comment explains the history: an earlier version of
this standalone passed just `traj_diffuse_factor ** arange(...)` — a flat per-node
noise of ~1.0 on every node — which made `sigma_scale` and `horizon_diffuse_factor`
**inert** (nothing consumed `sigma_control`) and over-perturbed the near-term nodes the
sim is about to execute. The fix restores `dial_core` parity. **(Confirmed:** both
formulas are algebraically identical, `[None, :]` merely makes the broadcast explicit.)

Then the warm-up scan runs (§5.1) and `n_diffuse` is set back to `Ndiffuse` for the code
that follows.

### 5.5 Every pass: the steady-state diffusion update (lines 216–223)

```python
traj_diffuse_factors = (
    self.mbdpi.sigma_control[None, :]
    * self.dial_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]
)
(self.rng, self.Y, _), info = jax.lax.scan(
    reverse_scan, (self.rng, self.Y, state), traj_diffuse_factors
)
```

Identical construction with `n_diffuse = Ndiffuse = 1` (the deploy config trades
iterations for latency — comment in `block1d_deploy.yaml:18-20`). Note that on the very
first pass this runs **in addition to** the 10-iteration warm-up (total 11 iterations),
and because the scan length differs (10 vs 1), XLA compiles **two** scan graphs during
the warm-up pass. Every later pass reuses the length-1 compilation and runs exactly one
`reverse_once`.

The scan's outputs: the carry gives back the advanced `rng` and the improved plan
`self.Y`; `info` is the stacked per-iteration diagnostics — a dict of arrays whose
leading axis is `n_diffuse`. The trailing `info` of the *last* scan is what the
publisher reads (§5.7).

### 5.6 Sidetrack — `MBDPI.reverse_once` (`dial_core.py:103-145`), the heart

`@functools.partial(jax.jit, static_argnums=(0,))` — jitted with `self` static.
Signature: `(state, rng, Ybar_i, noise_scale)` where `Ybar_i` is the current plan `(6,1)`
and `noise_scale` is one row of `traj_diffuse_factors`, shape `(6,)`.

**Step 1 — sample candidate plans (lines 106–115):**

```python
rng, Y0s_rng = jax.random.split(rng)
eps_Y = jax.random.normal(Y0s_rng, (Nsample, Hnode + 1, nu))     # (2048, 6, 1)
Y0s = eps_Y * noise_scale[None, :, None] + Ybar_i                # per-node σ
Y0s = Y0s.at[:, 0].set(Ybar_i[0, :])      # "we can't change the first control"
Y0s = jnp.concatenate([Y0s, Ybar_i[None]], axis=0)               # (2049, 6, 1)
Y0s = jnp.clip(Y0s, -1.0, 1.0)
```

Gaussian perturbations of the current plan, node `h` getting standard deviation
`noise_scale[h]` (this is where `sigma_control` finally bites). Node 0 — the control
being executed *right now* — is pinned to the current plan for every candidate.
The unperturbed `Ybar_i` itself is appended as candidate index `-1` (2049 total), so the
baseline is always evaluated. Everything is clipped to the normalized control box.

**Step 2 — nodes → dense controls → parallel rollouts (lines 117–124):**

```python
us = self.node2u_vvmap(Y0s)                            # (2049, 16, 1)
rewss, pipeline_statess = self.rollout_us_vmap(state, us)
# rewss: (2049, 16)   pipeline_statess.q: (2049, 16, 8)  .qd: (2049, 16, 7)
qss  = pipeline_statess.q
qdss = pipeline_statess.qd
xss  = pipeline_statess.x.pos                          # (2049, 16, nbody-1, 3)
```

Each candidate's 6 nodes become 16 dense controls via the quadratic spline (§3.1), and
all 2049 candidates are rolled out from the **same** measured `state`, 16 env steps
each — 2049 × 16 = 32,784 MJX physics steps per replan, in one fused XLA computation.

**Sidetrack-in-sidetrack — what one env step does (`Block1DEnv.step`,
`block1d_env.py:91-128`):** clip `action·action_scale` to the ctrlrange (this *is*
`act2ctrl`'s math inline), `pipeline_step` = one `mjx.step` at 0.02 s (n_frames=1),
then the reward:

```python
block_linvel = pipeline_state.qvel[0:3]
force = ctrl * self._actuator_gear                     # physical newtons, 6·ctrl
reward_vel  = -‖block_linvel − vel_tar‖²               # vel_tar = [0.3, 0, 0]
reward_ctrl = -‖force‖²
reward = 30.0 · reward_vel + 0.001 · reward_ctrl
```

Higher is better; the weights come straight from the MJPC task this env mirrors. `done`
is always 0 (no termination), and `info["step"]` increments — irrelevant to reward.

**Step 3 — score and reweight (lines 121–128):**

```python
rew_Ybar_i = rewss[-1].mean()                # baseline: appended unperturbed plan
rews = rewss.mean(axis=-1)                   # (2049,) mean reward over horizon
logp0 = (rews - rew_Ybar_i) / rews.std(axis=-1) / temp_sample
weights = jax.nn.softmax(logp0)              # (2049,), sums to 1
```

Each candidate is scored by its **mean reward along the horizon**, centered on the
baseline's score, normalized by the population std (`std(axis=-1)` on a 1-D array is
the scalar std over candidates), and sharpened by `1/temp_sample = 20`. Softmax turns
scores into MPPI importance weights: much-better-than-baseline candidates dominate;
`temp_sample → 0` approaches argmax, large values approach uniform averaging.

**Step 4 — weighted averages (lines 129–143):**

```python
Ybar  = jnp.einsum("n,nij->ij", weights, Y0s)    # (6, 1)   the new plan
qbar  = jnp.einsum("n,nij->ij", weights, qss)    # (16, 8)  "expected" qpos trajectory
qdbar = jnp.einsum("n,nij->ij", weights, qdss)   # (16, 7)  "expected" qvel trajectory
xbar  = jnp.einsum("n,nijk->ijk", weights, xss)
info = {"rews": rews, "qbar": qbar, "qdbar": qdbar, "xbar": xbar,
        "new_noise_scale": new_noise_scale}
return rng, Ybar, info
```

The new plan is the weights-weighted average of the candidate node matrices — the MPPI
update (`softmax_update` computed the same thing one line earlier; line 132 recomputes
it, with the comment "update only with reward"). The same weights also average the
**rollout state trajectories**, giving the planner's state prediction under the new
plan. **[interpretation]** Strictly, "the weighted average of rolled-out states" is not
"the rollout of the weighted-average plan" — nonlinear dynamics don't commute with
averaging — but with a sharp softmax it concentrates on good candidates and is used
only as a tracking reference, not for planning. `new_noise_scale` is just the input
noise vector passed through (MPPI doesn't adapt σ).

Back in `main_loop`, the scan stacks these per iteration:
`info["qbar"]` has shape `(n_diffuse, 16, 8)`, `info["qdbar"]` `(n_diffuse, 16, 7)`,
`info["rews"]` `(n_diffuse, 2049)`.

### 5.7 Plan → executable controls (lines 224–236)

```python
us = self.mbdpi.node2u_vmap(self.Y)          # (16, 1) dense normalized controls
if hasattr(self.env, "act2ctrl"):
    taus = self.env.act2ctrl(us)             # Block1DEnv path
else:
    taus = self.env.act2tau(us, state.pipeline_state)
if skip_publish:
    last_plan_time = plan_time
    continue
```

The freshly updated node plan is densified to the 16 control steps. Then the
**local-patch dispatch** (comment, lines 226–228):

- **`Block1DEnv.act2ctrl` (`block1d_env.py:79-89`)** — the path taken here:
  `clip(act · action_scale, ctrlrange)` per element, i.e. the identity-then-clip map
  from normalized action to MuJoCo `ctrl`. It exists precisely so deploy-side callers
  can bypass the legged-robot assumption below. `taus` is thus `(16, 1)` of values in
  `[-1, 1]` — **normalized ctrl, not torque**, despite the variable name.
- **`BaseEnv.act2tau` (`base_env.py:52-66`)** — the legged default, for envs without
  `act2ctrl`: map action → joint-position target (`act2joint`, an affine map into the
  joint range), then a PD law `τ = kp·(q_target − qpos[7:]) − kd·qvel[6:]`. The `[7:]`
  / `[6:]` slices hard-code "floating base + actuated joints" — for block1d, `qpos[7]`
  is the *EE slider*, and its actuator is a force motor, not a position servo; running
  the PD math here would emit garbage. Hence the `hasattr` dispatch. **(Confirmed** —
  same dispatch exists in `dial_plan.py:231`.)

`skip_publish` (latched in §5.4) makes the JIT-warm-up pass end here: record
`last_plan_time = plan_time` and jump to the next iteration, so the ~30 s-stale plan is
never sent to the sim.

### 5.8 Publishing `INPUT_TRAJ` (lines 237–261)

```python
gear = self.env.sys.mj_model.actuator_gear[0, 0]       # 6.0
q = np.asarray(info["qbar"][-1])                       # (16, 8) last iteration's prediction
v = np.asarray(info["qdbar"][-1])                      # (16, 7)
u_row = np.asarray(taus)[:, 0] * gear                  # (16,) physical force, newtons
knots = np.zeros((6, q.shape[0]))                      # (6, 16)
knots[0] = u_row          # row 0: feedforward force u
knots[1] = q[:, 7]        # row 1: predicted EE x position
knots[5] = v[:, 6]        # row 5: predicted EE x velocity
time_vec = plan_time + np.asarray(self.mbdpi.step_us, dtype=np.float64)
```

- `info["qbar"][-1]` — index `[-1]` selects the **last diffusion iteration** of the
  scan-stacked diagnostics: the state prediction corresponding to the most-refined
  plan. `np.asarray` pulls the arrays off-device to host numpy.
- **Force scaling:** `taus` is normalized ctrl in `[-1,1]`; inside MJX the motor
  applies `gear · ctrl` newtons. The C3-side consumer expects a *physical* feedforward
  force, so the row is scaled by `actuator_gear[0,0] = 6.0` — the planner's
  `ctrl=0.5` becomes `u = 3 N` on the wire. (`mj_model` here is the raw
  `mujoco.MjModel` the brax `System` still carries.)
- The six-row `knots` matrix byte-imitates the layout the native C3 controller
  publishes (its `OutputCombinedTrajectory`): rows `[u, x_ee, 0, 0, 0, v_ee]`. Rows
  2–4 are zero-filled placeholders; the sim-side receiver reads the rows it needs and
  the comment notes "consumer reads rows 0,1,2; Kd = 0 today" (§6 examines the
  consumer). Columns are the 16 knot times.
- **`time_vec` is absolute sim time** — `plan_time + [0, 0.02, …, 0.30]` — so the
  consumer can interpolate "what should be happening at *its* current sim time"
  without any handshake about when the plan was made. The explicit `dtype=np.float64`
  matters: `step_us` is a float32 JAX array, and adding a large absolute timestamp
  (e.g. `plan_time = 1234.56` s) in float32 quantizes to ~0.1 ms steps and would make
  knot spacing irregular; promoting to float64 first keeps microsecond precision
  (comment, line 247).

Then the two-level message assembly (lines 249–261):

```python
traj = lcmt_trajectory_block()
traj.trajectory_name = "position_force_target"
traj.num_points = knots.shape[1]          # 16
traj.num_datatypes = knots.shape[0]       # 6
traj.time_vec = time_vec.tolist()
traj.datapoints = knots.tolist()          # 6 rows × 16 cols, matches wire layout
traj.datatypes = ["double"] * 6

msg = lcmt_timestamped_saved_traj()
msg.utime = int(plan_time * 1e6)          # µs, matches the sim's clock domain
msg.saved_traj.num_trajectories = 1
msg.saved_traj.trajectories = [traj]
msg.saved_traj.trajectory_names = ["position_force_target"]
self.lc.publish(TRAJ_CHANNEL, msg.encode())
```

`encode()` walks the slots and packs the big-endian blob (including the nested
`lcmt_saved_traj` whose default-constructed empty `lcmt_metadata` rides along);
`lc.publish(channel, bytes)` hands it to the multicast socket — fire-and-forget, no
subscriber feedback.

### 5.9 Loop bookkeeping (lines 262–265)

```python
last_plan_time = plan_time
if time.time() - t0 > self.ctrl_dt:
    print(f"[WRAN] real overtime {(time.time()-t0)*1000:.1f} ms")
```

`last_plan_time` becomes the reference for the next pass's `shift_time`. The final check
compares **wall-clock** pass duration against the 20 ms control period — the companion
to the loop-head check, which measured **sim-time** slippage. Then back to
`wait_for_fresh_state`, forever (until Ctrl-C propagates out to `main`'s handler).

---

## 6. The consumer side — what the sim does with `INPUT_TRAJ`

<!-- C3-CONSUMER-SECTION -->

---

## 7. Variable lifecycle reference

| Variable | Type / shape | Born | Meaning & fate |
|---|---|---|---|
| `self.Y` | `jnp` `(Hnode+1, nu)` = `(6, 1)`, float32 | `__init__` (zeros) | The plan: control nodes over the next 0.3 s, normalized `[-1,1]`. Each pass: time-shifted (§5.3) or zeroed, then replaced by the scan's improved carry (§5.5). Never read outside the process. |
| `state` | brax `State` pytree | `init_mjx_state` (§5.2) | Template from `env.reset`; per pass only `pipeline_state.qpos/qvel` and `info["step"]` are refreshed. Broadcast (unbatched) to all 2049 rollouts inside `reverse_once`. |
| `qpos`, `qvel` | `np.float64` `(8,)`, `(7,)` | `get_state_snapshot` | MuJoCo-ordered measured state assembled from two LCM messages (§4.2); written into `state` each pass. |
| `plan_time`, `last_plan_time`, `shift_time` | Python float (s) | loop head | Sim-clock bookkeeping: current snapshot time; previous snapshot time; their difference drives the spline shift and the reset/warn thresholds. |
| `rng` (`self.rng`) | `jnp` uint32 `(2,)` | `__init__` | JAX PRNG key; split once per `reverse_once`, threaded through the scan carry so every pass uses fresh noise. |
| `traj_diffuse_factors` | `jnp` `(n_diffuse, 6)` | §5.4/5.5 | Per-iteration × per-node noise σ: `sigma_control[h]·0.5^i`. The scan's `xs`. |
| `info` | dict of stacked arrays | scan output | `"rews"` `(n_diffuse, 2049)`, `"qbar"` `(n_diffuse, 16, 8)`, `"qdbar"` `(n_diffuse, 16, 7)`, `"xbar"`, `"new_noise_scale"`. Only `qbar[-1][:, 7]` and `qdbar[-1][:, 6]` (last iteration, EE slots) are consumed — as published tracking references. |
| `us` | `jnp` `(16, 1)` | §5.7 | Dense normalized control sequence: quadratic-spline evaluation of `Y` at `step_us`. |
| `taus` | `jnp` `(16, 1)` | §5.7 | `act2ctrl(us)` = clipped normalized ctrl (name is a holdover from the legged `act2tau` path — these are *not* torques here). |
| `knots` | `np.float64` `(6, 16)` | §5.8 | The wire payload: row 0 force (N), row 1 EE x (m), row 5 EE ẋ (m/s), rows 2–4 zero. |
| `time_vec` | `np.float64` `(16,)` | §5.8 | Absolute sim-time stamps `plan_time + step_us` for the 16 knots. |
| `gear` | float, `6.0` | §5.8 | `actuator_gear[0,0]`; converts normalized ctrl → newtons for the wire. |

---

## 8. Confirmed findings, interpretations, and concerns

**Confirmed (read directly from code / verified live):**

1. The sigma-fix formula in this file is algebraically identical to
   `dial_core.py:259-261`; without the `sigma_control` factor, per-node noise shaping
   (`horizon_diffuse_factor`, `sigma_scale`) would be dead config.
2. `handle_timeout(0)` is a non-blocking single-message dispatch returning 0 when the
   queue is empty — the drain loop's exit condition (LCM docstring).
3. Queue capacity 1 drops *later* arrivals, so exactly the oldest in-flight message can
   survive a long planning pass, and the drain discards it before blocking for fresh
   data.
4. `mujoco 3.9` `Data.__dict__` has no `contact` key (it's behind `_impl`), so the
   explicit `contact=` kwarg in `pipeline_init` is required, not redundant; and
   `make_data` leaves `xpos` all-zero, confirming the missing-`mjx.forward` observation
   below.
5. The `(6,16)` `datapoints` orientation matches the LCM type's declared
   `double[num_datatypes][num_points]`.

**Interpretations (reasoned, not stated anywhere):**

6. `qbar/qdbar` as tracking references are a softmax-weighted average of sampled
   rollouts, an approximation of "the state trajectory under the published plan"
   (§5.6, step 4).
7. `reset` is used purely as an info-dict template factory; the keyframe state it
   builds is discarded (§4.3).
8. The spline `shift` with extrapolating tail is the asynchronous generalization of
   `dial_core.shift`'s roll-and-zero (§5.3).

**Concerns (possible issues, none fatal for this task):**

9. **One-`dt` misalignment between published rows.** `rollout_us` collects the state
   *after* each step, so `qbar[k]` is the predicted state at `plan_time+(k+1)·0.02`,
   while `time_vec[k] = plan_time + k·0.02` and `u_row[k]` is the control *applied
   during* `[k·0.02, (k+1)·0.02)`. Rows 1/5 (position/velocity targets) therefore lead
   their timestamps by one control period (20 ms). With feedforward-dominant tracking
   and Kd = 0 this is minor, but a position-feedback consumer would chase a slightly
   future target. (Also, `q[0]` — the prediction one step ahead — is published at
   `time_vec[0] = plan_time`, "now".)
10. **`pipeline_init` omits `mjx.forward`** (vs brax 0.14.1 `init`), leaving
    `x/xd/contact` inconsistent with `(q, qd)` in the initial state. Harmless for
    block1d (nothing reads them before the next `mjx.step`), but a trap for envs whose
    `_get_obs`/reward touch body poses of the initial state (§3.4).
11. **`MBDPI.ctrl_dt` is hard-coded 0.02**, independent of `env_config.dt`; the two
    agree here only because the YAML uses 0.02 (§3.1).
12. **Robot/block snapshot pairing is by arrival, not timestamp** — worst case one sim
    publish period of skew between the two halves of `qpos` (§4.1). `t` comes from the
    robot message only.
13. Dead code inherited from upstream: `sigma0/sigma1/A/B` locals and the broken
    `MBDPI.reverse` (`self.sigmas` never exists); `tqdm` imported unused; `[WRAN]`
    typos in both overtime warnings.
14. First pass runs `Ndiffuse_init + Ndiffuse` = 11 iterations across two scans of
    different lengths, costing two XLA scan compilations during warm-up (§5.4–5.5) —
    intended behavior, but worth knowing when profiling startup.
