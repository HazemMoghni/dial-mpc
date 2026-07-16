# dial_lcm_bridge.py: DIAL-MPC controller driving an EXTERNAL Drake sim over LCM.
# Console script: dial-mpc-lcm-bridge (setup.py). Third deploy mode next to the internal
# sync (core/dial_core.py) and async (deploy/dial_plan.py + dial_sim.py) modes, which are
# untouched. Refs are repo-relative path:line; sibling repos prefixed mujoco_mpc/, c3-lcs/.
#
# The controller is the same MBDPI imported from dial_core.py; its Brax/MJX env serves
# ONLY as the rollout model inside reverse_once. The state planned FROM comes over LCM
# (c3-lcs block_sim), and the resulting plan goes back out over LCM:
#   subscribe ROBOT_STATE_SIMULATION (lcmt_robot_output: ee_x pos/vel, 1000 Hz)
#   subscribe BLOCK_STATE_SIMULATION (lcmt_object_state: block pose/twist)
#   publish   INPUT_TRAJ             (lcmt_trajectory_block, one per replan)
#
# The bridge does NOT publish to ROBOT_INPUT. With controller: MJPC in its sim params,
# block_sim wires SplineToRobotCommand (the sim_params.controller branch,
# c3-lcs/.../block1d/block_sim.cc:171-216), which subscribes INPUT_TRAJ, linearly
# interpolates at sim time + 5 ms (spline_time, spline_to_command.cc:155), applies
# u_sol_next = u*Ku + (q_des-q)*Kp + (v_des-v)*Kd (:177; currently Ku=1, Kp=Kd=0 =>
# pure feedforward, test_matrix/c3_options.yaml:61-66), and publishes the single
# resulting force on ROBOT_INPUT itself. Publishing there too would race it.
#
# lcmt_trajectory_block layout consumed by SplineToRobotCommand (u_size=1, q_size=1):
#   num_datatypes = 3; row 0 = force u [N], row 1 = ee pos q, row 2 = ee vel v;
#   time_vec in ABSOLUTE sim seconds (the utime clock block_sim publishes).
#
# Threads mirror mujoco_mpc/.../standalone_block_push.cc (class Handler :55-142 +
# mjpc_control_loop :171-252): an LCM I/O thread stores the latest state under one lock;
# a free-running planner thread ports MBDPublisher.main_loop (dial_plan.py:158).
# MJPC-call map:
#   SetState -> update_mjx_state | OptimizePolicy -> reverse_once scan |
#   BestTrajectory -> node2u + act2ctrl + info[qbar/qdbar]

import os
import sys
import time
import argparse
import importlib
import threading
from dataclasses import dataclass

import yaml
import numpy as np
import art
import emoji

import jax
from jax import numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import lcm

import brax.envs as brax_envs
from brax.base import Motion, System, Transform
from brax.mjx.base import State as MjxState
from brax.mjx.pipeline import _reformat_contact
from mujoco import mjx

import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_core import DialConfig, MBDPI
from dial_mpc.utils.io_utils import load_dataclass_from_dict, get_example_path
from dial_mpc.examples import deploy_examples
from dial_mpc.deploy.lcm_msgs.dairlib import (
    lcmt_robot_output,
    lcmt_robot_input,
    lcmt_object_state,
    lcmt_trajectory_block,
)

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags


def pipeline_init(
    sys: System,
    q: jax.Array,
    qd: jax.Array,
) -> MjxState:
    # copied from dial_plan.py:45-62: builds the full MJX State pytree (x/xd/contact, not
    # just qpos/qvel) that env.step needs structurally. Called once to seed the first
    # rollout; afterwards update_mjx_state just overwrites qpos/qvel.
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


@dataclass
class DialLcmConfig:
    # channel defaults = c3-lcs/.../block1d/parameters/lcm_channels_simulation.yaml
    robot_state_channel: str = "ROBOT_STATE_SIMULATION"
    block_state_channel: str = "BLOCK_STATE_SIMULATION"
    traj_channel: str = "INPUT_TRAJ"
    robot_input_channel: str = "ROBOT_INPUT"  # subscribe-only (logging); never published to
    lcm_url: str = ""  # "" = LCM default URL; must match block_sim's
    state_timeout_s: float = 0.5  # no robot state for this long => warn (sim likely down)
    record: bool = True
    output_dir: str = "output"


# ---------------------------------------------------------------------------
# state translation (pure functions, unit-testable without LCM or JAX)
# ---------------------------------------------------------------------------

def decode_robot_output(msg: lcmt_robot_output):
    """lcmt_robot_output -> (ee_x, ee_x_dot, utime).

    block1d's robot has exactly one position ("ee_x") and one velocity ("ee_x_dot");
    anything else means we are pointed at the wrong sim -- fail loudly rather than
    silently mis-index.
    """
    if msg.num_positions != 1 or msg.num_velocities != 1:
        raise ValueError(
            f"expected 1 position / 1 velocity for the block1d end effector, got "
            f"{msg.num_positions}/{msg.num_velocities} (names: {msg.position_names})"
        )
    return float(msg.position[0]), float(msg.velocity[0]), int(msg.utime)


def decode_object_state(msg: lcmt_object_state):
    """lcmt_object_state -> (pos7, vel6, utime), kept in RAW WIRE ORDER.

    Wire order = Drake's floating-base generalized coordinates (ObjectStateSender copies
    them by index, c3-lcs/.../systems/robot_lcm_systems.cc:490):
      position = [qw qx qy qz x y z]   (quaternion FIRST)
      velocity = [wx wy wz vx vy vz]   (ANGULAR first -- Drake convention)
    The MuJoCo reorder happens in build_qpos_qvel, so this stays a 1:1 mirror of the
    wire (easy to test against a captured message).
    """
    if msg.num_positions != 7 or msg.num_velocities != 6:
        raise ValueError(
            f"expected 7 positions / 6 velocities for the block, got "
            f"{msg.num_positions}/{msg.num_velocities} (names: {msg.position_names})"
        )
    return (
        np.array(msg.position, dtype=np.float64),
        np.array(msg.velocity, dtype=np.float64),
        int(msg.utime),
    )


def build_qpos_qvel(block_pos7, block_vel6, ee_x, ee_x_dot):
    """LCM wire layout -> Block1DEnv layout (block1d.xml header):
        qpos = [block x y z, block quat w x y z, ee_x]
        qvel = [block vx vy vz, wx wy wz, ee_x_dot]
    BOTH halves reorder: wire position is quaternion-first, wire velocity is
    ANGULAR-first (Drake floating base), while MuJoCo's freejoint is position/linear
    first. Same double swap mjpc's Handler does (standalone_block_push.cc:79-95), and
    the same convention the analysis assumes (block forward velocity = velocity[-3],
    c3-lcs/.../block_plots/plot_1d_new.py:303). Getting the velocity halves backwards
    feeds the planner the block's SPIN as its forward velocity -- the reward signal.
    """
    qw, qx, qy, qz, bx, by, bz = block_pos7
    qpos = np.array([bx, by, bz, qw, qx, qy, qz, ee_x], dtype=np.float32)
    qvel = np.concatenate([block_vel6[3:6], block_vel6[0:3], [ee_x_dot]]).astype(np.float32)
    return qpos, qvel


def build_trajectory_block(u, q, v, time_vec, name="dial_mpc_input_traj"):
    """(u, q, v, time_vec) equal-length 1D arrays -> lcmt_trajectory_block in the fixed
    3-row layout SplineToRobotCommand::CalcRobotInput slices (row 0 = u, 1 = q, 2 = v;
    c3-lcs/.../block1d/systems/spline_to_command.cc:129-176)."""
    n = len(time_vec)
    assert len(u) == len(q) == len(v) == n
    msg = lcmt_trajectory_block()
    msg.trajectory_name = name
    msg.num_points = n
    msg.num_datatypes = 3
    msg.time_vec = [float(t) for t in time_vec]
    msg.datapoints = [
        [float(x) for x in u],
        [float(x) for x in q],
        [float(x) for x in v],
    ]
    msg.datatypes = ["u", "q", "v"]
    return msg


# ---------------------------------------------------------------------------
# the bridge
# ---------------------------------------------------------------------------

class MBDLcmBridge:
    def __init__(
        self,
        dial_config: DialConfig,
        env_config,
        lcm_config: DialLcmConfig,
    ):
        self.dial_config = dial_config
        self.env_config = env_config
        self.lcm_config = lcm_config

        print(emoji.emojize(":rocket:") + "Creating environment")
        self.env = brax_envs.get_environment(dial_config.env_name, config=env_config)
        self.mbdpi = MBDPI(dial_config, self.env)
        self.rng = jax.random.PRNGKey(seed=dial_config.seed)
        self.pipeline_init_jit = jax.jit(pipeline_init)
        self.shift_vmap = jax.jit(jax.vmap(self.shift, in_axes=(1, None), out_axes=1))

        self.Y = jnp.zeros([dial_config.Hnode + 1, self.mbdpi.nu])
        self.ctrl_dt = env_config.dt
        # MBDPI hardcodes its own ctrl_dt (core/dial_core.py:74); the shift/spline
        # bookkeeping below mixes the two, so they must agree.
        assert abs(self.ctrl_dt - self.mbdpi.ctrl_dt) < 1e-9, (
            f"env dt ({self.ctrl_dt}) must equal MBDPI's internal ctrl_dt "
            f"({self.mbdpi.ctrl_dt}); set dt/timestep accordingly in the yaml"
        )

        self.n_acts = dial_config.Hsample + 1
        self.nq = self.env.sys.mj_model.nq  # 8: block freejoint (7) + ee slide (1)
        self.nv = self.env.sys.mj_model.nv  # 7
        # ee columns in qpos/qvel (block1d.xml: freejoint first, then the slide joint)
        self.ee_qpos_idx = 7
        self.ee_qvel_idx = 6
        # act2ctrl returns ctrl in ctrlrange [-1,1]; the wire carries Newtons =
        # gear * ctrl. Same scaling mjpc applies before publishing to this sim
        # ("6 * trajectory->actions", standalone_block_push.cc:227-228).
        self.actuator_gear = np.array(self.env.sys.mj_model.actuator_gear[:, 0])

        # latest state received over LCM; the ONE lock in this process. Defaults from
        # the model home keyframe until the first messages arrive.
        self._state_lock = threading.Lock()
        self._latest_ee_pos = float(self.env.sys.mj_model.keyframe("home").qpos[7])
        self._latest_ee_vel = 0.0
        self._latest_ee_utime = None  # None until the first robot state arrives
        home_qpos = self.env.sys.mj_model.keyframe("home").qpos
        self._latest_block_pos = np.array(
            [home_qpos[3], home_qpos[4], home_qpos[5], home_qpos[6],
             home_qpos[0], home_qpos[1], home_qpos[2]]
        )  # wire order [qw qx qy qz x y z]
        self._latest_block_vel = np.zeros(6)
        self._last_robot_msg_wall_time = None

        self._stop_event = threading.Event()

        # record convention from DialSim (self.data.append rows, dial_sim.py:245-255;
        # one .npy on exit)
        self._plan_log = []  # [plan_time, qpos(8), qvel(7), u0_newtons]
        self._applied_log = []  # [utime_s, effort] from ROBOT_INPUT (what the sim applied)

        self._lc = lcm.LCM(lcm_config.lcm_url) if lcm_config.lcm_url else lcm.LCM()
        self._lc.subscribe(lcm_config.robot_state_channel, self._on_robot_state)
        self._lc.subscribe(lcm_config.block_state_channel, self._on_block_state)
        if lcm_config.record:
            self._lc.subscribe(lcm_config.robot_input_channel, self._on_robot_input)

    # copied from dial_plan.py:137-140: re-anchor the knot trajectory to a new time
    # origin after shift_time seconds have elapsed since it was computed
    def shift(self, x, shift_time):
        spline = InterpolatedUnivariateSpline(self.mbdpi.step_nodes, x, k=2)
        x_new = spline(self.mbdpi.step_nodes + shift_time)
        return x_new

    # copied from dial_plan.py:142-148
    def init_mjx_state(self, q, qd, t):
        state = self.env.reset(jax.random.PRNGKey(0))
        pipeline_state = self.pipeline_init_jit(self.env.sys, q, qd)
        obs = self.env._get_obs(pipeline_state, state.info)
        state = state.replace(pipeline_state=pipeline_state, obs=obs)
        return state

    # copied from dial_plan.py:150-156
    def update_mjx_state(self, state, q, qd, t):
        pipeline_state = state.pipeline_state.replace(qpos=q, qvel=qd)
        step = int(t / self.ctrl_dt)
        info = state.info
        info["step"] = step
        state = state.replace(pipeline_state=pipeline_state, info=info)
        return state

    # ---------------- LCM I/O thread ----------------

    def run_io_loop(self):
        while not self._stop_event.is_set():
            self._lc.handle_timeout(100)  # ms; bounded so shutdown is noticed

    def _on_robot_state(self, channel, data):
        msg = lcmt_robot_output.decode(data)
        ee_x, ee_x_dot, utime = decode_robot_output(msg)
        with self._state_lock:
            self._latest_ee_pos = ee_x
            self._latest_ee_vel = ee_x_dot
            self._latest_ee_utime = utime
            self._last_robot_msg_wall_time = time.time()

    def _on_block_state(self, channel, data):
        msg = lcmt_object_state.decode(data)
        pos7, vel6, _ = decode_object_state(msg)
        with self._state_lock:
            self._latest_block_pos = pos7
            self._latest_block_vel = vel6

    def _on_robot_input(self, channel, data):
        # passive log of what block_sim's SplineToRobotCommand actually applied
        msg = lcmt_robot_input.decode(data)
        if msg.num_efforts >= 1:
            self._applied_log.append([1e-6 * msg.utime, float(msg.efforts[0])])

    # ---------------- planner thread ----------------

    def run_planner_loop(self):
        # defined once, outside the loop (as in dial_plan.py:159-162): jax.lax.scan
        # caches traces by function identity; a fresh closure/bound method per
        # iteration would force a retrace+recompile every replan (~100x slower)
        def reverse_scan(rng_Y0_state, factor):
            rng, Y0, state = rng_Y0_state
            rng, Y0, info = self.mbdpi.reverse_once(state, rng, Y0, factor)
            return (rng, Y0, state), info

        # wait for the first robot state so the seed state is real, not the keyframe
        while not self._stop_event.is_set():
            with self._state_lock:
                ready = self._latest_ee_utime is not None
            if ready:
                break
            time.sleep(0.005)
        if self._stop_event.is_set():
            return

        with self._state_lock:
            q0, qd0 = build_qpos_qvel(
                self._latest_block_pos, self._latest_block_vel,
                self._latest_ee_pos, self._latest_ee_vel,
            )
            last_plan_time = 1e-6 * self._latest_ee_utime
        state = self.init_mjx_state(jnp.array(q0), jnp.array(qd0), last_plan_time)

        first_time = True
        while not self._stop_event.is_set():
            t0 = time.time()
            with self._state_lock:
                q, qd = build_qpos_qvel(
                    self._latest_block_pos, self._latest_block_vel,
                    self._latest_ee_pos, self._latest_ee_vel,
                )
                plan_time = 1e-6 * self._latest_ee_utime  # sim clock, not wall clock
                last_msg_wall = self._last_robot_msg_wall_time
            state = self.update_mjx_state(state, jnp.array(q), jnp.array(qd), plan_time)

            # shift the current plan by the sim time elapsed since it was made
            # (the shift_time block of MBDPublisher.main_loop, dial_plan.py:183-193).
            # zeros_like, not Y*0.0: nan*0.0 == nan, so the upstream idiom cannot
            # recover once a plan has gone non-finite.
            shift_time = plan_time - last_plan_time
            if shift_time < 0:
                # sim clock went backwards: sim restarted, or a SECOND sim is publishing
                # on this LCM URL (the default URL is machine-shared -- use the private
                # lcm_url in block1d_lcm.yaml and launch block_sim through
                # analysis/run_block1d_sim.py)
                print(f"[WARN] sim clock regressed {shift_time*1000:.1f} ms "
                      f"(sim restarted, or two sims on this LCM URL?), reset control")
                self.Y = jnp.zeros_like(self.Y)
            elif shift_time > self.ctrl_dt * self.n_acts:
                print(f"[WARN] long time unplanned {shift_time*1000:.1f} ms, reset control")
                self.Y = jnp.zeros_like(self.Y)
            else:
                self.Y = self.shift_vmap(self.Y, shift_time)

            n_diffuse = self.dial_config.Ndiffuse
            is_jit_plan = first_time  # the first plan JIT-compiles (tens of seconds)
            if first_time:
                n_diffuse = self.dial_config.Ndiffuse_init
                print("Performing JIT on DIAL-MPC (first plan)")
            # noise schedule includes mbdpi.sigma_control, as in the sync
            # traj_diffuse_factors (core/dial_core.py:259-261). dial_plan.py's own
            # traj_diffuse_factors (:201,209) omit it, which would silently disable
            # the sigma_scale / horizon_diffuse_factor knobs -- the knobs this
            # bridge exists to sweep.
            traj_diffuse_factors = (
                self.mbdpi.sigma_control
                * self.dial_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]
            )
            (self.rng, self.Y, _), info = jax.lax.scan(
                reverse_scan, (self.rng, self.Y, state), traj_diffuse_factors
            )
            if is_jit_plan:
                # also compile the steady-state scan shape (n_diffuse = Ndiffuse) now,
                # inside the unpublished warmup iteration -- otherwise iteration 2
                # recompiles mid-run (~1 s stall that starves the LCM I/O thread via
                # the GIL and fires a false "sim may be down")
                steady_factors = (
                    self.mbdpi.sigma_control
                    * self.dial_config.traj_diffuse_factor
                    ** (jnp.arange(self.dial_config.Ndiffuse))[:, None]
                )
                (self.rng, self.Y, _), info = jax.lax.scan(
                    reverse_scan, (self.rng, self.Y, state), steady_factors
                )
            first_time = False

            # plan -> per-tick controls -> wire units ("BestTrajectory")
            us = self.mbdpi.node2u_vmap(self.Y)  # [Hsample+1, nu]
            if hasattr(self.env, "act2ctrl"):
                ctrls = self.env.act2ctrl(us)
            else:
                ctrls = self.env.act2tau(us, state.pipeline_state)
            u_newtons = np.asarray(ctrls)[:, 0] * self.actuator_gear[0]
            # info from lax.scan is stacked over annealing iterations; [-1] = last
            # (most refined). qbar/qdbar are the weighted-mean predicted state over
            # the horizon (dial_core.py:133-134); take the ee column of each.
            qbar_ee = np.asarray(info["qbar"][-1][:, self.ee_qpos_idx])
            qdbar_ee = np.asarray(info["qdbar"][-1][:, self.ee_qvel_idx])
            # float64: step_us is float32 JAX and float32 knot times lose precision
            # once plan_time grows (SplineToRobotCommand reads doubles)
            time_vec = plan_time + np.asarray(self.mbdpi.step_us, dtype=np.float64)

            # Do NOT publish the first plan: its ~tens-of-seconds JIT latency leaves
            # time_vec far behind the sim clock, and block_sim's SplineToRobotCommand
            # (FirstOrderHold) EXTRAPOLATES rather than clamps -> a huge force impulse
            # that launches the (frictionless) ee out of range. The sim holds zero force
            # until a valid spline arrives; the shift-reset above re-seeds the next plan,
            # which is fresh (~60 ms latency) and in-range.
            # Also never publish a non-finite plan: if the rollouts diverged (e.g. the
            # ee got kicked out of the model's range), the MPPI softmax turns nan and
            # the sim's tracker would apply nan force ("sending u: nan"). Reset instead;
            # the next iteration replans from scratch off the live state.
            finite = bool(
                np.isfinite(u_newtons).all()
                and np.isfinite(qbar_ee).all()
                and np.isfinite(qdbar_ee).all()
            )
            if not finite:
                print("[WARN] non-finite plan (rollouts diverged), reset control, not publishing")
                self.Y = jnp.zeros_like(self.Y)
            elif not is_jit_plan:
                msg = build_trajectory_block(u_newtons, qbar_ee, qdbar_ee, time_vec)
                self._lc.publish(self.lcm_config.traj_channel, msg.encode())

            if self.lcm_config.record:
                self._plan_log.append(
                    np.concatenate([[plan_time], q, qd, [u_newtons[0]]])
                )

            # watchdog is meaningless during the JIT iteration: compilation holds the
            # GIL for seconds and starves the LCM I/O thread, which looks identical to
            # a dead sim from here
            if not is_jit_plan and last_msg_wall is not None and (
                time.time() - last_msg_wall > self.lcm_config.state_timeout_s
            ):
                print(
                    f"[WARN] no {self.lcm_config.robot_state_channel} for "
                    f"{time.time() - last_msg_wall:.2f}s, sim may be down"
                )

            last_plan_time = plan_time
            if time.time() - t0 > self.ctrl_dt:
                print(f"[WARN] planner overtime {(time.time()-t0)*1000:.1f} ms")

    # ---------------- lifecycle ----------------

    def shutdown(self):
        self._stop_event.set()
        if self.lcm_config.record and self._plan_log:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            out_dir = os.path.join(
                self.lcm_config.output_dir, f"lcm_bridge_block1d_{timestamp}"
            )
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "plans"), np.array(self._plan_log))
            if self._applied_log:
                np.save(os.path.join(out_dir, "applied_inputs"), np.array(self._applied_log))
            print(f"Saved {len(self._plan_log)} plan rows to {out_dir}")


def main(args=None):
    art.tprint("LeCAR @ CMU\nDIAL-MPC\nLCM BRIDGE", font="big", chr_ignore=True)
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, default=None, help="Path to config file")
    group.add_argument("--example", type=str, default=None, help="Example to run")
    group.add_argument("--list-examples", action="store_true", help="List available examples")
    parser.add_argument("--custom-env", type=str, default=None,
                        help="Custom environment to import dynamically")
    # controller knobs (noise / annealing); default None = keep the yaml's value.
    # argparse dest = flag with '-' -> '_' (case kept), so dests equal DialConfig
    # field names with no mapping table.
    for flag, cast in [
        ("--Nsample", int),
        ("--Ndiffuse", int),
        ("--Ndiffuse-init", int),
        ("--temp-sample", float),
        ("--sigma-scale", float),
        ("--horizon-diffuse-factor", float),
        ("--traj-diffuse-factor", float),
    ]:
        parser.add_argument(flag, type=cast, default=None)
    # LCM overrides
    parser.add_argument("--lcm-url", type=str, default=None)
    parser.add_argument("--robot-state-channel", type=str, default=None)
    parser.add_argument("--block-state-channel", type=str, default=None)
    parser.add_argument("--traj-channel", type=str, default=None)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(args)

    if args.custom_env is not None:
        sys.path.append(os.getcwd())
        importlib.import_module(args.custom_env)

    if args.list_examples:
        print("Available examples:")
        for example in deploy_examples:
            print(f"  - {example}")
        return
    if args.example is not None:
        if args.example not in deploy_examples:
            print(f"Example {args.example} not found.")
            return
        config_dict = yaml.safe_load(open(get_example_path(args.example + ".yaml"), "r"))
    else:
        config_dict = yaml.safe_load(open(args.config, "r"))

    # config loading identical to dial_plan.py main(); one yaml feeds all three
    # dataclasses (load_dataclass_from_dict matches keys to fields, io_utils.py:15)
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True
    )
    lcm_config = load_dataclass_from_dict(DialLcmConfig, config_dict)

    for field in (
        "Nsample", "Ndiffuse", "Ndiffuse_init", "temp_sample",
        "sigma_scale", "horizon_diffuse_factor", "traj_diffuse_factor",
    ):
        val = getattr(args, field)
        if val is not None:
            setattr(dial_config, field, val)
    for arg_name, field in (
        ("lcm_url", "lcm_url"),
        ("robot_state_channel", "robot_state_channel"),
        ("block_state_channel", "block_state_channel"),
        ("traj_channel", "traj_channel"),
        ("record", "record"),
    ):
        val = getattr(args, arg_name)
        if val is not None:
            setattr(lcm_config, field, val)

    # persistent compilation cache: sweeps re-launch this process once per trial, and
    # without the cache every trial pays the full ~30 s JIT before publishing anything.
    # With it, warm starts compile-load in a few seconds, so trial durations can stay
    # close to the other controllers' (mjpc harness convention).
    jax.config.update("jax_compilation_cache_dir",
                      os.path.expanduser("~/.cache/jax_dial_mpc"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    bridge = MBDLcmBridge(dial_config, env_config, lcm_config)
    io_thread = threading.Thread(target=bridge.run_io_loop, daemon=True)
    planner_thread = threading.Thread(target=bridge.run_planner_loop, daemon=True)
    io_thread.start()
    planner_thread.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Shutting down bridge...")
    finally:
        bridge.shutdown()
        io_thread.join(timeout=1.0)
        planner_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
