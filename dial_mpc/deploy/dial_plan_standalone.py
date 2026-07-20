# Terminal 1
# cd ~/Repos/c3-lcs && ./bazel-bin/examples/c3-import/block1d/block_sim
# Terminal 2
# PYTHONPATH="$HOME/Repos/c3-lcs/analysis:$PYTHONPATH" \
# python3 -m dial_mpc.deploy.dial_plan_standalone --example block1d_deploy
# Viz:
# cd ~/Repos/c3-lcs && ./bazel-bin/examples/c3-import/block1d/block_visualizer

import os
from dataclasses import dataclass
import time
import lcm
import importlib
import sys

import yaml
import argparse
import numpy as np
from tqdm import tqdm
import art
import emoji

import functools
from functools import partial
import jax
from jax import numpy as jnp
from mujoco import mjx

import brax.envs as brax_envs
from brax.envs.base import Env as BraxEnv
from brax.envs.base import State
from brax.mjx.base import State as MjxState
from brax.mjx.pipeline import _reformat_contact
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from brax.base import Contact, Motion, System, Transform

import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_core import DialConfig, MBDPI
from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
from dial_mpc.utils.io_utils import (
    load_dataclass_from_dict,
    get_model_path,
    get_example_path,
)
from dial_mpc.examples import deploy_examples

# generated lcm bindings; run with PYTHONPATH="$HOME/Repos/c3-lcs/analysis:$PYTHONPATH"
from dairlib import (
    lcmt_robot_output,
    lcmt_object_state,
    lcmt_trajectory_block,
)

# values from examples/c3-import/block1d/parameters/lcm_channels_simulation.yaml
ROBOT_STATE_CHANNEL = "ROBOT_STATE_SIMULATION"
BLOCK_STATE_CHANNEL = "BLOCK_STATE_SIMULATION"
TRAJ_CHANNEL = "INPUT_TRAJ"


# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags


def pipeline_init(
    sys: System,
    q: jax.Array,
    qd: jax.Array,
) -> MjxState:
    data = mjx.make_data(sys)
    data = data.replace(qpos=q, qvel=qd)

    q, qd = data.qpos, data.qvel
    x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
    cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
    offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
    offset = Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)

    # local patch vs upstream dial-mpc: _reformat_contact takes (sys, contact) and
    # returns a Contact in current brax; upstream passed/kept the whole `data`
    contact = _reformat_contact(sys, data.contact)
    return MjxState(q=q, qd=qd, x=x, xd=xd, contact=contact, **data.__dict__)

class MBDPublisher_standalone:
    def __init__(
        self, env: BaseEnv, env_config: BaseEnvConfig, dial_config: DialConfig,
        sim_q_init=None,
    ):
        # trial's true initial qpos (mujoco order); when set, the JIT warm-up
        # doubles as the first real plan, published at t=0 before the sim runs
        self.sim_q_init = sim_q_init
        # MBD related
        # setup MBDPI controller
        self.dial_config = dial_config
        self.env = env
        self.env_config = env_config

        self.mbdpi = MBDPI(self.dial_config, self.env)
        self.rng = jax.random.PRNGKey(seed=self.dial_config.seed)
        self.pipeline_init_jit = jax.jit(pipeline_init)
        self.shift_vmap = jax.jit(jax.vmap(self.shift, in_axes=(1, None), out_axes=1))

        # control parameters
        self.Y = jnp.zeros([self.dial_config.Hnode + 1, self.mbdpi.nu])
        self.ctrl_dt = env_config.dt
        self.n_acts = self.dial_config.Hsample + 1

        # lcm setup, replacing the shared-memory transport
        self.lc = lcm.LCM()
        self.latest_robot_state = None
        self.latest_block_state = None
        self.received_robot = False
        self.received_block = False
        sub_r = self.lc.subscribe(ROBOT_STATE_CHANNEL, self.handle_robot_state)
        sub_b = self.lc.subscribe(BLOCK_STATE_CHANNEL, self.handle_block_state)
        # capacity 1: at most one stale message survives a planning pass
        sub_r.set_queue_capacity(1)
        sub_b.set_queue_capacity(1)

    def handle_robot_state(self, channel, data):
        self.latest_robot_state = lcmt_robot_output.decode(data)
        self.received_robot = True

    def handle_block_state(self, channel, data):
        self.latest_block_state = lcmt_object_state.decode(data)
        self.received_block = True

    def wait_for_fresh_state(self):
        # discard anything queued while planning, then block for a new pair
        while self.lc.handle_timeout(0) > 0:
            pass
        self.received_robot = False
        self.received_block = False
        while not (self.received_robot and self.received_block):
            self.lc.handle()
        return self.get_state_snapshot()

    def get_state_snapshot(self):
        robot = self.latest_robot_state
        block = self.latest_block_state
        t = robot.utime * 1e-6
        p = block.position
        v = block.velocity
        # wire is quat-first / angular-first (drake); mujoco is pos-first / linear-first
        qpos = np.array([p[4], p[5], p[6], p[0], p[1], p[2], p[3], robot.position[0]],
                        dtype=np.float32)
        qvel = np.array([v[3], v[4], v[5], v[0], v[1], v[2], robot.velocity[0]],
                        dtype=np.float32)
        return t, qpos, qvel

    def shift(self, x, shift_time):
        spline = InterpolatedUnivariateSpline(self.mbdpi.step_nodes, x, k=2)
        x_new = spline(self.mbdpi.step_nodes + shift_time)
        return x_new

    def publish_plan(self, taus, info, plan_time):
        # MJPC-branch "points" input: bare lcmt_trajectory_block with
        # rows [u, x_ee, v_ee] (SplineToRobotCommand reads rows 0,1,2)
        gear = self.env.sys.mj_model.actuator_gear[0, 0]
        q = np.asarray(info["qbar"][-1])
        v = np.asarray(info["qdbar"][-1])
        knots = np.vstack([
            np.asarray(taus)[:, 0] * gear,
            q[:, 7],
            v[:, 6],
        ])
        # absolute sim seconds; float64 since step_us is float32
        time_vec = plan_time + np.asarray(self.mbdpi.step_us, dtype=np.float64)
        traj = lcmt_trajectory_block()
        traj.trajectory_name = "position_force_target"
        traj.num_points = knots.shape[1]
        traj.num_datatypes = knots.shape[0]
        traj.time_vec = time_vec.tolist()
        traj.datapoints = knots.tolist()
        traj.datatypes = ["double"] * knots.shape[0]
        self.lc.publish(TRAJ_CHANNEL, traj.encode())

    def init_mjx_state(self, q, qd, t):
        state = self.env.reset(jax.random.PRNGKey(0))
        pipeline_state = self.pipeline_init_jit(self.env.sys, q, qd)
        obs = self.env._get_obs(pipeline_state, state.info)
        state = state.replace(pipeline_state=pipeline_state, obs=obs)
        return state

    # @partial(jax.jit, static_argnums=(0,))
    def update_mjx_state(self, state, q, qd, t):
        pipeline_state = state.pipeline_state.replace(qpos=q, qvel=qd)
        step = int(t / self.ctrl_dt)
        info = state.info
        info["step"] = step
        state = state.replace(pipeline_state=pipeline_state, info=info)
        return state

    def main_loop(self):

        def reverse_scan(rng_Y0_state, factor):
            rng, Y0, state = rng_Y0_state
            rng, Y0, info = self.mbdpi.reverse_once(state, rng, Y0, factor)
            return (rng, Y0, state), info

        # JIT warmup before the sim starts, so the experiment never waits on
        # compilation. With sim_q_init it runs on the trial's true initial state
        # and its solution is the first real plan; otherwise on the home keyframe.
        print("Performing JIT on DIAL-MPC")
        if self.sim_q_init is not None:
            qpos0 = self.sim_q_init
        else:
            qpos0 = np.array(self.env.sys.mj_model.keyframe("home").qpos, dtype=np.float32)
        qvel0 = np.zeros(self.env.sys.mj_model.nv, dtype=np.float32)
        state = self.init_mjx_state(jnp.array(qpos0), jnp.array(qvel0), 0.0)
        # numpy qpos/qvel like every loop iteration, else the scan compiles a second
        # executable for the loop's input sharding (~4 s stall on the first real state)
        state = self.update_mjx_state(state, qpos0, qvel0, 0.0)
        for n_diffuse in (self.dial_config.Ndiffuse_init, self.dial_config.Ndiffuse):
            traj_diffuse_factors = (
                self.dial_config.traj_diffuse_factor ** (jnp.arange(n_diffuse))[:, None]
            )
            (self.rng, self.Y, _), info = jax.lax.scan(
                reverse_scan, (self.rng, self.Y, state), traj_diffuse_factors
            )
        self.Y = self.shift_vmap(self.Y, 0.0)
        us = self.mbdpi.node2u_vmap(self.Y)
        if hasattr(self.env, "act2ctrl"):
            taus = self.env.act2ctrl(us)
        else:
            taus = self.env.act2tau(us, state.pipeline_state)
        print("JIT done, waiting for sim")

        if self.sim_q_init is not None:
            # publish the t=0 plan until the sim consumes it and states flow;
            # the sim's subscriber may not exist yet, so keep republishing
            while not (self.received_robot and self.received_block):
                self.publish_plan(taus, info, 0.0)
                self.lc.handle_timeout(50)
            last_plan_time = 0.0
        else:
            # first "input" state for the planner
            last_plan_time, qpos, qvel = self.wait_for_fresh_state()
            state = self.update_mjx_state(state, qpos, qvel, last_plan_time)

        while True:
            t0 = time.time()
            # get state; the "input" state for the planner
            plan_time, qpos, qvel = self.wait_for_fresh_state()
            state = self.update_mjx_state(state, qpos, qvel, plan_time)
            # shift Y
            shift_time = plan_time - last_plan_time
            if shift_time > self.ctrl_dt + 1e-3:
                print(f"[WRAN] sim overtime {(shift_time-self.ctrl_dt)*1000:.1f} ms")
            if shift_time > self.ctrl_dt * self.n_acts:
                print(
                    f"[WARN] long time unplanned {shift_time*1000:.1f} ms, reset control"
                )
                self.Y = self.Y * 0.0
            else:
                self.Y = self.shift_vmap(self.Y, shift_time)
            # run planner
            traj_diffuse_factors = (
                self.dial_config.traj_diffuse_factor
                ** (jnp.arange(self.dial_config.Ndiffuse))[:, None]
            )
            (self.rng, self.Y, _), info = jax.lax.scan(
                reverse_scan, (self.rng, self.Y, state), traj_diffuse_factors
            )
            # convert plan to control
            us = self.mbdpi.node2u_vmap(self.Y)
            # local patch vs upstream: direct-ctrl envs (Block1DEnv.act2ctrl,
            # envs/block1d_env.py) bypass act2tau, whose PD math assumes a legged
            # robot with qpos[7:] actuated. Legged envs are unaffected (no act2ctrl).
            if hasattr(self.env, "act2ctrl"):
                taus = self.env.act2ctrl(us)
            else:
                taus = self.env.act2tau(us, state.pipeline_state)
            self.publish_plan(taus, info, plan_time)
            # record time
            last_plan_time = plan_time
            if time.time() - t0 > self.ctrl_dt:
                print(f"[WRAN] real overtime {(time.time()-t0)*1000:.1f} ms")


# DialConfig fields exposed as CLI overrides for experiment sweeps
DIAL_OVERRIDE_FIELDS = [
    ("seed", int),
    ("Nsample", int),
    ("Hsample", int),
    ("Hnode", int),
    ("Ndiffuse", int),
    ("Ndiffuse_init", int),
    ("temp_sample", float),
    ("traj_diffuse_factor", float),
]


def main(args=None):
    art.tprint("LeCAR @ CMU\nDIAL-MPC\nSTANDALONE PLANNER", font="big", chr_ignore=True)
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    group.add_argument(
        "--example",
        type=str,
        default=None,
        help="Example to run",
    )
    group.add_argument(
        "--list-examples",
        action="store_true",
        help="List available examples",
    )
    parser.add_argument(
        "--custom-env",
        type=str,
        default=None,
        help="Custom environment to import dynamically",
    )
    parser.add_argument(
        "--sim-config",
        type=str,
        default=None,
        help="Drake sim params yaml; plan for its q_init at t=0 instead of "
        "warming up on the home keyframe",
    )
    # experiment-sweep overrides; when omitted the config file value is used
    for flag, cast in DIAL_OVERRIDE_FIELDS:
        parser.add_argument(
            f"--{flag}",
            type=cast,
            default=None,
            help=f"Override {flag} from the config file",
        )
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
        config_dict = yaml.safe_load(
            open(get_example_path(args.example + ".yaml"), "r")
        )
    else:
        config_dict = yaml.safe_load(open(args.config, "r"))

    for flag, _ in DIAL_OVERRIDE_FIELDS:
        value = getattr(args, flag)
        if value is not None:
            config_dict[flag] = value

    print(emoji.emojize(":rocket:") + "Creating environment")
    dial_config = load_dataclass_from_dict(DialConfig, config_dict)
    env_config_type = dial_envs.get_config(dial_config.env_name)
    env_config = load_dataclass_from_dict(
        env_config_type, config_dict, convert_list_to_array=True
    )
    env = brax_envs.get_environment(dial_config.env_name, config=env_config)

    sim_q_init = None
    if args.sim_config is not None:
        sim_cfg = yaml.safe_load(open(args.sim_config, "r"))
        qb = sim_cfg["q_init_block"]
        # drake yaml is quat-first; mujoco qpos is [block xyz, quat wxyz, ee_x]
        # (same reordering as get_state_snapshot)
        sim_q_init = np.array(
            [qb[4], qb[5], qb[6], qb[0], qb[1], qb[2], qb[3],
             sim_cfg["q_init"][0]],
            dtype=np.float32,
        )

    mbd_publisher = MBDPublisher_standalone(env, env_config, dial_config,
                                            sim_q_init=sim_q_init)

    try:
        mbd_publisher.main_loop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
