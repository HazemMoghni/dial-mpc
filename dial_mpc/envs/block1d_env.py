# block1d_env.py: DIAL-MPC env for the 1D block-push task. Structure mirrors the shipped
# manipulation env (dial_mpc/envs/manipulation.py); task numbers come from the MJPC block
# task (mujoco_mpc/.../block/task_push.xml + block_push.cc) so DIAL, MJPC, and C3 solve the
# same problem. Refs are repo-relative path:line; sibling repos prefixed mujoco_mpc/, c3-lcs/.

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

import mujoco
from brax import envs as brax_envs
from brax.base import System
from brax.envs.base import State
from brax.io import mjcf

from dial_mpc.envs.base_env import BaseEnv, BaseEnvConfig
from dial_mpc.utils.io_utils import get_model_path


@dataclass
class Block1DEnvConfig(BaseEnvConfig):
    # BaseEnvConfig (dial_mpc/config/base_env_config.py) supplies dt/timestep/action_scale
    # etc.; kp/kd are unused (they feed act2tau, which this env bypasses). Fields below are
    # filled from the yaml top level, like default_vx in unitree_h1_push_crate.yaml.
    vel_target: float = 0.3     # block_vel_des, mujoco_mpc/.../block/block_push.cc:48; = block_vel_target, c3-lcs base_opts.yaml
    w_vel: float = 30.0         # "Block Velocity" user-sensor weight, mujoco_mpc/.../block/task_push.xml:76
    w_ctrl: float = 0.001       # "Actuation" user-sensor weight, mujoco_mpc/.../block/task_push.xml:78
    block_init_x: float = 0.25  # "home" key qpos[0] (task_push.xml:69) = FLAGS_block_x default (standalone_block_push.cc:26)
    ee_init_x: float = -0.04    # "home" key qpos[7] (task_push.xml:69); the testbed sweeps it via FLAGS_ee_x


class Block1DEnv(BaseEnv):
    def __init__(self, config: Block1DEnvConfig):
        # BaseEnv.__init__ (dial_mpc/envs/base_env.py:17-29): builds the MJX system, sets
        # n_frames = dt/timestep, reads jnt_range[1:] (assumes freejoint is joint 0 -- our
        # body order satisfies that) and actuator ctrlrange.
        super().__init__(config)

        # initial qpos from the model "home" keyframe, with the two x entries overridden
        # from config -- same key_qpos[0]/[7] overwrite the testbed does
        # (standalone_block_push.cc:370-371).
        init_q = jnp.array(self.sys.mj_model.keyframe("home").qpos)
        init_q = init_q.at[0].set(config.block_init_x)
        init_q = init_q.at[7].set(config.ee_init_x)
        self._init_q = init_q

        # gear column ([6.0], motor "ee_x_motor" in block1d.xml). force = gear * ctrl; kept so the
        # reward penalizes FORCE, the quantity the MJPC residual reads
        # (data->actuator_force, mujoco_mpc/.../block/block_push.cc:58), not raw ctrl.
        self._actuator_gear = jnp.array(self.sys.mj_model.actuator_gear[:, 0])

    def make_system(self, config: Block1DEnvConfig) -> System:
        # verbatim from AllegroReorientEnv.make_system (manipulation.py:38-43), only the path differs
        model_path = get_model_path("block1d", "block1d.xml")
        mj_model = mujoco.MjModel.from_xml_path(model_path.as_posix())
        sys = mjcf.load_model(mj_model)
        # yaml `timestep` overrides the xml <option timestep> at load
        sys = sys.tree_replace({"opt.timestep": config.timestep})
        return sys

    def reset(self, rng: jax.Array) -> State:
        rng, _ = jax.random.split(rng)  # parity with AllegroReorientEnv.reset (manipulation.py:45); no randomness yet

        # keyframe qpos (with config overrides), zero qvel -- testbed also starts at rest
        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "vel_tar": jnp.array([self._config.vel_target, 0.0, 0.0]),  # block_vel_des, block_push.cc:48
            "step": 0,
        }
        obs = self._get_obs(pipeline_state, state_info)  # logging only; DIAL plans on pipeline_state
        reward, done = jnp.zeros(2)
        metrics = {}
        return State(pipeline_state, obs, reward, done, metrics, state_info)

    @partial(jax.jit, static_argnums=(0,))
    def act2ctrl(self, act: jax.Array) -> jax.Array:
        # direct force mapping, identical to the ctrl computation in step(). Deploy-side
        # callers (MBDPublisher.main_loop's hasattr branch, dial_plan.py:224, and
        # MBDLcmBridge.run_planner_loop) use this instead of the inherited
        # act2joint/act2tau PD machinery, which assumes a legged robot (qpos[7:] actuated).
        return jnp.clip(
            act * self._config.action_scale,
            self.joint_torque_range[:, 0],
            self.joint_torque_range[:, 1],
        )

    def step(self, state: State, action: jax.Array) -> State:
        rng, _ = jax.random.split(state.info["rng"], 2)

        # action -> ctrl in ctrlrange [-1, 1] (joint_torque_range = actuator_ctrlrange,
        # base_env.py:25). Gear 6 makes the applied force +/-6 N, matching the MJPC site
        # actuator (gainprm 6, mujoco_mpc/.../block/end_effector_push.xml:18).
        ctrl = jnp.clip(
            action * self._config.action_scale,
            self.joint_torque_range[:, 0],
            self.joint_torque_range[:, 1],
        )
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        # block world linear velocity = qvel[0:3] (MuJoCo freejoint: linear then angular);
        # same quantity as the "block_vel" framelinvel sensor the residual reads
        # (task_push.xml:83).
        block_linvel = pipeline_state.qvel[0:3]
        force = ctrl * self._actuator_gear  # penalized quantity, see __init__

        # reward = -(w_vel * ||block_vel - target||^2 + w_ctrl * ||force||^2): the only two
        # nonzero-weight terms among the user sensors (task_push.xml:74-78); all three
        # velocity components penalized, matching the dim-3 mju_sub3(block_vel,
        # block_vel_des) residual (block_push.cc:54). Sign: higher = better.
        reward_vel = -jnp.sum(jnp.square(block_linvel - state.info["vel_tar"]))
        reward_ctrl = -jnp.sum(jnp.square(force))
        reward = self._config.w_vel * reward_vel + self._config.w_ctrl * reward_ctrl

        # no termination: testbed runs until Ctrl-C (while running.load(), standalone_block_push.cc:484)
        done = jnp.array(0.0)

        state_info = {  # same keys/shapes as reset (jax pytree requirement)
            "rng": rng,
            "vel_tar": state.info["vel_tar"],
            "step": state.info["step"] + 1,
        }
        obs = self._get_obs(pipeline_state, state_info)
        metrics = {}
        return State(pipeline_state, obs, reward, done, metrics, state_info)

    def _get_obs(self, pipeline_state, state_info) -> jax.Array:
        # [block_x, block_vx, ee_x, ee_vx, vel_err_x] -- logging/debug only (the shipped
        # env even uses a dummy obs, manipulation.py). Index map from block1d.xml:
        #   qpos = [block xyz, quat wxyz, ee_x] -> ee_x = qpos[7]
        #   qvel = [block linvel(3), angvel(3), ee_vx] -> ee_vx = qvel[6]
        return jnp.stack(
            [
                pipeline_state.qpos[0],
                pipeline_state.qvel[0],
                pipeline_state.qpos[7],
                pipeline_state.qvel[6],
                pipeline_state.qvel[0] - state_info["vel_tar"][0],
            ]
        )


# registry entry consumed via env_name: block1d in the yamls; same register_environment
# bottom-line pattern as manipulation.py:117
brax_envs.register_environment("block1d", Block1DEnv)
