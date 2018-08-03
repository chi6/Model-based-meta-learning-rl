import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class InvertedDoublePendulumEnv_Rand(MujocoEnv, Serializable):
    FILE = 'inverted_double_pendulum.xml.mako'

    @autoargs.arg("random_start", type=bool,
                  help="Randomized starting position by adjusting the angles"
                       "When this is false, the double pendulum started out"
                       "in balanced position")
    def __init__(
            self, goal_len = None,
            *args, **kwargs):
        self.random_start = kwargs.get("random_start", True)

        self.goal_len = goal_len
        super(InvertedDoublePendulumEnv_Rand, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def sample_goals(self, num_goals):
        return np.random.uniform(0.0, 2.0, (num_goals,))

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        goal_len = reset_args
        if goal_len is not None:
            self._goal_len = goal_len
        elif self._goal_len is None:
            #self._goal_vel = np.random.uniform(0.1, 0.8)
            self._goal_len = np.random.uniform(0.0, 2.0)
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs


    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[:1],  # cart x pos
            np.sin(self.model.data.qpos[1:]),  # link angles
            np.cos(self.model.data.qpos[1:]),
            np.clip(self.model.data.qvel, -10, 10),
            np.clip(self.model.data.qfrc_constraint, -10, 10)
        ]).reshape(-1)

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        x, _, y = self.model.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.model.data.qvel[1:3]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10
        r = float(alive_bonus - dist_penalty - vel_penalty)
        done = y <= 1
        return Step(next_obs, r, done)

    @overrides
    def reset_mujoco(self, init_state=None):
        assert init_state is None
        qpos = np.copy(self.init_qpos)
        if self.random_start:
            qpos[1] = (np.random.rand() - 0.5) * 40 / 180. * np.pi
        self.model.data.qpos = qpos
        self.model.data.qvel = self.init_qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
