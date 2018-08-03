from .mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger


class AntEnvRandDisable(MujocoEnv, Serializable):

    FILE = 'ant.xml'

    def __init__(self, goal=None, *args, **kwargs):
        self._goal_index = goal
        super(AntEnvRandDisable, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    def sample_goals(self, num_goals):
        # for fwd/bwd env, goal direc is backwards if < 1.5, forwards if > 1.5
        return np.random.randint(0, 4, (num_goals, ))

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        _goal_index = reset_args
        if _goal_index is not None:
            self._goal_index = _goal_index
            if self._goal_index == 0:
                idx = self.model.geom_names.index("left_ankle_geom")
            elif self._goal_index == 1:
                idx = self.model.geom_names.index("right_ankle_geom")
            elif self._goal_index == 2:
                idx = self.model.geom_names.index("third_ankle_geom")
            elif self._goal_index == 3:
                idx = self.model.geom_names.index("fourth_ankle_geom")
            size = np.array(self.model.geom_size)
            size[idx] = np.array([0.05, 0.1, 0.0])
            self.model.geom_size = size


        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs


    def step(self, action):
        action = np.reshape(action, [1, -1])
        #print(action.shape)
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular(prefix+'AverageForwardProgress', np.mean(progs))
        logger.record_tabular(prefix+'MaxForwardProgress', np.max(progs))
        logger.record_tabular(prefix+'MinForwardProgress', np.min(progs))
        logger.record_tabular(prefix+'StdForwardProgress', np.std(progs))

