
import numpy as np

class Policy(object):
    def __init__(self,
                 env ):
        self.recurrent = None
        self.env = env

    def get_action(self,obs):
        # this function takes a numpy array observations and outputs randomly sampled actions.
        # idx: index corresponding to the task/updated policy.
        # flat_obs = self.env_spec.observation_space.flatten(observation)
        # f_dist = self._cur_f_dist
        mean = np.random.uniform(low=-1.0, high=1.0, size=[1, self.env.action_space.shape[0]])
        # mean, log_std = [x[0] for x in f_dist([flat_obs])]

        action = mean
        return action, dict(mean=mean)

    def reset(self):
        pass