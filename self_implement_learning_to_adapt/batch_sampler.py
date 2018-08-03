from rllab.misc import logger
from rllab.sampler.base import BaseSampler
from self_implement_learning_to_adapt import parrallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
from rllab.misc.tensor_utils import flatten_tensors
import tensorflow as tf
import numpy as np
import itertools


def worker_init_tf(G):
    G.sess = tf.Session()
    G.sess.__enter__()


def worker_init_tf_vars(G):
    G.sess.run(tf.global_variables_initializer())


class ParrallelSampler(BaseSampler):
    def __init__(self, algo, n_envs=1):
        super(ParrallelSampler, self).__init__(algo)
        self.n_envs = n_envs

    def start_worker(self):
        if singleton_pool.n_parallel > 1:
            singleton_pool.initialize(singleton_pool.n_parallel)
            singleton_pool.run_each(worker_init_tf)
        parrallel_sampler.populate_task(self.algo.env, self.algo.policy)
        #if singleton_pool.n_parallel > 1:
        #    singleton_pool.run_each(worker_init_tf_vars)
        print('done')

    def shutdown_worker(self):
        parrallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr, init_state = None, reset_args=None, return_dict=False, log_prefix=''):
        init_policy_params = cur_policy_params = self.algo.get_param_values()
        if hasattr(self.algo.env,"get_param_values"):
            try:
                cur_env_params = self.algo.env.get_param_values()
            except:
                cur_env_params = None
        else:
            cur_env_params = None
        import time
        start = time.time()
        if type(reset_args) != list and type(reset_args)!=np.ndarray:
            reset_args = [reset_args]*self.n_envs

        # do tasks sequentially and parallelize within rollouts per task.
        paths = {}
        for i in range(self.n_envs):
            #print("reset: ", reset_args[i])
            paths[i] = parrallel_sampler.sample_paths(
                policy_params=self.algo.cur_params[i],
                env_params=cur_env_params,
                max_samples=self.algo.batch_size / self.n_envs,
                max_path_length=self.algo.max_path_length,
                scope=self.algo.scope,
                reset_arg=reset_args[i],
                show_prog_bar=False,
            )
        total_time = time.time() - start
        logger.record_tabular(log_prefix+"TotalExecTime", total_time)

        if not return_dict:
            flatten_list = lambda l: [item for sublist in l for item in sublist]
            paths = flatten_list(paths.values())

        #self.algo.policy.set_param_values(init_policy_params)

        # currently don't support not whole paths (if desired, add code to truncate paths)
        assert self.algo.whole_paths

        return paths
