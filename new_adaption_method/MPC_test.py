import tensorflow as tf
from new_adaption_method.agent import NAT
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.mujoco.half_cheetah_disable_joints import HalfCheetahEnvRandDisableJoints
from rllab.envs.mujoco.half_cheetah_multi_friction import HalfCheetahEnvRandFriction
from rllab.envs.mujoco.ant_env_rand import AntEnvRand
#from rllab.envs.mujoco.half_cheetah_env_rand import HalfCheetahEnvRand
from rllab.envs.mujoco.ant_env_rand_crippled_joints import AntEnvRandDisable
from rllab.envs.box2d.double_pendulum_env_rand import DoublePendulumEnvRand
from rllab.envs.normalized_env import normalize
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from new_adaption_method.policy import Policy
from self_implement_maml.pendulum import PendulumEnv
#from rllab.misc.instrument import stub, run_experiment_lite
import numpy as np

def train():
    #stub(globals())
    learning_rate = 0.001
    meta_step_size = 0.01
    meta_batch_size = 1
    fast_batch_size = 10
    seed = 2
    max_path_length = 1000
    num_grad_updates = 1
    num_samples = 1000
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # multi velocity goal
    #env = TfEnv(normalize(HalfCheetahEnvRand()))

    # multi length pendulum
    env1 = TfEnv(normalize(HalfCheetahEnvRandFriction(),normalize_obs= True))
    env2 = None#TfEnv(normalize(HalfCheetahEnvRand()))

    baseline = LinearFeatureBaseline(env_spec=env1.spec)

    policy = Policy(env = env1)

    agent = NAT(env = env1,
                 fake_env= env2,
                 batch_size = fast_batch_size,
                 seed = seed,
                 max_path_length = max_path_length,
                 num_grad_updates = num_grad_updates,
                 n_itr = 145,
                 num_samples= num_samples,
                 step_size=meta_step_size,
                 meta_batch_size=meta_batch_size,
                 baseline = baseline,
                 policy = policy,
                 load_policy= True,
                 save_video= False,
                 lr= learning_rate
                 )

    # do multiprocess training
    agent.meta_online_train(goal= 0.9)

if __name__ == '__main__':
    train()