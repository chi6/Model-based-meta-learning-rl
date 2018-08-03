from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.mujoco.half_cheetah_multi_friction import HalfCheetahEnvRandFriction
from rllab.envs.mujoco.half_cheetah_disable_joints import HalfCheetahEnvRandDisableJoints
from rllab.envs.mujoco.ant_env_rand_crippled_joints import AntEnvRandDisable
from rllab.envs.normalized_env import normalize
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from new_adaption_method.policy import Policy
from rllab.misc.instrument import stub, run_experiment_lite
from new_adaption_method.agent import NAT




#stub(globals())

meta_step_size = 0.001
fast_lr = 0.01
meta_batch_size = 40
fast_batch_size = 20
seed = 1
max_path_length = 100
num_grad_updates = 1


# multi velocity goal
#env = TfEnv(normalize(HalfCheetahEnvRand()))

# multi length pendulum
#env = PendulumEnv()
env = TfEnv(normalize(HalfCheetahEnvRandFriction(),normalize_obs=True))
env = env.wrapped_env

baseline = LinearFeatureBaseline(env_spec=env.spec)

policy = Policy(env = env)

agent = NAT(env = env,
             batch_size = fast_batch_size,
             seed = seed,
             max_path_length = max_path_length,
             num_grad_updates = num_grad_updates,
             n_itr = 500,
             step_size=meta_step_size,
             meta_batch_size=meta_batch_size,
             baseline = baseline,
             policy = policy,
             load_policy= False,
             fast_lr = fast_lr
             )

# do multiprocess training
agent.train()
'''run_experiment_lite(
    agent.train(),
    exp_prefix='./trpo_maml_cheetah' + str(max_path_length),
    exp_name='./maml',
    # Number of parallel workers for sampling
    n_parallel=8,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="gap",
    snapshot_gap=25,
    sync_s3_pkl=True,
    python_command='python3',
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=[1],
    mode="local",
    #mode="ec2",
    #variant=v,
    # plot=True,
    # terminate_machine=False,
)
'''


