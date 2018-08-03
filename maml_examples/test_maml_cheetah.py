from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.half_cheetah_env_rand import HalfCheetahEnvRand
from rllab.envs.mujoco.half_cheetah_env_oracle import HalfCheetahEnvOracle
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

import joblib
import numpy as np
import pickle
import tensorflow as tf

file1 = 'data/trpo-maml-cheetah200/maml1_fbs10_mbs40_flr_0.1_mlr0.01/itr_75.pkl'
file2 = 'data/s3/bugfix-trpo-maml-cheetah200/randenv/itr_475.pkl'
file3 = 'data/s3/bugfix-trpo-maml-cheetah200/oracleenv/itr_975.pkl'

make_video = False  # generate results if False, run code to make video if True
run_id = 1  # for if you want to run this script in multiple terminals (need to have different ids)

if not make_video:
    test_num_goals = 40
    np.random.seed(1)
    #goals = np.random.uniform(0.1, 0.8, size=(test_num_goals, )) # pre-ICML
    goals = np.random.uniform(0.0, 2.0, size=(test_num_goals, ))  # post-ICML
else:
    np.random.seed(2)
    test_num_goals = 2
    #goals = [0.1, 0.8]
    goals = [0.0, 2.0]
    file_ext = 'mp4'  # can be mp4 or gif
print(goals)

gen_name = 'icml_cheetah_results_'
names = ['maml']# ,'pretrain','random', 'oracle']
exp_names = [gen_name + name for name in names]

step_sizes = [0.1] #, 0.02, 0.1, 0.0]
initial_params_files = [file1, file2, None, file3]

all_avg_returns = []
for step_i, initial_params_file in zip(range(len(step_sizes)), initial_params_files):
    avg_returns = []
    for goal in goals:


        if initial_params_file is not None and 'oracle' in initial_params_file:
            env = normalize(HalfCheetahEnvOracle())
            n_itr = 1
        else:
            env = normalize(HalfCheetahEnvRand())
            n_itr = 800
        env = TfEnv(env)
        policy = MAMLGaussianMLPPolicy(  # random policy
            name='policy',
            env_spec=env.spec,
            hidden_nonlinearity=tf.nn.relu,
            hidden_sizes=(100,100),
        )



        baseline = LinearFeatureBaseline(env_spec=env.spec)
        algo = MAMLTRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=20, # number of trajs for grad update
            max_path_length=20,
            meta_batch_size= 1,
            num_grad_updates=1,
            n_itr=n_itr,
            use_maml=True,
            step_size=step_sizes[step_i],
            plot=False,
            )



        run_experiment_lite(
            algo.meta_online_train(),
            # Number of parallel workers for sampling
            n_parallel=2,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="all",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=2, #1  # don't set the seed for oracle, since it's already deterministic.
            exp_prefix='cheetah_test',
            exp_name='test' + str(run_id),
            plot=True,
        )
        # get return from the experiment
        import csv
        with open('../data/local/cheetah-test/test'+str(run_id)+'/progress.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            i = 0
            row = None
            returns = []
            for row in reader:
                i+=1
                if i ==1:
                    ret_idx = row.index('AverageReturn')
                else:
                    returns.append(float(row[ret_idx]))
            avg_returns.append(returns)

        if make_video:
            data_loc = '../data/local/cheetah-test/test'+str(run_id)+'/'
            save_loc = '../data/local/cheetah-test/test/'
            param_file = initial_params_file
            save_prefix = save_loc + names[step_i] + '_goal_' + str(goal)
            video_filename = save_prefix + 'prestep.' + file_ext
            import os
            os.system('python scripts/sim_policy.py ' + param_file + ' --speedup=4 --max_path_length=500 --video_filename='+video_filename)
            for itr_i in range(3):
                param_file = data_loc + 'itr_' + str(itr_i)  + '.pkl'
                video_filename = save_prefix + 'step_'+str(itr_i)+'.'+file_ext
                os.system('python scripts/sim_policy.py ' + param_file + ' --speedup=4 --max_path_length=500 --video_filename='+video_filename)

    all_avg_returns.append(avg_returns)


    task_avg_returns = []
    for itr in range(len(all_avg_returns[step_i][0])):
        task_avg_returns.append([ret[itr] for ret in all_avg_returns[step_i]])

    if not make_video:
        results = {'task_avg_returns': task_avg_returns}
        with open(exp_names[step_i] + '.pkl', 'wb') as f:
            pickle.dump(results, f)


for i in range(len(initial_params_files)):
    returns = []
    std_returns = []
    returns.append(np.mean([ret[itr] for ret in all_avg_returns[i]]))
    std_returns.append(np.std([ret[itr] for ret in all_avg_returns[i]]))
    print(initial_params_files[i])
    print(returns) #np.mean(all_avg_returns[i]), np.std(all_avg_returns[i])
    print(std_returns) #np.mean(all_avg_returns[i]), np.std(all_avg_returns[i])


