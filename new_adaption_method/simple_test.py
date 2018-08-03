from rllab.envs.mujoco.ant_env_rand_crippled_joints import AntEnvRandDisable
import numpy as np

env = AntEnvRandDisable()
init_state = env.reset(reset_args= 3)
new_obs = init_state

for i in range(8000):
    env.render()
    action = np.random.uniform(-1.0, 1.0, env.action_space.shape[0])

    print(env.model.geom_size)
    #env.reset(init_state=new_obs)
    new_obs, reward, _, _ = env.step(action)
    #print(new_obs[0])
