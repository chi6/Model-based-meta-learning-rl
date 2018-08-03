
import numpy as np

class Policy(object):
    def __init__(self,
                 env ,
                 params = None,
                 sess = None,
                 network = None):
        self.recurrent = None
        self.env = env
        self.params = params
        self.sess = sess
        self.network = network

    def get_action(self,obs, meta_index, num_samples = 500, batch_size = 5):
        # this function takes a numpy array observations and outputs randomly sampled actions.
        # idx: index corresponding to the task/updated policy.
        #mean = np.random.uniform(low=-1.0, high=1.0, size=[1, self.env.action_space.shape[0]])
        #action = mean
        #return action, dict(mean=mean)
        if self.params == None:
            action = np.random.uniform(-1.0, 1.0, self.env.action_space.shape[0])
            return action, dict(mean = action)

        # multi friction
        adv_list = np.zeros([num_samples])
        old_obs = np.asarray([obs for i in range(num_samples)])
        new_obs = old_obs
        for i in range(batch_size):
            action = (np.random.rand(num_samples, self.env.action_space.shape[0])-0.5)*2
            if i == 0:
                action_list = action
            diff = self.sess.run(self.network.t_final_output, feed_dict={self.network.t_action_inputs: np.asarray(action).reshape([-1,self.env.action_space.shape[0]]),
                                                       self.network.t_state_inputs: np.asarray(new_obs).reshape([-1,self.env.observation_space.shape[0]]),
                                                       self.network.param_inputs: np.asarray(self.params[meta_index]).reshape([-1, 256])})
            new_obs = diff + old_obs
            #angle = np.arccos(old_obs[:,0]/goal)
            #rewards = -((((angle+np.pi) % (2*np.pi)) - np.pi) **2 + old_obs[:,2]**2*0.1 + 0.001* np.sum((action)**2))
            rewards = diff[:,17]/0.01 - 0.05 * np.sum(np.square(action), axis=1)#self.cheetah_cost_fn(old_obs, action, new_obs)
            adv_list[:] += rewards

        index = np.argmax(adv_list)
        return action_list[index], dict(mean = action_list[index])

    def set_param_values(self, params):
        self.params = params

    def reset(self):
        pass