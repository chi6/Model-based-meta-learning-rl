import tensorflow as tf
import numpy as np
from new_adaption_method.model import LSTM_network
from new_adaption_method.batch_sampler import ParrallelSampler
from new_adaption_method.vectorized_sampler import VectorizedSampler
from rllab.misc import ext
import matplotlib.pyplot as plt
import scipy.signal as signal
from rllab.sampler.stateful_pool import singleton_pool

class NAT(object):
    def __init__(self,
                 step_size,
                 env,
                 batch_size,
                 meta_batch_size,
                 seed,
                 n_itr,
                 max_path_length,
                 num_grad_updates,
                 baseline,
                 policy,
                 num_samples = 1000,
                 scope = None,
                 sess = None,
                 center_adv=True,
                 positive_adv=False,
                 store_paths=False,
                 whole_paths=True,
                 fixed_horizon=False,
                 load_policy = False,
                 fake_env = None,
                 save_video = False,
                 fast_lr = 0.1,
                 lr = 0.001,
                 discount = 0.99,
                 gae_lambda = 1,
                 ):
        self.step_size = step_size
        self.env = env
        self.fake_env = fake_env
        self.batch_size = batch_size
        self.meta_batch_size = meta_batch_size
        self.seed = seed
        self.n_itr = n_itr
        self.max_path_length = max_path_length
        self.num_grad_updates = num_grad_updates
        self.discount = discount
        self.baseline = baseline
        self.gae_lambda = gae_lambda
        self.policy = policy
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.load_policy = load_policy
        self.save_video = save_video
        self.scope = scope
        self.num_samples = num_samples
        self.s_size = self.env.observation_space.shape[0]
        self.a_size = self.env.action_space.shape[0]
        print(self.s_size,  self.a_size)

        self.lr = lr
        self.fast_lr = fast_lr
        self.loss_list = []
        self.reward_list = []
        self.task_params = [np.zeros([1, 256],dtype= np.float32) for i in range(self.meta_batch_size)]
        self.fig = None

        # select sampler
        if singleton_pool.n_parallel >1:
            self.sampler =  ParrallelSampler(self, n_envs= self.meta_batch_size)
        else:
            self.sampler = VectorizedSampler(self, n_envs= self.meta_batch_size)


        # define trainer
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.lr)
        # construct input tensors
        self.policy.network = LSTM_network(s_size= self.s_size, a_size= self.a_size, scope = 'NAT_network', trainer= self.trainer, num_hidden= 256)
        self.params = tf.global_variables(scope = 'NAT_network')
        self.cur_params = [self.params for i in range(self.meta_batch_size)]
        self.saver = tf.train.Saver()

    def obtain_samples(self, itr, init_state, reset_args ):
        paths = self.sampler.obtain_samples(itr,init_state = init_state,reset_args= reset_args, return_dict= True)
        return paths

    def process_samples(self, itr, path):
        return self.sampler.process_samples(itr, path, log = False)

    def update_target_graph(self, params, to_scope):
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(params, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder


    def MPC(self,itr, num_samples, init_state, goal, task_param):

        '''
        # disable multiple joints
        adv_list = np.zeros([num_samples])
        old_obs = np.asarray([init_state for i in range(num_samples)])
        new_obs = old_obs
        for i in range(self.batch_size):
            action = (np.random.rand(num_samples, self.a_size)-0.5)*2
            action[:, goal] = 0.0
            if i == 0:
                action_list = action
            diff = self.sess.run(self.first_outputs, feed_dict={self.f_state_inputs: np.asarray(new_obs).reshape([-1,self.s_size]),
                                                            self.f_action_inputs: np.asarray(action).reshape([-1,self.a_size])})
            new_obs = diff + old_obs
            rewards = diff[:,17]/0.01 - 0.05 * np.sum(np.square(action),axis=1)
            adv_list[:] += rewards

        index = np.argmax(adv_list)
        return action_list[index]
        '''

        # multi friction
        adv_list = np.zeros([num_samples])
        old_obs = np.asarray([init_state for i in range(num_samples)])
        new_obs = old_obs
        for i in range(self.batch_size):
            action = (np.random.rand(num_samples, self.a_size)-0.5)*2
            if i == 0:
                action_list = action
            diff = self.sess.run(self.policy.network.t_final_output, feed_dict={self.policy.network.t_action_inputs: np.asarray(action).reshape([-1,self.a_size]),
                                                       self.policy.network.t_state_inputs: np.asarray(new_obs).reshape([-1,self.s_size]),
                                                       self.policy.network.param_inputs: np.asarray(task_param).reshape([-1, 256])})
            new_obs = diff + old_obs
            #angle = np.arccos(old_obs[:,0]/goal)
            #rewards = -((((angle+np.pi) % (2*np.pi)) - np.pi) **2 + old_obs[:,2]**2*0.1 + 0.001* np.sum((action)**2))
            rewards = diff[:,17]/0.01 - 0.05 * np.sum(np.square(action), axis=1)#self.cheetah_cost_fn(old_obs, action, new_obs)
            adv_list[:] += rewards

        index = np.argmax(adv_list)
        return action_list[index]


    def meta_online_train(self, goal):
        '''
        meta online adaption: load prior meta model, select action by doing MPC, adapt model in each step
        :param goal: sample task
        :return:
        '''
        self.goal = goal
        self.sess = tf.Session()
        with self.sess as sess:

            self.summary_writer = tf.summary.FileWriter("./graph/", self.sess.graph)

            loss_plot = None
            loss_summary = tf.Summary()
            loss_summary.value.add(tag='loss', simple_value=loss_plot)
            reward_plot = None
            reward_summary = tf.Summary()
            reward_summary.value.add(tag = 'reward', simple_value = reward_plot)
            diff_plot = None
            diff_summary = tf.Summary()
            diff_summary.value.add(tag='state_difference', simple_value=diff_plot)


            if self.load_policy:
                sess.run(tf.global_variables_initializer())
                self.c_in = np.load('c_state.npy')
                self.saver = tf.train.import_meta_graph('./half_cheetah_model/maml_model.ckpt-136.meta')
                self.saver.restore(sess, tf.train.latest_checkpoint('./half_cheetah_model/'))
                self.sampler.start_worker()
            else:
                sess.run(tf.global_variables_initializer())
                self.sampler.start_worker()

            self.env = self.env.wrapped_env
            self.env.reset(reset_args=goal) # set the goal for env

            nstep = 0
            for itr in range(self.n_itr):
                rewards = []
                obs, act, diffs, images = [], [], [], []
                new_state = self.env.reset()
                #action = np.random.uniform(-1.0, 1.0, self.a_size)
                self.policy.params = self.c_in#sess.run(self.policy.network.c_state_out, feed_dict= {self.policy.network.l_action_inputs: np.asarray([action]),
                                     #                                     self.policy.network.l_state_inputs: np.asarray([new_state])})
                for step in range(self.max_path_length):
                    #if step>int(self.max_path_length)*0.9:
                    #    self.env.render()
                    if len(act) > 0 :#and step%10 == 0:
                       # indices = np.random.randint(0, len(act), len(act))
                        _ = sess.run([ self.policy.network.apply_training],
                                            feed_dict={self.policy.network.g_action_inputs: np.asarray(act),
                                                       self.policy.network.l_action_inputs: np.asarray(act),
                                                       self.policy.network.g_state_inputs: np.asarray(obs),
                                                       self.policy.network.l_state_inputs: np.asarray(obs),
                                                       self.policy.network.goal: np.asarray(diffs),
                                                       self.policy.network.c_in: self.c_in})

                        loss, self.policy.params = sess.run([self.policy.network.loss, self.policy.network.c_state_out], feed_dict={self.policy.network.g_action_inputs: np.asarray(act),
                                                       self.policy.network.l_action_inputs: np.asarray(act),
                                                       self.policy.network.g_state_inputs: np.asarray(obs),
                                                       self.policy.network.l_state_inputs: np.asarray(obs),
                                                       self.policy.network.goal: np.asarray(diffs),
                                                       self.policy.network.c_in: self.c_in})
                        self.c_in = self.policy.params
                        #diff = np.mean(abs(np.asarray(obs[1:-1])-np.asarray(obs[0:-2]) - output[0:-2]))
                        #diff_summary.value[0].simple_value = diff
                        loss_summary.value[0].simple_value = loss
                        self.summary_writer.add_summary(loss_summary, nstep)
                        self.summary_writer.add_summary(diff_summary, nstep)

                        self.sess.run(self.policy.network.op_holder)

                    obs.append(new_state)
                    if step%100 == 0:
                        print("Doing MPC, step:", step)

                    action = self.MPC(itr = itr, num_samples= self.num_samples, goal= goal, init_state= new_state, task_param = self.policy.params)
                    new_obs, reward, done,_= self.env.step(action)
                    act.append(action)
                    diffs.append(new_obs - new_state)
                    rewards.append(reward)

                    nstep +=1
                    new_state = new_obs

                    if done:
                        break
                    if self.save_video:
                        from PIL import Image
                        self.env.render()
                        image = self.env.wrapped_env.get_viewer().get_image()
                        pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
                        images.append(np.flipud(np.array(pil_image)))


                if self.save_video :
                    import moviepy.editor as mpy
                    clip = mpy.ImageSequenceClip(images, fps=20 * 1)
                    clip.write_videofile("./video/half_cheetah.mp4", fps=20 * 1)
                self.saver.save(sess, './MPC_model/mpc_model.ckpt',global_step= itr)
                if itr >= 0:
                    sum_rewards = np.sum(np.asarray(rewards))
                    print(sum_rewards)

                    reward_summary.value[0].simple_value = sum_rewards
                    self.summary_writer.add_summary(reward_summary, itr)



    def train(self):
        '''
        training of transition model : sample trajectories based on different tasks, doing optimization
        :return:
        '''
        self.sess = tf.Session()
        self.policy.sess = self.sess
        with self.sess as sess:

            self.summary_writer = tf.summary.FileWriter("./graph/", self.sess.graph)
            if self.load_policy:
                sess.run(tf.global_variables_initializer())
                self.saver = tf.train.import_meta_graph('./half_cheetah_model/maml_model.ckpt-3.meta')
                self.saver.restore(sess, tf.train.latest_checkpoint('./half_cheetah_model/'))
                self.sampler.start_worker()
            else:
                sess.run(tf.global_variables_initializer())
                self.sampler.start_worker()
            self.env = self.env.wrapped_env
            loss_plot = None
            loss_summary = tf.Summary()
            loss_summary.value.add(tag='loss', simple_value=loss_plot)
            reward_plot = None
            reward_summary = tf.Summary()
            reward_summary.value.add(tag='reward', simple_value=reward_plot)

            self.c_in = np.zeros((1, 256), np.float32)
            for itr in range(self.n_itr):

                if itr > 0:
                    print("------------------ total loss: %f" % total_loss_before)
                    print("------------------ total loss: %f" % total_loss)

                # set goals of meta tasks
                learner_goals = self.env.sample_goals(self.meta_batch_size)

                obs_list, action_list, adv_list, newobs_list, newaction_list, newadv_list = [], [], [], [], [], []


                print("-------------------- obtaining samples :")
                paths = self.obtain_samples(itr, reset_args=learner_goals, init_state=None)

                print("-------------------- processing samples :")
                samples = {}
                for key in paths.keys():
                    samples[key] = self.process_samples(itr, paths[key])

                for i in range(self.meta_batch_size):
                    inputs = ext.extract(
                        samples[i],
                        "observations", "actions", "rewards"
                    )
                    obs_list.append(inputs[0])
                    action_list.append(inputs[1])
                    adv_list.append(np.asarray(inputs[2]).reshape([-1, 1]))

                print("-------------------------- optimize policy :")

                feedict = {}
                total_loss1, total_loss2 = [], []
                self.task_params = []
                for i in range(self.meta_batch_size):

                    feedict.update({self.policy.network.l_action_inputs: action_list[i][0:-1]})
                    feedict.update({self.policy.network.l_state_inputs: obs_list[i][0:-1]})
                    feedict.update({self.policy.network.goal: obs_list[i][1::] - obs_list[i][0:-1]})
                    feedict.update({self.policy.network.g_state_inputs: obs_list[i][0:-1]})
                    feedict.update({self.policy.network.g_action_inputs: action_list[i][0:-1],
                                    self.policy.network.c_in : self.c_in})


                    total_loss1.append(sess.run(self.policy.network.loss, feed_dict= feedict))
                    _ = sess.run(self.policy.network.apply_training, feed_dict= feedict)
                    loss2, new_params = sess.run([self.policy.network.loss, self.policy.network.c_state_out], feed_dict=feedict)
                    self.c_in = new_params

                    total_loss2.append(loss2)
                    self.task_params.append(new_params)
                self.policy.params = self.task_params
                total_loss_before = np.mean(total_loss1)
                total_loss = np.mean(total_loss2)
                if itr > 0:
                    self.loss_list.append(total_loss_before)
                    reward_summary.value[0].simple_value = total_loss_before
                    self.summary_writer.add_summary(reward_summary, itr)
                    if self.fig == None:
                        self.fig = plt.figure()
                        self.fig.set_size_inches(12, 6)
                    else:
                        self.show_rewards(self.loss_list, self.fig, "loss")
                if itr % 1 == 0:
                    save_path = self.saver.save(sess, './half_cheetah_model/maml_model.ckpt', global_step=itr)
                    np.save('c_state',self.c_in)
                    print("-------------save model : %s " % save_path)
            self.sampler.shutdown_worker()

    def show_rewards(self, rewards, fig, name,width=12, height=6, window_size=1000):
        # sanity checks for plotting
        assert (fig is not None)

        #if len(rewards) == 0:
        #    return

        plt.figure(fig.number)
        plt.clf()
        moving_avg = self.compute_moving_average(rewards, window_size)
        gcf = plt.gcf()
        ax = plt.gca()
        gcf.set_size_inches(width, height)
        plt.xlim((0, len(rewards)))
        r, = plt.plot(rewards, color='red', linestyle='-', linewidth=0.5, label=name, alpha=0.5)
        ave_r, = plt.plot(moving_avg, color='blue', linestyle='-', linewidth=0.8, label='avg_' + name)
        # e, = plt.plot(epsilons, color='blue', linestyle='--', alpha=0.5, label='epsilon')
        plt.legend([r, ave_r], [name, 'average '+ name])
        plt.ylabel(name)
        plt.xlabel('Episode #')
        plt.savefig(name+' fig')
        #plt.pause(0.1)

    def compute_moving_average(self, rewards, window):
        cur_window_size = 1
        moving_average = []
        for i in range(len(rewards) - 1):
            lower_idx = max(0, i - cur_window_size)
            average = sum(rewards[lower_idx:i + 1]) / cur_window_size
            moving_average.append(average)
            cur_window_size += 1
            if cur_window_size > window:
                cur_window_size = window
        return moving_average

    def get_param_values(self):
        all_params = self.old_params
        param_values = tf.get_default_session().run(all_params)
        return param_values

    def _discount(self, x, gamma):
        return signal.lfilter([1.0], [1.0, gamma], x[::-1])[::-1]

    def add_params(self, param_1, param_2):
        if len(param_1) == 0:
            return param_2

        return [param_1[i] + param_2[i] for i in  range(len(param_1))]

    def sub_params(self, param_1, param_2):
        return [param_1[i] - param_2[i] for i in range(len(param_1))]

    def mult_params(self, param_1, param_2 ):
        return [param_1[i] - param_2[i] for i in range(len(param_1))]

    def divide_nums(self, param_1, num):
        return [param_1[i]/num for i in range(len(param_1))]





