import tensorflow as tf
import numpy as np
from self_implement_learning_to_adapt.model import construct_fc_weights,construct_inputs,construct_loss,forward_fc
from self_implement_learning_to_adapt.batch_sampler import ParrallelSampler
from self_implement_learning_to_adapt.vectorized_sampler import VectorizedSampler
from rllab.misc import ext
import matplotlib.pyplot as plt
import scipy.signal as signal
from rllab.sampler.stateful_pool import singleton_pool

class MAML(object):
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
        self.scope = scope
        self.num_samples = num_samples
        self.s_size = self.env.observation_space.shape[0]
        self.a_size = self.env.action_space.shape[0]
        print(self.s_size,  self.a_size)

        self.lr = lr
        self.fast_lr = fast_lr
        self.loss_list = []
        self.reward_list = []
        self.fig = None
        self.save_video = save_video
        self.train_action_inputs, self.train_state_inputs, self.train_goal_inputs = [], [], []
        self.test_action_inputs, self.test_state_inputs, self.test_goal_inputs = [], [], []
        # select sampler
        if singleton_pool.n_parallel >1:
            self.sampler =  ParrallelSampler(self, n_envs= self.meta_batch_size)
        else:
            self.sampler = VectorizedSampler(self, n_envs= self.meta_batch_size)

        # define trainer
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.lr)

        # this is a hacker
        self.f_action_inputs, self.f_state_inputs, self.f_goal = construct_inputs(self.s_size, self.a_size, "first_test")
        with tf.variable_scope("meta_rl_global"):
            self.old_params = construct_fc_weights(self.s_size, self.s_size+ self.a_size, num_hidden= 512)
        self.first_outputs = forward_fc(self.f_action_inputs, self.f_state_inputs, self.old_params, reuse= False)
        self.f_loss = construct_loss(self.first_outputs, self.f_goal)
        self.f_optimizer = self.trainer.minimize(self.f_loss)

        # construct input tensors
        self.construct_tensor_graph()

        self.saver = tf.train.Saver()

    def construct_tensor_graph(self):
        '''
        build maml final graph, directly optimize the initial prior model
        :return:
        '''
        self.test_outputs, self.train_outputs, self.new_params, self.train_goal_inputs = [], [], [], []
        # construct inputs and network for each meta task
        for i in range(self.meta_batch_size):
            tensor_action_inputs, tensor_state_inputs, tensor_goal_inputs = construct_inputs(a_size=self.a_size, s_size=self.s_size,
                                                                         scpoe="train_inputs" + str(i))
            outputs = forward_fc(tensor_action_inputs, tensor_state_inputs, weights=self.old_params,
                                 reuse=True)
            self.train_action_inputs.append(tensor_action_inputs)
            self.train_state_inputs.append(tensor_state_inputs)
            self.train_goal_inputs.append(tensor_goal_inputs)
            self.train_outputs.append(outputs)
        # maml train case, do first gradients
        for i in range(self.meta_batch_size):
            loss = construct_loss(self.train_outputs[i], self.train_goal_inputs[i])

            grads = tf.gradients(loss, list(self.old_params.values()))
            gradients = dict(zip(self.old_params.keys(), grads))
            # save the params
            self.new_params.append(dict(zip(self.old_params.keys(),
                                            [self.old_params[key] - self.fast_lr * gradients[key] for key in
                                             self.old_params.keys()])))

        # maml test case, second order gradients
        for i in range(self.meta_batch_size):
            tensor_action_inputs, tensor_state_inputs, tensor_goal_inputs = construct_inputs(a_size=self.a_size, s_size=self.s_size,
                                                                         scpoe="test_inputs" + str(i))
            outputs = forward_fc(tensor_action_inputs, tensor_state_inputs, weights=self.new_params[i],
                                 reuse=True)
            self.test_action_inputs.append(tensor_action_inputs)
            self.test_state_inputs.append(tensor_state_inputs)
            self.test_goal_inputs.append(tensor_goal_inputs)
            self.test_outputs.append(outputs)
        self.cur_params = [self.old_params for i in range(self.meta_batch_size)]

        # define total loss
        self.total_loss_list = []
        for i in range(self.meta_batch_size):
            # save the params
            self.total_loss_list.append(construct_loss(self.test_outputs[i], self.test_goal_inputs[i]))
        with tf.variable_scope("total_loss"):
            self.total_loss_before = tf.reduce_mean(tf.stack(self.total_loss_list))
            self.second_gradients = self.trainer.minimize(self.total_loss_before, var_list= self.old_params)

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

    def cheetah_cost_fn(self,state, action, next_state):
        if len(state.shape) > 1:
            heading_penalty_factor = 10
            scores = np.zeros((state.shape[0],))

            # dont move front shin back so far that you tilt forward
            front_leg = state[:, 5]
            my_range = 0.2
            scores[front_leg >= my_range] += heading_penalty_factor

            front_shin = state[:, 6]
            my_range = 0
            scores[front_shin >= my_range] += heading_penalty_factor

            front_foot = state[:, 7]
            my_range = 0
            scores[front_foot >= my_range] += heading_penalty_factor

            scores -= (next_state[:, 17] - state[:, 17]) / 0.01   + 0.1 * (np.sum(action**2, axis=1))
            return scores

        heading_penalty_factor = 10
        score = 0

        # dont move front shin back so far that you tilt forward
        front_leg = state[5]
        my_range = 0.2
        if front_leg >= my_range:
            score += heading_penalty_factor

        front_shin = state[6]
        my_range = 0
        if front_shin >= my_range:
            score += heading_penalty_factor

        front_foot = state[7]
        my_range = 0
        if front_foot >= my_range:
            score += heading_penalty_factor

        score -= (next_state[17] - state[17]) / 0.01  + 0.1 * (np.sum(action**2))
        return score

    def MPC(self,itr, num_samples, init_state, goal):

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
            diff = self.sess.run(self.first_outputs, feed_dict={self.f_state_inputs: np.asarray(new_obs).reshape([-1,self.s_size]),
                                                            self.f_action_inputs: np.asarray(action).reshape([-1,self.a_size])})
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
                for step in range(self.max_path_length):
                    #if step>int(self.max_path_length)*0.7:
                    #    self.env.render()
                    if len(act) > 0:
                        indices = np.random.randint(0, len(act), len(act))
                        _ = sess.run([ self.f_optimizer],
                                            feed_dict={self.f_action_inputs: np.asarray(act)[indices,:],
                                                       self.f_state_inputs: np.asarray(obs)[indices,:],
                                                       self.f_goal: np.asarray(diffs)[indices,:]})
                        loss, output = sess.run([self.f_loss,self.first_outputs], feed_dict={self.f_action_inputs: np.asarray(act)[indices,:],
                                                       self.f_state_inputs: np.asarray(obs)[indices,:],
                                                       self.f_goal: np.asarray(diffs)[indices,:]})
                        #diff = np.mean(abs(np.asarray(obs[1:-1])-np.asarray(obs[0:-2]) - output[0:-2]))
                        #diff_summary.value[0].simple_value = diff
                        loss_summary.value[0].simple_value = loss
                        self.summary_writer.add_summary(loss_summary, nstep)
                        self.summary_writer.add_summary(diff_summary, nstep)

                    obs.append(new_state)
                    if step%100 == 0:
                        print("Doing MPC, step:", step)

                    action = self.MPC(itr = itr, num_samples= self.num_samples, goal= goal, init_state= new_state)
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
                        image = self.env.wrapped_env.get_viewer().get_image()
                        pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
                        images.append(np.flipud(np.array(pil_image)))

                if self.save_video and itr == self.n_itr -1 :
                    import moviepy.editor as mpy
                    clip = mpy.ImageSequenceClip(images, fps=20 * 1)
                    clip.write_videofile("./video/half_cheetah/", fps=20 * 1)
                self.saver.save(sess, './MPC_model/mpc_model.cpkt', global_step=itr)

                if itr >= 0:
                    sum_rewards = np.sum(np.asarray(rewards))
                    print(sum_rewards)
                    self.reward_list.append(sum_rewards)

                    reward_summary.value[0].simple_value = sum_rewards
                    self.summary_writer.add_summary(reward_summary, itr)

                    if self.fig == None :
                        self.fig = plt.figure()
                        self.fig.set_size_inches(12, 6)
                        self.fig1= plt.figure()
                    else:
                        self.show_rewards(self.reward_list, self.fig, "rewards")


    def train(self):
        '''
        meta training of transition model : sample trajectories based on different tasks, doing optimization
        :return:
        '''
        self.sess = tf.Session()
        with self.sess as sess:

            self.summary_writer = tf.summary.FileWriter("./graph/", self.sess.graph)
            if self.load_policy:
                sess.run(tf.global_variables_initializer())
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
            reward_summary.value.add(tag = 'reward', simple_value = reward_plot)
            for itr in  range(self.n_itr):


                if itr>0:

                    print("------------------ total loss: %f" % total_loss_before)
                    print("------------------ total loss: %f" % total_loss)

                # set goals of meta tasks
                learner_goals = self.env.sample_goals(self.meta_batch_size)

                obs_list, action_list, adv_list, newobs_list, newaction_list, newadv_list = [], [], [], [], [], []
                for step in range(self.num_grad_updates+1):


                    print("-------------------- step: " + str(step))
                    print("-------------------- obtaining samples :")
                    paths = self.obtain_samples(itr, reset_args= learner_goals,init_state= None)

                    print("-------------------- processing samples :")
                    samples = {}
                    for key in paths.keys():
                        samples[key] = self.process_samples(itr, paths[key])

                    if step == 0:
                        for i in range(self.meta_batch_size):
                            inputs = ext.extract(
                                samples[i],
                                "observations", "actions", "rewards"
                            )
                            obs_list.append(inputs[0])
                            action_list.append(inputs[1])
                            adv_list.append(np.asarray(inputs[2]).reshape([-1,1]))

                    else:
                        for i in range(self.meta_batch_size):
                            inputs = ext.extract(
                                samples[i],
                                "observations", "actions", "rewards"
                            )
                            newobs_list.append(inputs[0])
                            newaction_list.append(inputs[1])
                            newadv_list.append(np.asarray(inputs[2]).reshape([-1,1]))


                    #if step == 0:
                    #    print("--------------------  Compute local gradients : ")
                    #    # apply first gradients, optimize original params
                    #    assign_op = []



                print("-------------------------- optimize policy :")

                feedict = {}
                for i in range(self.meta_batch_size):

                    feedict.update({self.train_action_inputs[i]: action_list[i][0:-1]})
                    feedict.update({self.train_state_inputs[i]: obs_list[i][0:-1]})
                    feedict.update({self.train_goal_inputs[i]: obs_list[i][1::] - obs_list[i][0:-1]})
                    feedict.update({self.test_action_inputs[i]: newaction_list[i][0:-1]})
                    feedict.update({self.test_state_inputs[i]: newobs_list[i][0:-1]})
                    feedict.update({self.test_goal_inputs[i]: newobs_list[i][1::] - newobs_list[i][0:-1] })

                total_loss_before= sess.run(self.total_loss_before, feed_dict= feedict)
                _ = sess.run([ self.second_gradients], feed_dict= feedict)
                total_loss = sess.run(self.total_loss_before,
                                      feed_dict=feedict)
                if itr > 0:
                    self.loss_list.append(total_loss_before)
                    reward_summary.value[0].simple_value = total_loss_before
                    self.summary_writer.add_summary(reward_summary, itr)
                    if self.fig == None :
                        self.fig = plt.figure()
                        self.fig.set_size_inches(12, 6)
                    else:
                        self.show_rewards(self.loss_list, self.fig, "loss")
                if itr%1 == 0:
                    save_path = self.saver.save(sess, './half_cheetah_model/maml_model.ckpt', global_step = itr)
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

    def set_param_values(self, params):
        tf.get_default_session().run(self.update_target_graph(params, "meta_rl" + str(i)))

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





