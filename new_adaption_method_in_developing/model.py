import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib as contr

class LSTM_network():
    def __init__(self, s_size, a_size, scope, trainer,  num_hidden = 100):
        print("init network: " + str(scope))
        with tf.variable_scope(scope + "lstm_tuning"):
            # Input
            self.l_state_inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name= 'l_state')
            self.l_action_inputs = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name= 'l_action')
            self.l_inputs = tf.concat((self.l_state_inputs, self.l_action_inputs),axis= 1)
            self.goal = tf.placeholder(shape= [None, s_size], dtype= tf.float32, name= 'goal')

            # lstm layers
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, state_is_tuple=True)

            self.c_in = tf.placeholder( tf.float32, (1, lstm_cell.state_size.c))
            self.h_in = tf.zeros((1, lstm_cell.state_size.h), tf.float32)
            self.state_in = [self.c_in, self.h_in]

            rnn_in = tf.expand_dims(self.l_inputs, [0])
            state_in = tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in,
                initial_state=state_in,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            self.c_state_out = lstm_c
            #self.l_outputs = tf.reshape(lstm_outputs, [-1, num_hidden])

        with tf.variable_scope(scope + "global_structure"):
            self.g_state_inputs = tf.placeholder(shape= [None, s_size], dtype=tf.float32, name= 'g_state')
            self.g_action_inputs = tf.placeholder(shape= [None, a_size], dtype= tf.float32, name= 'g_action')
            self.g_inputs = tf.concat((self.g_state_inputs, self.g_action_inputs), axis= 1)
            state_out =tf.reshape( tf.tile(self.c_state_out[:, 0:int(num_hidden/2)], [tf.shape(self.g_inputs)[0],1]),[-1, int(num_hidden/2)])
            self.inputs = tf.concat((self.g_inputs, state_out), axis= 1)

            # fc layers
            layers = contr.layers.fully_connected(self.inputs, num_hidden, activation_fn= tf.nn.relu, normalizer_fn= contr.layers.layer_norm)
            layers = contr.layers.fully_connected(layers, num_hidden, activation_fn= tf.nn.relu, normalizer_fn= contr.layers.layer_norm)

            # output
            self.g_outputs = contr.layers.fully_connected(layers, int(num_hidden/2), activation_fn= tf.nn.relu, normalizer_fn= contr.layers.layer_norm)

            feature_out = tf.multiply(self.c_state_out[:,int(num_hidden/2)::], self.g_outputs)
            layers = contr.layers.fully_connected(feature_out, num_hidden, activation_fn=tf.nn.relu,
                                                  normalizer_fn=contr.layers.layer_norm)
            layers = contr.layers.fully_connected(layers, num_hidden, activation_fn=tf.nn.relu,
                                                  normalizer_fn=contr.layers.layer_norm)
            self.final_output = contr.layers.fully_connected(layers, s_size, activation_fn= None)

        with tf.variable_scope(scope+ "loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.final_output, self.goal))
            self.apply_training = trainer.minimize(self.loss)


        with tf.variable_scope(scope + "target_structure"):
            self.t_state_inputs = tf.placeholder(shape = [None, s_size], dtype= tf.float32, name = 't_state')
            self.t_action_inputs = tf.placeholder(shape = [None, a_size], dtype= tf.float32, name = 't_action')
            self.t_inputs = tf.concat((self.t_state_inputs, self.t_action_inputs), axis= 1)
            self.param_inputs = tf.placeholder(shape= [None, num_hidden], dtype= tf.float32, name= 'param_inputs')
            t_state_out = tf.reshape(tf.tile(self.param_inputs[:, 0:int(num_hidden/2)], [tf.shape(self.t_inputs)[0], 1]), [-1, int(num_hidden/2)])
            self.target_inputs = tf.concat((self.t_inputs, t_state_out), axis= 1)

            #fc layers
            t_layers = contr.layers.fully_connected(self.target_inputs, num_hidden, activation_fn= tf.nn.relu, normalizer_fn= contr.layers.layer_norm)
            t_layers = contr.layers.fully_connected(t_layers, num_hidden, activation_fn= tf.nn.relu, normalizer_fn= contr.layers.layer_norm)

            # output
            self.t_outputs = contr.layers.fully_connected(t_layers, int(num_hidden/2), activation_fn= tf.nn.relu, normalizer_fn= contr.layers.layer_norm)
            t_feature_out = tf.multiply(self.t_outputs, self.param_inputs[:, int(num_hidden/2)::])
            layers = contr.layers.fully_connected(t_feature_out, num_hidden, activation_fn=tf.nn.relu,
                                                  normalizer_fn=contr.layers.layer_norm)
            layers = contr.layers.fully_connected(layers, num_hidden, activation_fn=tf.nn.relu,
                                                  normalizer_fn=contr.layers.layer_norm)
            self.t_final_output = contr.layers.fully_connected(layers, s_size, activation_fn= None)


        with tf.variable_scope(scope + "assign"):
            # assign target params
            global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope='NAT_network' + 'global_structure')
            target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope='NAT_network' + 'target_structure')
            self.op_holder = []
            for from_var, to_var in zip(global_vars, target_vars):
                self.op_holder.append(to_var.assign(from_var))



