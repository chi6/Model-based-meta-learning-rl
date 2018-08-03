import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim



def construct_inputs(s_size, a_size, scpoe):
    with tf.variable_scope(scpoe):
        # Input
        state_inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name= 'state')
        action_inputs = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name= 'action')
        goal = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name='goal')

    return action_inputs, state_inputs, goal

### Network construction functions (fc networks and conv networks)
def construct_fc_weights(s_size, input_size, num_hidden):
    weights = {}
    weights['w1'] = tf.Variable(tf.truncated_normal([input_size, num_hidden], stddev=0.01))
    weights['b1'] = tf.Variable(tf.zeros(num_hidden))
    weights['w' + str(2)] = tf.Variable(
        tf.truncated_normal([num_hidden, num_hidden], stddev=0.01))
    weights['b' + str(2)] = tf.Variable(tf.zeros([num_hidden]))
    #weights['w' + str(3)] = tf.Variable(
    #    tf.truncated_normal([num_hidden, num_hidden], stddev=0.01))
    #weights['b' + str(3)] = tf.Variable(tf.zeros([num_hidden]))
    #for i in range(1,len(1)):
    #    weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
    #    weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
    weights['w'+str(3)] = tf.Variable(tf.truncated_normal([num_hidden, s_size], stddev=0.01))
    weights['b'+str(3)] = tf.Variable(tf.zeros([s_size]))
    return weights

def forward_fc( action_inputs, state_inputs , weights, reuse=False):
    inputs = tf.concat((state_inputs, action_inputs), axis=1)
    hidden = normalize(tf.matmul(inputs, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
    hidden = normalize(tf.matmul(hidden, weights['w2']) + weights['b2'], activation=tf.nn.relu, reuse=reuse, scope='1')
    #hidden = normalize(tf.matmul(hidden, weights['w3']) + weights['b3'], activation=tf.nn.relu, reuse=reuse, scope='2')
    #for i in range(1,len(self.dim_hidden)):
    #hidden = self.normalize(tf.matmul(hidden, weights['w'+str(2)]) + weights['b'+str(2)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
    return tf.matmul(hidden, weights['w'+str(3)]) + weights['b'+str(3)]

def normalize(inp, activation, reuse, scope):
    return tf.contrib.layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)

def construct_loss(outputs, goal):
    # loss
    loss = tf.reduce_mean(tf.squared_difference(outputs, goal))

    return loss


