import tensorflow as tf
import numpy as np

# NN Parameters
NN1_NEURONS = 32
NN2_NEURONS = 16
beta = 0.01 # regularization
dropout = 0.03 # dropout

# TODO: refactor this class to create components for layer creation - better stats

class DQN_NNET:

    def __init__(self, state_dim, action_dim, learning_rate, name):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.name = name
        
        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):
            
            # We create the placeholders
            self.state_input = tf.placeholder(dtype=tf.float64, shape=[None,self.state_dim], name='state_input')
            self.action_input = tf.placeholder(dtype=tf.float64, shape=[None, self.action_dim], name='action_input')
            self.Q_input = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='target')

            self.W1 = tf.Variable(dtype=tf.float64,
                                initial_value=tf.random_normal([self.state_dim, NN1_NEURONS],
                                dtype=tf.float64,
                                name='W1'))
            self.B1 = tf.Variable(dtype=tf.float64,
                                initial_value=tf.zeros([NN1_NEURONS], tf.float64),
                                name='B1')
            self.layer1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(self.state_input, self.W1), self.B1)), keep_prob=(1-dropout), name='layer1_dropout')

            self.W2 = tf.Variable(dtype=tf.float64,
                                initial_value=tf.random_normal([NN1_NEURONS, NN2_NEURONS],
                                dtype=tf.float64,
                                name='W2'))
            self.B2 = tf.Variable(dtype=tf.float64,
                                initial_value=tf.zeros([NN2_NEURONS], tf.float64),
                                name='B2')
            self.layer2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(self.layer1, self.W2), self.B2)), keep_prob=(1-dropout), name='layer2_dropout')

            self.W_O = tf.Variable(dtype=tf.float64,
                                initial_value=tf.random_normal([NN2_NEURONS, self.action_dim],
                                dtype=tf.float64,
                                name='W_Output'))
            self.B_O = tf.Variable(dtype=tf.float64,
                                initial_value=tf.zeros([self.action_dim], tf.float64),
                                name='B_Output')
            self.output = tf.add(tf.matmul(self.layer2, self.W_O), self.B_O, name='output')

            self.Q_value = tf.reduce_sum(tf.multiply(self.output, self.action_input), axis=1, name='Q_value')

            # Cost
            # self.cost = tf.reduce_mean(tf.square(self.Q_input - tf.reduce_sum(
                # self.output * self.action_input)), reduction_indices=1)
            self.cost = tf.reduce_mean(tf.square(self.Q_input - self.Q_value))

            # L2 Regularization
            reg = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2) + tf.nn.l2_loss(self.W_O)
            
            # Optimizer
            with tf.name_scope("train"):
                self.cost = tf.reduce_mean(self.cost + reg * beta, name='cost')
                self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

            # Tensorboard Stats
        #     W1_summary = variable_summaries(self.W1)
        #     W2_summary = variable_summaries(self.W2)
        #     W_O_summary = variable_summaries(self.W_O)

        #     B1_summary = variable_summaries(self.B1)
        #     B2_summary = variable_summaries(self.B2)
        #     B_O_summary = variable_summaries(self.B_O)

        #     layer1_summary = variable_summaries(self.layer1)
        #     layer2_summary = variable_summaries(self.layer2)

        #     output_summary = variable_summaries(self.output)
        #     Q_value_summary = variable_summaries(self.Q_value)
        #     cost_summary = variable_summaries(self.cost)

        # self.merged_summary = tf.summary.merge(
        #                W1_summary +
        #                W2_summary +
        #                W_O_summary +
        #             #    B1_summary +
        #             #    B2_summary +
        #             #    B_O_summary +
        #             #    layer1_summary +
        #             #    layer2_summary +
        #                output_summary +
        #                Q_value_summary +
        #                cost_summary
        #                )