import tensorflow as tf
import random, time
import numpy as np
from experiencebuffer import Experience_Buffer
from net.dqnet import DQN_NNET
from utilities.utilities import take_action
from utilities.window import WindowMgr

# Hyper Parameters for DQN
LEARNING_RATE = 1e-4
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.05 # ending value of epislon
DECAY = 0.993 # epsilon decay
GAMMA = 0.95 # discount factor for q value
UPDATE_TARGET_NETWORK = 2
SAVE_NETWORK = 3
w = WindowMgr()
class BrawlAgent:
    def __init__(self, env):
        # init some parameters
        self.epsilon = INITIAL_EPSILON
        self.final_epsilon = FINAL_EPSILON
        self.env = env
        self.replay_buffer = Experience_Buffer()
        self.state_dim = env.observation_space.shape[1] # TODO, need to define a structure
        self.action_dim = len(env.action_space)
        self.movement_dim = len(env.movement_space)
        self.learning_rate = LEARNING_RATE
        self.update_target_net_freq = UPDATE_TARGET_NETWORK # how many timesteps to update target network params
        self.is_updated_target_net = False
        
        # Action Q_networks
        self.a_network = DQN_NNET(self.state_dim, self.action_dim, self.learning_rate, 'action_q_network')
        self.a_target_network = DQN_NNET(self.state_dim, self.action_dim, self.learning_rate, 'action_target_q_network')

        # Movement Q_networks
        self.m_network = DQN_NNET(self.state_dim, self.movement_dim, self.learning_rate, 'movement_q_network')
        self.m_target_network = DQN_NNET(self.state_dim, self.movement_dim, self.learning_rate, 'movement_target_q_network')

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initializers.global_variables())

        # # Tensorboard
        # self.summary_writer = tf.summary.FileWriter('logs/' + str(get_latest_run_count()))
        # self.summary_writer.add_graph(self.session.graph)

        # loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if (checkpoint and checkpoint.model_checkpoint_path):
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def act(self, state):
        # if self.isTrain is True and self.epsilon > FINAL_EPSILON:
            # self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / self.env.data_length
        if random.random() <= self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            movement = random.randint(0, self.movement_dim - 1)
        else:
            a_output = self.a_network.output.eval(feed_dict = {
			self.a_network.state_input:state
			})[0]
            action = np.argmax(a_output)

            m_output = self.m_network.output.eval(feed_dict = {
			self.m_network.state_input:state
			})[0]
            movement = np.argmax(m_output)

        # print('Selected Action: {0}'.format(action))
        # print('Selected Movement: {0}'.format(movement))

        take_action(movement, action)

        return [movement, action]


    def perceive(self, state, action, reward, next_state, done):
        # Assumes "replay_buffer" contains [state, movement, action, reward, next_state, done]
        one_hot_movement = np.zeros(self.movement_dim)
        one_hot_movement[action[0]] = 1

        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action[1]] = 1
        
        self.replay_buffer.add([state, one_hot_movement, one_hot_action, reward, next_state, done])

    def update_target_q_net_if_needed(self, step):
        if step % self.update_target_net_freq == 0 and step > 0 and self.is_updated_target_net is False:
            # Get the parameters of our DQNNetwork
            m_from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "m_q_network")
            
            # Get the parameters of our Target_network
            m_to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "m_target_q_network")

            # Get the parameters of our DQNNetwork
            a_from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "a_q_network")
            
            # Get the parameters of our Target_network
            a_to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "a_target_q_network")

            op_holder = []
            
            # Update our target_q_network parameters with q_network parameters
            for from_var,to_var in zip(m_from_vars,m_to_vars):
                op_holder.append(to_var.assign(from_var))
            for from_var,to_var in zip(a_from_vars,a_to_vars):
                op_holder.append(to_var.assign(from_var))
            self.session.run(op_holder)

            self.is_updated_target_net = True
            print('Timesteps:{} | Target Q-network has been updated.'.format(self.env.time_step))

    def train_dqn_network(self, ep, batch_size=32):
        self.update_target_q_net_if_needed(ep)
        # Assumes "replay_samples" contains [state, movement, action, reward, next_state, done]
        replay_samples = self.replay_buffer.sample(batch_size)

        state_batch = np.reshape([data[0] for data in replay_samples], (batch_size, self.state_dim))
        movement_batch = np.reshape([data[1] for data in replay_samples], (batch_size, self.movement_dim))
        action_batch = np.reshape([data[2] for data in replay_samples], (batch_size, self.action_dim))
        reward_batch = np.reshape([data[3] for data in replay_samples], (batch_size, 1))
        next_state_batch = np.reshape([data[4] for data in replay_samples], (batch_size, self.state_dim))

        # Get the Target Q-value for the next state using the target network,
        # by making a second forward-prop
        m_target_q_val_batch = self.session.run(self.m_target_network.output, feed_dict={self.m_target_network.state_input:next_state_batch})
        a_target_q_val_batch = self.session.run(self.a_target_network.output, feed_dict={self.a_target_network.state_input:next_state_batch})

        # Get Q values for next state using the q-network
        m_q_val_batch = self.session.run(self.m_network.output, feed_dict={self.m_network.state_input:next_state_batch})
        a_q_val_batch = self.session.run(self.a_network.output, feed_dict={self.a_network.state_input:next_state_batch})
        
        # Target Q-value - "advantages/q-vals" derived from rewards
        m_y_batch = []
        a_y_batch = []
        for i in range(0, batch_size):
            # Use Q-network to select the best action for next state
            movement = np.argmax(m_q_val_batch[i])
            action = np.argmax(a_q_val_batch[i])

            done = replay_samples[i][5]
            if done:
                m_y_batch.append(reward_batch[i])
                a_y_batch.append(reward_batch[i])
            else:
                m_y_batch.append(reward_batch[i] + GAMMA * m_target_q_val_batch[i][movement])
                a_y_batch.append(reward_batch[i] + GAMMA * a_target_q_val_batch[i][action])

        # Train on one batch on the Q-network
        start_time = time.time()
        # _, m_c, m_summary = self.session.run([self.m_network.optimizer, self.m_network.cost, self.m_network.merged_summary],
        _, m_c = self.session.run([self.m_network.optimizer, self.m_network.cost],
                            feed_dict={
                                self.m_network.Q_input: np.reshape(m_y_batch, (batch_size, 1)),
                                self.m_network.action_input: movement_batch,
                                self.m_network.state_input: state_batch
                            }
        )

        # _, a_c, a_summary = self.session.run([self.a_network.optimizer, self.a_network.cost, self.a_network.merged_summary],
        _, a_c = self.session.run([self.a_network.optimizer, self.a_network.cost],
                            feed_dict={
                                self.a_network.Q_input: np.reshape(a_y_batch, (batch_size, 1)),
                                self.a_network.action_input: action_batch,
                                self.a_network.state_input: state_batch
                            }
        )

        print('Training time: ' + str(time.time() - start_time))
        # self.summary_writer.add_summary(m_summary, ep)
        # self.summary_writer.add_summary(a_summary, ep)

        # save network 9 times per episode
        if ep % SAVE_NETWORK == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = ep)

        return m_c, a_c
