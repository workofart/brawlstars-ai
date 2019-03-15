import tensorflow as tf
import random
import numpy as np
from experiencebuffer import Experience_Buffer
from net.dqnet import DQN_NNET

# Hyper Parameters for DQN
LEARNING_RATE = 5e-6
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.05 # ending value of epislon
DECAY = 0.993 # epsilon decay
GAMMA = 0.95 # discount factor for q value
UPDATE_TARGET_NETWORK = 10
SAVE_NETWORK = 50

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

        # Reset the graph
        # tf.reset_default_graph()
        
        # Action Q_networks
        self.a_network = DQN_NNET(self.state_dim, self.action_dim, self.learning_rate, 'action_q_network')
        self.a_target_network = DQN_NNET(self.state_dim, self.action_dim, self.learning_rate, 'action_target_q_network')

        # Movement Q_networks
        self.m_network = DQN_NNET(self.state_dim, self.movement_dim, self.learning_rate, 'movement_q_network')
        self.m_target_network = DQN_NNET(self.state_dim, self.movement_dim, self.learning_rate, 'movement_target_q_network')

        # Init session
        # self.session = tf.InteractiveSession()
        # self.session.run(tf.initializers.global_variables())

        # # Tensorboard
        # self.summary_writer = tf.summary.FileWriter('logs/' + str(get_latest_run_count()))
        # self.summary_writer.add_graph(self.session.graph)

        # loading networks
        # self.saver = tf.train.Saver()
        # checkpoint = tf.train.get_checkpoint_state("saved_networks")
        # if (self.isTrain is False and checkpoint and checkpoint.model_checkpoint_path) or isLoad is True:
        #     self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
        # else:
        #     print("Could not find old network weights")

    def act(self, state):
        # if self.isTrain is True and self.epsilon > FINAL_EPSILON:
            # self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / self.env.data_length

        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            output = self.network.output.eval(feed_dict = {
			self.network.state_input:state
			})[0]
            action = np.argmax(output)
            return action


    def perceive(self, state, action, reward, next_state, done):
        # Assumes "replay_buffer" contains [state, action, reward, next_state, done]
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.add([state, one_hot_action, reward, next_state, done])

    def update_target_q_net_if_needed(self, step):
        if step % self.update_target_net_freq == 0 and step > 0 and self.is_updated_target_net is False:
            # Get the parameters of our DQNNetwork
            from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "q_network")
            
            # Get the parameters of our Target_network
            to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "target_q_network")

            op_holder = []
            
            # Update our target_q_network parameters with q_network parameters
            for from_var,to_var in zip(from_vars,to_vars):
                op_holder.append(to_var.assign(from_var))
            self.session.run(op_holder)

            self.is_updated_target_net = True
            # print('Timesteps:{} | Target Q-network has been updated.'.format(self.env.time_step))

    def train_dqn_network(self, ep, batch_size=32):
        self.update_target_q_net_if_needed(ep)
        # Assumes "replay_samples" contains [state, action, reward, next_state, done]
        replay_samples = self.replay_buffer.sample(batch_size)

        state_batch = np.reshape([data[0] for data in replay_samples], (batch_size, 4))
        action_batch = np.reshape([data[1] for data in replay_samples], (batch_size, self.action_dim))
        reward_batch = np.reshape([data[2] for data in replay_samples], (batch_size, 1))
        next_state_batch = np.reshape([data[3] for data in replay_samples], (batch_size, 4))

        # Get the Target Q-value for the next state using the target network,
        # by making a second forward-prop
        target_q_val_batch = self.session.run(self.target_network.output, feed_dict={self.target_network.state_input:next_state_batch})

        # Get Q values for next state using the q-network
        q_val_batch = self.session.run(self.network.output, feed_dict={self.network.state_input:next_state_batch})
        
        # Target Q-value - "advantages/q-vals" derived from rewards
        y_batch = []
        for i in range(0, batch_size):
            # Use Q-network to select the best action for next state
            action = np.argmax(q_val_batch[i])

            done = replay_samples[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(max(-1, min(reward_batch[i] + GAMMA * target_q_val_batch[i][action], 1)))

        # Train on one batch on the Q-network
        _, c, summary = self.session.run([self.network.optimizer, self.network.cost, self.network.merged_summary],
                            feed_dict={
                                self.network.Q_input: np.reshape(y_batch, (batch_size, 1)),
                                self.network.action_input: action_batch,
                                self.network.state_input: state_batch
                            }
        )
        self.summary_writer.add_summary(summary, ep)

        # save network 9 times per episode
        if ep % SAVE_NETWORK == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = ep)

        return c
