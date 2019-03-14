import tensorflow as tf

# Hyper Parameters for DQN
LEARNING_RATE = 5e-6
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.05 # ending value of epislon
DECAY = 0.993 # epsilon decay
GAMMA = 0.95 # discount factor for q value
UPDATE_TARGET_NETWORK = 10
SAVE_NETWORK = 50

class BrawlAgent():

    def __init__(self, env):
        # init some parameters
        self.epsilon = INITIAL_EPSILON
        self.final_epsilon = FINAL_EPSILON
        self.env = env
        self.replay_buffer = Experience_Buffer()
        self.state_dim = env.observation_space.shape[1] # TODO, need to define a structure
        self.action_dim = len(env.action_space) # TODO, need to define a structure
        self.learning_rate = LEARNING_RATE
        self.update_target_net_freq = UPDATE_TARGET_NETWORK # how many timesteps to update target network params
        self.is_updated_target_net = False

        # Reset the graph
        tf.reset_default_graph()
        self.network = DQN_NNET(self.state_dim, self.action_dim, self.learning_rate, 'q_network')
        self.target_network = DQN_NNET(self.state_dim, self.action_dim, self.learning_rate, 'target_q_network')

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initializers.global_variables())

        # # Tensorboard
        # self.summary_writer = tf.summary.FileWriter('logs/' + str(get_latest_run_count()))
        # self.summary_writer.add_graph(self.session.graph)

        # loading networks
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if (self.isTrain is False and checkpoint and checkpoint.model_checkpoint_path) or isLoad is True:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
                print("Could not find old network weights")