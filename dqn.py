import numpy as np, pandas as pd, random
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
from agent import BrawlAgent
from env.brawlstars import Brawlstars
from keras.backend import set_session

EPISODE = 500 # Episode limitation
TRAIN_EVERY_STEPS = 512
BATCH_SIZE = 128 # size of minibatch

# reproducible
random.seed(1992)
np.random.seed(1992)
tf.set_random_seed(1992)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.log_device_placement = True

sess = tf.Session(config=config)
set_session(sess)

def main(isLoad=False):
    # Reset the graph
    tf.reset_default_graph()
    env = Brawlstars()
    agent = BrawlAgent(env)
    for i in tqdm(range(EPISODE)):
        agent.is_updated_target_net = False
        state = agent.env.reset() # To start the process
        done = False
        agent.replay_buffer.clear()
        avg_reward_list = []
    #     actions_list = []
        previous_reward = -1 # to prevent print too much noise
        while done is False:
            action = agent.act(state) # Return Format: [movementArray, actionArray]
            state, reward, done = agent.env.step() # No longer needs action to be passed in
            if reward != previous_reward:
                previous_reward = reward
                # print(reward)
    #         actions_list.append(action)
            avg_reward_list.append(reward)
            if done is False:
                next_state = agent.env._getObservation() # Get the next state
                agent.perceive(state, action, reward, next_state, done)
                if agent.replay_buffer.size() > BATCH_SIZE and env.time_step % TRAIN_EVERY_STEPS == 0:
                    agent.train_dqn_network(i, batch_size=BATCH_SIZE)
        # Update epsilon after every episode
        if agent.epsilon > agent.final_epsilon:
            agent.epsilon -= (1 - agent.final_epsilon) / (EPISODE/1.2)
        print('[{0}] Average Reward: {1}'.format(i+1, np.mean(avg_reward_list)))
        
    #     # log_histogram(agent.summary_writer, 'reward_dist', avg_reward_list, i)
    #     # log_scalars(agent.summary_writer, 'avg_reward', np.mean(avg_reward_list), i)
    #     # log_scalars(agent.summary_writer, 'drawdown', np.mean(np.sum(np.array(avg_reward_list) < INIT_CASH, axis=0)), i)
    #     # log_scalars(agent.summary_writer, 'action_errors', np.mean(agent.env.error_count), i)
        
if __name__ == '__main__':
    main()