import numpy as np, pandas as pd, random
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
from agent import BrawlAgent
from env.brawlstars import Brawlstars

EPISODE = 10 # Episode limitation
TRAIN_EVERY_STEPS = 16
TEST_EVERY_EP = 100
BATCH_SIZE = 32 # size of minibatch
DATA_LENGTH = 250 # How many times steps to use in the data

# reproducible
random.seed(1992)
np.random.seed(1992)
tf.set_random_seed(1992)

def main(isLoad=False):
    env = Brawlstars()
    agent = BrawlAgent(env)
    for i in tqdm(range(EPISODE)):
    #     agent.isTrain = True
    #     agent.is_updated_target_net = False
        state = agent.env.reset() # To start the process
        done = False
    #     agent.replay_buffer.clear()
    #     avg_reward_list = []
    #     actions_list = []
        previous_reward = -1 # to prevent print too much noise
        while done is False:
            action = agent.act(state)
    #         state, reward, done, _ = agent.env.step(action)
            state, reward, done = agent.env.step(1)
            if reward != previous_reward:
                previous_reward = reward
                print(reward)
    #         actions_list.append(action)
    #         avg_reward_list.append(reward)
    #         if done is False:
    #             next_state = agent.env._get_obs() # Get the next state
    #             agent.perceive(state, action, reward, next_state, done)
    #             if agent.replay_buffer.size() > BATCH_SIZE and env.time_step % TRAIN_EVERY_STEPS == 0:
    #                 agent.train_dqn_network(i, batch_size=BATCH_SIZE)
        # Update epsilon after every episode
        if agent.epsilon > agent.final_epsilon:
            agent.epsilon -= (1 - agent.final_epsilon) / (EPISODE/1.2)
        
    #     # log_histogram(agent.summary_writer, 'reward_dist', avg_reward_list, i)
    #     # log_scalars(agent.summary_writer, 'avg_reward', np.mean(avg_reward_list), i)
    #     # log_scalars(agent.summary_writer, 'drawdown', np.mean(np.sum(np.array(avg_reward_list) < INIT_CASH, axis=0)), i)
    #     # log_scalars(agent.summary_writer, 'action_errors', np.mean(agent.env.error_count), i)
        
    #     if i % TEST_EVERY_EP == 0 and i > 0:
    #         test(agent, i)

def test(agent, ep = 0):
    agent.isTrain = False
    state = agent.env.reset() # To start the process

    prices = []
    actions = []
    for i in range(DATA_LENGTH):
        prices.append(state[0][2])
        action = agent.act(state)
        actions.append(action)
        state, reward, done, _ = agent.env.step(action)
    plot_trades(ep, prices, actions, agent.env.permitted_trades)



if __name__ == '__main__':
    main()