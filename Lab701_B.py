# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:12:39 2019

@author: user
"""

import numpy as np
import tensorflow as tf
import random
import gym
from collections import deque
import _model_dqn_701_A as m

env = gym.make('CartPole-v0')
# env = gym.wrappers.Monitor(env, 'gym-results/', force=True)

INPUT_SIZE = env.observation_space.shape[0]     # 4
OUTPUT_SIZE = env.action_space.n                # 2

DISCOUNT_RATE = 0.9
REPLAY_MEMORY = 50000
MAX_EPISODE = 5000
BATCH_SIZE = 64
MIN_E = 0.0     # minimum epsilon for E-greedy
EPSILON_DECAYING_EPISODE = MAX_EPISODE * 0.01


def bot_play(dqn: m.DQN) -> None:
    """Runs a single episode with rendering and prints a reward

    Args:
        mainDQN (dqn.DQN): DQN Agent
    """
    _state = env.reset()
    _reward_sum = 0
    while True:
        env.render()
        _action = np.argmax(dqn.predict(_state))
        _state, _reward, _done, _ = env.step(action)
        _reward_sum += _reward
        if done:
            print("Total score: {}".format(_reward_sum))
            break


# def simple_replay_train(DQN, train_batch):
def train_sample_replay(dqn: m.DQN, train_batch: list) -> list:
    """Prepare X_batch, y_batch and train them

    Recall our loss function is
        target = reward + discount * max Q(s',a)
                 or reward if done early
        Loss function: [target - Q(s, a)]^2

    Hence,
        X_batch is a state list
        y_batch is reward + discount * max Q
                   or reward if terminated early

    Args:
        dqn (m.DQN): DQN Agent to train & run
        train_batch (list): Mini batch of Sample Replay memory
            Each element is a tuple of (state, action, reward, next_state, done)

    Returns:
        loss: Returns a loss
    """
    state_array = np.vstack([x[0] for x in train_batch])
    next_state_array = np.vstack([x[3] for x in train_batch])
    action_array = np.hstack([x[1] for x in train_batch])
    reward_array = np.hstack([x[2] for x in train_batch])
    done_array = np.hstack([x[4] for x in train_batch])

    x_batch = state_array
    ''' ??? '''
    y_batch = np.array(np.arange(len(x_batch)), action_array)  # Q_Predict ???

    # Q_Target
    y_batch[np.arange(len(x_batch)), action_array]\
        = reward_array \
          + DISCOUNT_RATE * np.max(dqn.predict(next_state_array), axis=1)\
          * ~done_array   # if done is 1 then ignore next_Q

    '''
     x_stack = np.vstack([x_stack, state])
     y_stack = np.vstack([y_stack, Q])
     Q_pred[0, action] = reward + dis * np.max(DQN.predict(next_state))
    '''

    return dqn.update(x_batch, y_batch)


def annealing_epsilon(episode: int, min_e: float, max_e: float, target_episode: int) -> float:
    """Return an linearly annealed epsilon
    Epsilon will decrease over time until it reaches `target_episode`

         (epsilon)
             |
    max_e ---|\
             | \
             |  \
             |   \
    min_e ---|____\_______________(episode)
                  |
                 target_episode

     slope = (min_e - max_e) / (target_episode)
     intercept = max_e

     e = slope * episode + intercept

    Args:
        episode (int): Current episode
        min_e (float): Minimum epsilon
        max_e (float): Maximum epsilon
        target_episode (int): epsilon becomes the `min_e` at `target_episode`

    Returns:
        float: epsilon between `min_e` and `max_e`
    """
    slope = (min_e - max_e) / (target_episode)
    intercept = max_e

    return max(min_e, slope * episode + intercept)


replay_buffer = deque(maxlen=REPLAY_MEMORY)
last_100_game_reward = deque(maxlen=100)

with tf.Session() as sess:
    mainDQN = m.DQN(sess, INPUT_SIZE, OUTPUT_SIZE)
    sess.run(tf.global_variables_initializer())

    for episode in range(MAX_EPISODE):
        e = 1./((episode/10)+1)
        # e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)
        done = False
        step_count = 0
        state = env.reset()

        while not done:
            if np.random.rand() < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(mainDQN.predict(state))

            next_state, reward, done, _ = env.step(action)

            if done:
                reward = -1

            replay_buffer.append((state, action, reward, next_state, done))

            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()

            state = next_state
            step_count += 1
            if step_count > 10000:
                break

        print("Episode: {} steps: {}".format(episode, step_count))
        if step_count > 10000:
            pass

        if episode % 10 == 1:
            for _ in range(50):
                minibatch = random.sample(replay_buffer, 10)
                cost, _ = train_sample_replay(mainDQN, minibatch)
            print("Cost: ", cost)
    bot_play(mainDQN)

# https://mclearninglab.tistory.com/35
