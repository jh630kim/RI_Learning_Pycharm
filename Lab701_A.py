# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:12:39 2019

@author: user
"""

import numpy as np
import tensorflow as tf
import random
import os

import gym
from collections import deque
# 모델 클래스
import _model_dqn_701_B as m

# Cart Pole 환경
env = gym.make('CartPole-v0')
# env = gym.wrappers.Monitor(env, 'gym-results/', force=True)

INPUT_SIZE = env.observation_space.shape[0]     # 4
OUTPUT_SIZE = env.action_space.n                # 2

dis = 0.9   # discount rate
REPLAY_MEMORY = 50000
MAX_EPISODE = 5000
BATCH_SIZE = 64
MIN_E = 0.0     # minimum epsilon for E-greedy
EPSILON_DECAYING_EPISODE = MAX_EPISODE * 0.01

dataDir = './data/'
saveData = dataDir + 'save'

def bot_play(mainDQN: m.DQN) -> None:
    """Runs a single episode with rendering and prints a reward

    Args:
        mainDQN (dqn.DQN): DQN Agent
    """
    state = env.reset()
    reward_sum = 0
    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break


# def simple_replay_train(DQN, train_batch):
def train_sample_replay(dqn: m.DQN, train_batch: list) -> float:
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
    x_stack = np.empty(0).reshape(0, dqn.input_size)
    y_stack = np.empty(0).reshape(0, dqn.output_size)

    # get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q_pred = dqn.predict(state)

        if done:    # terminal?
            Q_pred[0, action] = reward
        else:
            Q_pred[0, action] = reward + dis * np.max(dqn.predict(next_state))

        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q_pred])

    return dqn.update(x_stack, y_stack)


def main():
    # 5000
    max_episodes = MAX_EPISODE
    # store the previous observation in replay memory
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = m.DQN(sess, INPUT_SIZE, OUTPUT_SIZE)

        # 학습 데이터 저장소
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

        tf.global_variables_initializer().run()

        for episode in range(max_episodes):
            e = 1./((episode/10)+1)
            done = False
            step_count = 0

            state = env.reset()
            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, _ = env.step(action)

                if done:
                    reward = -100

                replay_buffer.append((state, action, reward,
                                      next_state, done))

                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000:  # Good Enough
                    break

            print("Episode: {} steps: {}".format(episode, step_count))
            if step_count > 10000:
                pass

            if episode % 10 == 1:   # train every 10 episodes
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    cost, _ = train_sample_replay(mainDQN, minibatch)
                print("Cost: ", cost)

            # 1회 배치 학습이 끝나면 모델 저장
            checkpoint_path = os.path.join(saveData, "model.ckpt")
            filename = saver.save(sess, checkpoint_path)
            print("Model saved in file: %s" % filename)

        bot_play(mainDQN)


if __name__ == "__main__":
    main()
