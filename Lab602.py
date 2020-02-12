# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:21:02 2019

@author: user
"""
import colorama as cr

import gym
import _util as u


# Command창에서 색깔 표시
cr.init(autoreset=True)

env = gym.make('CartPole-v0')
# env = gym.make('FrozenLake-v0')
env.reset()
random_episodes = 0
reward_sum = 0
key = b'_K'

while random_episodes < 5:
    env.render()
    # if u.kbhit():
    key = u.inkey()
    if key == b'q':
        print("Game aborted!")
        break

    action = u.arrow_keys[key]  # 0-Left, 1-Down, 2-Right, 3-Up
    # action = env.action_space.sample()
    print(action)
    observation, reward, done, info = env.step(action)
    print(observation, reward, done, action)
    reward_sum += reward
    if done:
        random_episodes += 1
        print("Reward for this episode was:", random_episodes, reward_sum)
        reward_sum = 0
        env.reset()

env.close()
