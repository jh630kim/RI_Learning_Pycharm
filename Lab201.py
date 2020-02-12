# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:27:41 2019

@author: R570
"""
import colorama as cr

import gym
from gym.envs.registration import register
import _util as u

# Command창에서 색깔 표시
cr.init(autoreset=True)

# 새로운 Frozen Lake를 등록함
register(
        id='FrozenLake-v3',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False})

env = gym.make("FrozenLake-v3")  # 환경 생성
state = env.reset()  # 환경 초기화
env.render()  # Game board 그리기

while True:
    # Keyboard에서 Action(방향키 입력)
    key = u.inkey()
    if key not in u.arrow_keys.keys():
        print("Game aborted!")
        break

    action = u.arrow_keys[key]  # 0-Left, 1-Down, 2-Right, 3-Up
    # action = env.action_space.sample()  # sample 입력을 전달
    '''
    <<< gym의 핵심 idea >>>
    (1) Agent -> Environment: action을 주면
    (2) Environment -> Agent: state, reward, done, info를 돌려준다!!!
    '''
    state, reward, done, info = env.step(action)
    env.render()  # Game Board 그리기

    if done:
        print("Finished with reward", reward)
        break
