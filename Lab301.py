# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:27:41 2019

@author: R570

<<< Dummy Q-learning algorithm >>>
1) 모든 s(state)에 대해서 Q(s,a)를 0으로 초기화
2) 현재 s를 관찰
3) 반복
3-1) a(action)을 선택하고 실행
3-2) r(reward)를 즉시 받음
3-3) Q를 갱신 Q(s,a) = r + maxQ(s', a')
    # r - Goal에서 주는 reward
    # maxQ(s', a') - 목적지까지 길을 아는 state에서 주는 reward
3-4) s(현재 상태)를 s'(a 실행에 의한 다음 상태)로 변환(s = s')
"""

import colorama as cr

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import _util as u

# Command창에서 색깔 표시
cr.init(autoreset=True)

# 새로운 Frozen Lake를 등록함
register(
        id='FrozenLake-v3',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False})

# 환경 생성 후 초기화
env = gym.make("FrozenLake-v3")  # 환경 생성
state = env.reset()  # 환경 초기화
env.render()  # Game board 그리기

# 변수 초기화
num_episodes = 200  # 반복 횟수
rList = []          # 각 Episode의 리워드(Goal 도착 여부)를 저장하는 list
key = ''

'''
1) 모든 s(state)에 대해서 Q(s,a)를 0으로 초기화
'''
Q = np.zeros([env.observation_space.n, env.action_space.n])

# episodes만큼 반복
for i in range(num_episodes):
    '''
    2) env를 초기화 하고 현재 s(state)를 관찰
    '''
    state = env.reset()
    done = False

    '''
    3) Dummy Q learning Algorithm
    '''
    while not done:
        # 3-1) a(action)을 선택하고
        action = u.rargmax(Q[state, :])

        # 3-2) 실행하고 r(reward)를 즉시 받음
        new_state, reward, done, info = env.step(action)

        # 3-3) Q table을 갱신 Q(s,a) = r + maxQ(s', a')
        Q[state, action] = reward + np.max(Q[new_state, :])

        # 3-4) s(현재 상태)를 s'(a 실행에 의한 다음 상태)로 변환(s = s')
        state = new_state

    # 각 episode의 결과 저장 및 Q 상태 출력
    rList.append(reward)
    print(" 1L 1D 1R 1U 2L 2D 2R 2U 3L 3D 3R 3U 4L 4D 4R 4U")
    print(Q.reshape([4, 16]))

    # 각 episode 후 Q 상태 Check
    if key != b'q':
        key = u.inkey()

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(" 1L 1D 1R 1U 2L 2D 2R 2U 3L 3D 3R 3U 4L 4D 4R 4U")
print(Q.reshape([4, 16]))
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
