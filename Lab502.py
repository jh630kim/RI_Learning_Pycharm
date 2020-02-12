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

# 환경 생성 후 초기화
env = gym.make('FrozenLake-v0')  # 환경 생성
state = env.reset()  # 환경 초기화
env.render()  # Game board 그리기
print("Input q(uit), n(ext), e(xit) or othre:")

# 변수 초기화
num_episodes = 2000     # 반복 횟수
dis = 0.99              # discount factor
learning_rate = 0.85    # learning rate
rList = []              # 각 Episode의 리워드(Goal 도착 여부)를 저장하는 list
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
        # [A] Dummy Q learning Algorithm (성공률 : 약 5%)
        select_action = 'RANDOM_NOISE'
        if select_action == 'DUMMY':
            action = u.rargmax(Q[state, :])
        # [B] add random noise (성공률 : 약 50%)
        elif select_action == 'RANDOM_NOISE':
            action = np.argmax(Q[state, :] +
                               np.random.randn(1, env.action_space.n) /
                               (i + 1))  # random noise 효과를 점점 낮춤
                                         # 효과를 점점 낮추지 않으면 성공률 : 2%
        # [C] e-greedy (성공률 : 약 15%)
        elif select_action == 'E_GREEDY':
            e = 1. / ((i // 100) + 1)  # 한계값을 점점 낮춤
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
        else:
            print("Select Action Fail!!!")
        # 3-2) 실행하고 r(reward)를 즉시 받음
        new_state, reward, done, info = env.step(action)

        # 3-3) Q table을 갱신 Q(s,a) = r + maxQ(s', a')
        # [A] Dummy Q learning
        '''
        Q[state, action] = reward + np.max(Q[new_state, :])
        '''
        # [B] Q learing with Discoute factor
        '''
        Q[state, action] = reward + dis * np.max(Q[new_state, :])
        '''
        # [C] Q learning with learing rate and discount factor
        Q[state, action] = (1 - learning_rate) * Q[state, action] +\
            learning_rate * (reward + dis * np.max(Q[new_state, :]))

        # 3-4) s(현재 상태)를 s'(a 실행에 의한 다음 상태)로 변환(s = s')
        state = new_state

        if key != b'q':
            if key != b'n':
                key = u.inkey()

        if key != b'q':
            env.render()  # Game board 그리기
            u.print_Q(Q)

        if key == b'e':
            exit()
    # 각 episode의 결과 저장 및 Q 상태 출력
    rList.append(reward)
    if i % 1000 == 0:
        print("num_episodes =", i)
        u.print_Q(Q)

    if key != b'q':
        print("num_episodes =", i)
        u.print_Q(Q)
        key = ''

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
u.print_Q(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
