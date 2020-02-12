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

import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
import _util as u
import _model_601_A as m  # linear Regression 정상
# import _model_601_B as m  # 3단 Neural Network : 비정상 -> 학습 X
                            # 1회, leraning Rate 0.1로는 3단 NN의 학습이 안된다.
                            # 그러나... 이게 학습의 가치가 있는가?

# Command창에서 색깔 표시
cr.init(autoreset=True)

# 환경 생성 후 초기화
env = gym.make('FrozenLake-v0')  # 환경 생성
state = env.reset()  # 환경 초기화
env.render()  # Game board 그리기

# 변수 초기화
num_episodes = 2000     # 반복 횟수
dis = 0.99              # discount factor
rList = []              # 각 Episode의 리워드(Goal 도착 여부)를 저장하는 list
local_cost = []         # 각 Episode의 cost를 저장하는 list
Q = np.zeros(shape=[16,4], dtype=float)
key = ''


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# episodes만큼 반복
for i in range(num_episodes):
    '''
    2) env를 초기화 하고 현재 s(state)를 관찰
    '''
    state = env.reset()
    done = False
    cost_sum = 0
    count = 0

    '''
    3) Dummy Q learning Algorithm
    '''
    while not done:
        count += 1
        # 3-1) a(action)을 선택하고
        Q_pred = sess.run(m.Y_,
                          feed_dict={m.X: u.one_hot(state)})

        select_action = 'E_GREEDY'
        # [A] Dummy Q learning Algorithm (성공률 : 약 5%)
        if select_action == 'DUMMY':
            pass
        # [B] add random noise (성공률 : 약 20%)
        elif select_action == 'RANDOM_NOISE':
            action = np.argmax(Q_pred +
                               np.random.randn(1, env.action_space.n) /
                               (i + 1))  # random noise 효과를 점점 낮춤
                                         # 효과를 점점 낮추지 않으면 성공률 : 2%
        # [C] e-greedy (성공률 : 약 50%)
        elif select_action == 'E_GREEDY':
            e = 1. / ((i / 50) + 10)  # 한계값을 점점 낮춤
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_pred)
        else:
            print("Select Action Fail!!!")
        # 3-2) 실행하고 r(reward)를 즉시 받음
        new_state, reward, done, info = env.step(action)

        # 3-3) Q table을 갱신 Q(s,a) = r + dis * maxQ(s', a')
        # [C] Q learning with learing rate and discount factor
        if done:    # reward
            Q_pred[0, action] = reward
        else:       # reward + dis * maxQ(s', a')
            new_Q_pred = sess.run(m.Y_,
                                  feed_dict={m.X: u.one_hot(new_state)})
            Q_pred[0, action] = reward + dis * np.max(new_Q_pred)

        cost, _ = sess.run([m.cost, m.train],
                           feed_dict={m.X: u.one_hot(state),
                                      m.Y: Q_pred})

        # 3-4) s(현재 상태)를 s'(a 실행에 의한 다음 상태)로 변환(s = s')
        state = new_state
        cost_sum += cost

    # 각 episode의 결과 저장 및 Q 상태 출력
    rList.append(reward)
    local_cost.append(cost_sum/count)

    if i % 100 == 0:
        print('num_episodes = {:4d}, cost = {:7.5f} '
              .format(i, cost_sum/count))
        for j in range(16):
            Q[j] = sess.run(m.Y_, feed_dict={m.X: u.one_hot(j)})
        u.print_Q(Q) 

print("Success rate: " + str(sum(rList) / num_episodes) + "%")

for i in range(16):
    Q[i] = sess.run(m.Y_, feed_dict={m.X: u.one_hot(i)})
u.print_Q(Q)    
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
