# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:21:02 2019

@author: user
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import _util as u
import _model_603_B as m

# 환경 생성 후 초기화
env = gym.make('CartPole-v0')  # 환경 생성
state = env.reset()  # 환경 초기화

# 변수 초기화
num_episodes = 2000     # 반복 횟수
dis = 0.9              # discount factor
reward_sum = 0
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
    state = env.reset()   # shape = (4,)  
    done = False
    cost_sum = 0
    reward_sum = 0
    count = 0

    '''
    3) Dummy Q learning Algorithm
    '''
    while not done:
        count += 1
        # 3-1) a(action)을 선택하고
        state = np.reshape(state, [1, m.input_size])  # shape = (1,4)
        Q_pred = sess.run(m.Y_,
                          feed_dict={m.X: state})

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
            e = 1. / ((i / 10) + 1)  # 한계값을 점점 낮춤
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
            Q_pred[0, action] = -100
        else:       # reward + dis * maxQ(s', a')
            new_state = np.reshape(new_state, [1, m.input_size])
            new_Q_pred = sess.run(m.Y_,
                                  feed_dict={m.X: new_state})
            Q_pred[0, action] = reward + dis * np.max(new_Q_pred)

        cost, _ = sess.run([m.cost, m.train],
                           feed_dict={m.X: state,
                                      m.Y: Q_pred})

        # 3-4) s(현재 상태)를 s'(a 실행에 의한 다음 상태)로 변환(s = s')
        state = new_state
        cost_sum += cost
        reward_sum += reward

    # 각 episode의 결과 저장 및 Q 상태 출력
    rList.append(reward_sum)
    local_cost.append(cost_sum/count)

    if i % 100 == 0:
        print('num_episodes = {:4d}, cost = {:7.5f} '
              .format(i, cost_sum/count))

    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        break

print("Success rate: " + str(sum(rList) / num_episodes) + "%")

observation = env.reset()
reward_sum = 0

while True:
    env.render()
    observation = np.reshape(observation, [1, m.input_size])
    Q_pred = sess.run(m.Y_, feed_dict={m.X: observation})
    action = np.argmax(Q_pred)
    
    observation, reward, done, _ = env.step(action)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break

plt.bar(range(len(rList)), rList, color="blue")
plt.show()

k = u.inkey()

env.close()
