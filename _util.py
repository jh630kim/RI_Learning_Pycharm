# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:40:22 2019

@author: R570
"""

import msvcrt
import numpy as np
import random as pr


class _Getch:
    # Key 입력 및 방향키 처리
    def __call__(self):
        key = msvcrt.getch()
        if key == b'\xe0':
            key = msvcrt.getch()
            key = b'_' + key
        return key


class _KB_Hit:
    # Key 입력 및 방향키 처리
    def __call__(self):
        key = msvcrt.kbhit()
        return key


def rargmax(vector):
    # https://gist.github.com/stober/1943451
    # 가능한 최대 Index 중에서 임의로 선택
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)


def print_Q(Q):
    # Q = Q.reshape([16,4])
    print(" 1L   1D   1R   1U  |",
          " 2L   2D   2R   2U  |",
          " 3L   3D   3R   3U  |",
          " 4L   4D   4R   4U  |")

    Q1 = Q.reshape(64)
    for i in range(4):
        for j in range(16):
            print('{:.2f} '.format(Q1[i*16+j]), end='')
            if (j+1) % 4 == 0:
                print('| ', end='')
        print('')


def one_hot(x):
    return np.identity(16)[x:x+1]


inkey = _Getch()
kbhit = _KB_Hit()

# MACROS
# env = gym.make('FrozenLake-v0')
'''
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
'''

# env = gym.make('CartPole-v0')
LEFT = 0
RIGHT = 1
DOWN = 1
UP = 3

arrow_keys = {
        b'_H': UP,
        b'_P': DOWN,
        b'_M': RIGHT,
        b'_K': LEFT}

if __name__ == '__main__':
    while(True):
        key = ''

        if msvcrt.kbhit():  # KB 입력이 없으면 pass
            key = inkey()   # KB 입력이 들어올 때 까지 기다린다
            print(key)

        if key in arrow_keys.keys():
            print(arrow_keys[key])

        if key == b'q':
            exit()
