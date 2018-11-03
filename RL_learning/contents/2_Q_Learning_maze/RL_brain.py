#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:yiluzhang
"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    # learning_rate=0.01, reward_decay=0.9, e_greedy=0.9)
    # 学习率改为0.1，学习速度有明显改善
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list,number formation of action
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy   # the except rate
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)   # 使用pandas存数据，https://www.cnblogs.com/chenice/p/7257237.html

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection, 90 percent to choose best action
        if np.random.uniform() < self.epsilon:
            # choose best action
            # state_action = self.q_table.loc[observation, :]
            # # some actions may have the same value, randomly choose on in these actions
            # action = np.random.choice(state_action[state_action == np.max(state_action)].index)

            state_action = self.q_table.ix[observation, :]

            # for test
            # print(state_action)
            # print(state_action.index)

            # 同一个 state, 可能会有多个相同的 Q action value, 所以我们乱序一下
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            # print(state_action)
            action = state_action.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            # 发现下一步的回报比较大，说明更接近宝藏了
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update，用（学习率*回报）更新概率

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def print_q_table(self):
        print(self.q_table)

