#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author:yiluzhang
"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable


# 主循环
def update():
    step_num = 0
    is_hell = False
    for episode in range(10000):
        #print(step_num)
        #print(is_hell)
        if ((0 < step_num < 7) and (not is_hell)):
            print(episode)
        step_num = 0
        # initial observation
        # state，当前位置是指红色正方形的坐标，如起点正方形对角坐标分别为 (5，5) 和 (35, 35)
        observation = env.reset()

        # for test
        # if(episode > 20):
        #     RL.print_q_table()

        while True:
            # fresh env.require continuous fresh to keep dynamic
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            if (observation_ == 'terminal' and reward == -1):
                is_hell = True
            else:
                is_hell = False
            step_num += 1

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()  # after operating 100 times game over and destroy the environment


# 相关注释网站https://cloud.tencent.com/developer/article/1148483
# https://blog.csdn.net/duanyajun987/article/details/78614902
if __name__ == "__main__":
    env = Maze()  # 使用tkinter 初始化和创建环境
    RL = QLearningTable(actions=list(range(env.n_actions)))  # 定义强化学习类，初始化相关参数

    env.after(100, update)  # call update() after 100ms
    # update()
    env.mainloop()  #
    # update()  #放在后面就用不了
