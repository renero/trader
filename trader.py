#
# Largely based on:
# https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
#

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, InputLayer
from keras.models import Sequential

from myenv import *


def q_learning_with_table(env, num_episodes=500):
    q_table = np.zeros((env.num_states_, env.num_actions_))
    y = 0.95
    lr = 0.8
    for i in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            if np.sum(q_table[s, :]) == 0:
                # make a random selection of actions
                a = np.random.randint(0, env.num_actions_)
            else:
                # select the action with largest q value in state s
                a = np.argmax(q_table[s, :])
            new_s, r, done, _ = env.step(a)
            q_table[s, a] += r + lr * (
                    y * np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s
        print('\n--')
    return q_table


def eps_greedy_q_learning_with_table(env, num_episodes=500):
    q_table = np.zeros((env.num_states_, env.num_actions_))
    y = 0.95
    eps = 0.5
    lr = 0.8
    decay_factor = 0.999
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        done = False
        while not done:
            # select the action with highest cummulative reward
            if np.random.random() < eps or np.sum(q_table[s, :]) == 0:
                a = np.random.randint(0, env.num_actions_)
            else:
                a = np.argmax(q_table[s, :])
            # pdb.set_trace()
            new_s, r, done, _ = env.step(a)
            q_table[s, a] += r + lr * (
                    y * np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s
        print('\n--')  #
    return q_table


def create_model(env):
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, env.num_states_)))
    model.add(Dense(36, activation='sigmoid'))
    model.add(Dense(env.num_actions_, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.summary()
    return model


def onehot(num_states, state):
    return np.identity(num_states)[state:state + 1]


def q_learning(env, num_episodes=1000):
    # create the keras model
    model = create_model(env)
    # now execute the q learning
    y = 0.95
    eps = 0.5
    decay_factor = 0.999
    r_avg_list = []
    num_states = env.num_states_
    num_actions = env.num_actions_
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        if i % 100 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
        done = False
        r_sum = 0
        while not done:
            if np.random.random() < eps:
                a = np.random.randint(0, num_actions)
            else:
                a = np.argmax(
                    model.predict(np.identity(num_states)[s:s + 1]))
            new_s, r, done, _ = env.step(a)
            target = r + y * np.max(
                model.predict(np.identity(num_states)[new_s:new_s + 1]))
            target_vec = model.predict(np.identity(num_states)[s:s + 1])[0]
            target_vec[a] = target
            model.fit(np.identity(num_states)[s:s + 1],
                      target_vec.reshape(-1, num_actions),
                      epochs=1, verbose=0)
            s = new_s
            r_sum += r
        r_avg_list.append(r_sum / num_episodes)
    plt.plot(r_avg_list)
    plt.ylabel('Average reward per game')
    plt.xlabel('Number of games')
    plt.show()
    strategy = [
        np.argmax(model.predict(np.identity(num_states)[i:i + 1])[0])
        for i in range(num_states)
    ]
    for i in range(num_states):
        print("State {:<12s} -> {:<5s} {}".format(
            state_dict[i], action_dict[strategy[i]],
            model.predict(np.identity(num_states)[i:i + 1])))
    print()
    return strategy


env = MyEnv(debug=False)
strategy = q_learning(env, 1000)

done = False
state = env.reset(debug=True)
while not done:
    a = env.decide(state, strategy)
    new_state, r, done, _ = env.step(a)

# table = q_learning_with_table(env)
# print(table)
# table_greedy = eps_greedy_q_learning_with_table(env)
# print(table_greedy)
