#
# Largely based on:
# https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
#

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, InputLayer
from keras.models import Sequential

from myenv import *


def create_model(env: MyEnv) -> Sequential:
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, env.num_states_)))
    model.add(Dense(36, activation='sigmoid'))
    model.add(Dense(env.num_actions_, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.summary()
    return model


def onehot(num_states: int, state: int) -> np.ndarray:
    return np.identity(num_states)[state:state + 1]


def predict(model, num_states, state) -> int:
    return int(np.argmax(model.predict(onehot(num_states, state))))


def predict_value(model, num_states, state):
    return np.max(model.predict(onehot(num_states, state)))


def q_learning(env, num_episodes=1000, plot=False):
    # create the keras model
    model = create_model(env)
    # now execute the q learning
    y = 0.95
    eps = 0.5
    decay_factor = 0.999
    r_avg_list = []
    num_states: int = env.num_states_
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
                a = predict(model, num_states, s)
            new_s, r, done, _ = env.step(a)
            target = r + y * predict_value(model, num_states, new_s)
            target_vec = model.predict(onehot(num_states, s))[0]
            target_vec[a] = target
            model.fit(onehot(num_states, s),
                      target_vec.reshape(-1, num_actions),
                      epochs=1, verbose=0)
            s = new_s
            r_sum += r
        r_avg_list.append(r_sum / num_episodes)

    if plot:
        plt.plot(r_avg_list)
        plt.ylabel('Average reward per game')
        plt.xlabel('Number of games')
        plt.show()

    strategy = [
        np.argmax(model.predict(np.identity(num_states)[i:i + 1])[0])
        for i in range(num_states)
    ]
    for i in range(num_states):
        print("State {:<12s} -> {:<10s} {}".format(
            env.states.name(i), action_dict[strategy[i]],
            model.predict(np.identity(num_states)[i:i + 1])))
    print()
    return strategy


def main():
    env = MyEnv(debug=False)
    strategy = q_learning(env, 100)

    done = False
    state = env.reset(debug=True)
    while not done:
        a = env.decide(state, strategy)
        new_state, r, done, _ = env.step(a)

    # table = q_learning_with_table(env)
    # print(table)
    # table_greedy = eps_greedy_q_learning_with_table(env)
    # print(table_greedy)


main()
