#
# Largely based on:
# https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
#

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, InputLayer
from keras.models import Sequential

from myenv import *

value_states = ['EVEN', 'WIN', 'LOSE']
forecast_states = ['EVEN', 'WIN', 'LOSE']
share_states = ['HAVE', 'DONTHAVE']


def print_strategy(env, model, num_states, strategy):
    print('\nStrategy learned')
    strategy_string = "State {{:<{}s}} -> {{:<10s}} {{}}".format(
        env.states.max_len)
    for i in range(num_states):
        print(strategy_string.format(
            env.states.name(i), action_dict[strategy[i]],
            model.predict(np.identity(num_states)[i:i + 1])))
    print()


def plot_reinforcement(r_avg_list, plot: bool = False):
    if plot is False:
        return
    plt.plot(r_avg_list)
    plt.ylabel('Average reward per game')
    plt.xlabel('Number of games')
    plt.show()


def create_model(env: MyEnv) -> Sequential:
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, env.num_states_)))
    model.add(Dense(env.num_states_ * env.num_actions_, activation='sigmoid'))
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


def q_learning(env: MyEnv, num_episodes: int = 1000,
               plot: bool = False) -> list:
    """
    Implements the RL learning loop over an environment.

    :type env: MyEnv
    :type num_episodes: int
    :type plot: bool
    """
    # create the Keras model
    model = create_model(env)

    # now execute the q learning
    y = 0.95
    eps = 0.2
    decay_factor = 0.999
    r_avg_list = []
    num_states: int = env.num_states_
    num_actions = env.num_actions_

    # Loop over 'num_episodes'
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        if i % 100 == 0:
            print("Episode {} of {}".format(i, num_episodes))
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

    plot_reinforcement(r_avg_list, plot)
    strategy = [
        np.argmax(model.predict(np.identity(num_states)[i:i + 1])[0])
        for i in range(num_states)
    ]
    print_strategy(env, model, num_states, strategy)

    return strategy


def main():
    env = MyEnv([value_states, forecast_states, share_states], debug=True)
    strategy = q_learning(env, 10000)

    done = False
    state = env.reset(debug=True)
    while not done:
        a = env.decide(state, strategy)
        new_state, r, done, _ = env.step(a)


main()
