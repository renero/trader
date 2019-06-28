"""
This class is here to help with managing several states, resulting from
generating a cartesian product of sub_states. For example:

    # value_states = {'EVEN', 'WIN', 'LOSE'}
    # forecast_states = {'EVEN', 'WIN', 'LOSE'}
    #
    # states = RLStates([value_states, forecast_states])
    # print(states.state)
    # print(states.get_id('EVEN', 'LOSE'))
    # print(states.name(7))

"""
from itertools import product


class RLStates:
    state_list = []
    state = {}
    ivd = {}
    nr_substates = 0
    max_id = 0

    def __init__(self, list_of_states):
        self.nr_substates = len(list_of_states)
        for s in list_of_states:
            self.state_list.append(s)
        self.combine()
        return

    def combine(self):
        states = [state for state in product(*self.state_list)]
        for i, t in enumerate(states):
            key = '_'.join(t)
            self.state[key] = i
        self.ivd = {v: k for k, v in self.state.items()}
        self.max_id = len(self.state)
        return self

    def get_id(self, *sub_states):
        assert len(
            sub_states) == self.nr_substates, \
            'Incorrect nr. of states. Read {}, should be {}'.format(
                len(sub_states), self.nr_substates)
        s = '_'.join(sub_states)
        assert s in self.state, \
            'The state ({}) does NOT exist in RL set of states'.format(s)
        return self.state[s]

    def name(self, state_id):
        assert state_id in self.ivd, \
            'State ID {} not in list (0..{}).'.format(
                state_id, self.max_id - 1)
        return self.ivd[state_id]

    @property
    def max_len(self):
        ml = -1
        for s in self.state.keys():
            if len(s) > ml:
                ml = len(s)
        return ml
