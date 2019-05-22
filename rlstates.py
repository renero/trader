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

    def id(self, *sub_states):
        assert len(
            sub_states) == self.nr_substates, \
            'Incorrect nr. of states. Read {}, should be {}'.format(
                len(sub_states), self.nr_substates)
        s = '_'.join(sub_states)
        assert s in self.state,\
            'The state ({}) does NOT exist in RL set of states'.format(s)
        return self.state[s]

    def name(self, id):
        assert id in self.ivd, \
            'State ID {} not in list (0..{}).'.format(
                id, self.max_id - 1)
        return self.ivd[id]


# value_states = {'EVEN', 'WIN', 'LOOSE'}
# forecast_states = {'EVEN', 'WIN', 'LOOSE'}
#
# states = RLStates([value_states, forecast_states]).combine()
# print(states.state)
# print(states.id('EVEN', 'LOOSE'))
# print(states.name(7))
