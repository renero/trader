#!/usr/bin/env python
# coding: utf-8

s1 = [ 'a', 'b', 'c' ]
s2 = [ 'd', 'e', 'f' ]
s3 = [ 'g', 'h', 'i' ]


class RLStates:
    state_list = []
    states = []
    state = {}
    size = 0
    max_id = 0
    
    def __init__(self, list_of_states):
        self.size = len(list_of_states)
        for s in list_of_states:
            self.state_list.append(s)
        return
    
    def combine(self):
        self.states = [state for state in product(*[s1, s2, s3])]
        for i, t in enumerate(self.states):
            key = '_'.join(t)
            self.state[key] = i
        self.ivd = {v: k for k, v in self.state.items()}
        self.max_id = len(self.state)
        return self
    
    def id(self, *sub_states):
        assert len(sub_states) == self.size,            'Incorrect nr. of states. Read {}, should be {}'.format(
                len(sub_states), self.size)
        s = '_'.join(sub_states)
        return self.state[s]
    
    def name(self, id):
        assert id in self.ivd,            'State ID {} not in list (0..{}).'.format(id,self.max_id-1)
        return self.ivd[id]


states = States([s1, s2, s3]).combine()
states.id('a','e','i')
states.name(22)

