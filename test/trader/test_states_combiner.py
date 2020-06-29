import sys
from unittest import TestCase

from states_combiner import StatesCombiner
from utils.my_dict import MyDict


class TestStatesCombiner(TestCase):

    params = MyDict()
    params.log = sys.stdout
    params.states_list = [['A', 'B'], ['C', 'D']]
    states = StatesCombiner(params)

    def test_combine(self):
        self.assertIsInstance(self.states, StatesCombiner)
        self.assertEqual(self.states.max_id, 4)
        self.assertEqual(len(self.states.state), 4)

        self.assertEqual(self.states.state['A_C'], 0)
        self.assertEqual(self.states.state['A_D'], 1)
        self.assertEqual(self.states.state['B_C'], 2)
        self.assertEqual(self.states.state['B_D'], 3)

        self.assertEqual(self.states.ivd[0], 'A_C')
        self.assertEqual(self.states.ivd[1], 'A_D')
        self.assertEqual(self.states.ivd[2], 'B_C')
        self.assertEqual(self.states.ivd[3], 'B_D')

    def test_get_id(self):
        self.assertEqual(self.states.get_id(*['A', 'C']), 0)
        self.assertEqual(self.states.get_id(*['A', 'D']), 1)
        self.assertEqual(self.states.get_id(*['B', 'C']), 2)
        self.assertEqual(self.states.get_id(*['B', 'D']), 3)

        try:
            self.states.get_id(*['A', 'B'])
        except Exception as e:
            self.assertIsInstance(e, AssertionError)

    def test_name(self):
        self.assertEqual(self.states.name(0), 'A_C')
        self.assertEqual(self.states.name(1), 'A_D')
        self.assertEqual(self.states.name(2), 'B_C')
        self.assertEqual(self.states.name(3), 'B_D')

        try:
            self.states.name(5)
        except Exception as e:
            self.assertIsInstance(e, AssertionError)
