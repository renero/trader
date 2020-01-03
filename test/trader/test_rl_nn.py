from unittest import TestCase

from my_dict import MyDict
from rl_nn import RL_NN
from states_combiner import StatesCombiner


def do_nothing(*args, **kwargs):
    pass


class TestRL_NN(TestCase):
    params = MyDict()
    params.log = MyDict()
    params.log.debug = do_nothing
    params.states_list = [['A', 'B'], ['C', 'D'], ['E', 'F', 'G']]
    params.num_substates = 7
    params.display = None
    params.tensorboard = False
    states = StatesCombiner(params)
    NN = RL_NN(params, None)

    def test_onehot(self):
        enc = self.NN.onehot('A_C_E', self.states.state_list)
        self.assertListEqual(list(enc[0]), [1, 0, 1, 0, 1, 0, 0])

        enc = self.NN.onehot('B_D_G', self.states.state_list)
        self.assertListEqual(list(enc[0]), [0, 1, 0, 1, 0, 0, 1])

        # Negative case
        try:
            self.NN.onehot('A_B_C', self.states.state_list)
        except Exception as e:
            self.assertIsInstance(e, ValueError)
