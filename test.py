import unittest
import neat1
import neat2

xor_inp = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_out = [0, 1, 1, 0]

p1 = neat1.Population()
p2 = neat2.Population()

neat1.Options.set_options(2, 1, 150, 3.9, weight_mutate_prob=0.5,
                          add_node_prob=0.005, add_conn_prob=0.1)

neat2.Options.set_options(2, 1, 150, 3.9, weight_mutate_prob=0.5,
                          add_node_prob=0.005, add_conn_prob=0.1)


class TestSum(unittest.TestCase):
    def test_neat1_xor(self):

        while p1.best.fitness < 3.9:
            for nn in p1.pool:
                nn.fitness = 4

                for xi, xo in zip(xor_inp, xor_out):
                    output = nn.predict(xi)[0]
                    nn.fitness -= (output - xo) ** 2

            p1.epoch()
            print(p1)

        self.assertEqual(p1.gen, 141)


if __name__ == '__main__':
    unittest.main()
