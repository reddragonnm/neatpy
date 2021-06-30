import unittest
import random

import neat


class TestGen(unittest.TestCase):
    def gen(self, i, o):
        neat.Options.inputs = i
        neat.Options.outputs = o

        b = neat.Brain()

        self.assertEqual(len(b._nodes), i + o + 1)
        self.assertEqual(len(b._conns), (i + 1) * o)

    def test_main(self):
        for _ in range(10):
            self.gen(random.randrange(1, 20), random.randrange(1, 20))


if __name__ == '__main__':
    unittest.main()
