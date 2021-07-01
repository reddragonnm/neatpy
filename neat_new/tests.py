import unittest
import random

from neat import Brain, Options


class TestGen(unittest.TestCase):
    def check_num(self, i, o, b):
        self.assertEqual(len(b._nodes), i + o + 1)
        self.assertEqual(len(b._conns), (i + 1) * o)

    def check_valid(self, i, o, b):
        for conn in list(b._conns):
            self.assertLessEqual(conn[0], i)
            self.assertGreater(conn[1], i)

    def test_main(self):
        for _ in range(10):
            i, o = random.randrange(1, 20), random.randrange(1, 20)

            Options.inputs = i
            Options.outputs = o
            b = Brain()

            self.check_num(i, o, b)
            self.check_valid(i, o, b)


if __name__ == '__main__':
    unittest.main()
