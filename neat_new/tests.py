import random
from neat import Brain, Options, Node


class TestGen:
    def check_num(self, i, o, b):
        assert len(b._nodes) == i + o + 1
        assert len(b._conns) == (i + 1) * o

    def check_valid(self, i, o, b):
        for node_id in range(i + o + 1):
            assert node_id in b._nodes

        for conn in list(b._conns):
            assert 0 <= conn[0] <= i
            assert i < conn[1] <= i + o

    def test_main(self):
        for _ in range(10):
            i, o = random.randrange(1, 20), random.randrange(1, 20)

            Options.inputs = i
            Options.outputs = o
            b = Brain()

            self.check_num(i, o, b)
            self.check_valid(i, o, b)


class TestNode:
    def check_node_history(self):
        pass

    def test_main(self):
        self.check_node_history()
