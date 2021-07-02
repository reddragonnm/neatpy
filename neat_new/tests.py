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
        for _ in range(1000):
            Node.reset_test()
            i, o = random.randrange(1, 20), random.randrange(1, 20)

            Options.inputs = i
            Options.outputs = o
            b = Brain()

            self.check_num(i, o, b)
            self.check_valid(i, o, b)


class TestNode:
    def check_add_node(self, b):
        n1 = b._nodes.copy()
        b._add_node()
        n2 = b._nodes.copy()

        node_id = list(n2 - n1)[0]
        conn = None

        for c in Node._history:
            if Node._history[c] == node_id:
                conn = c

        assert b._conns[conn]['enabled'] == False
        assert b._conns.get((conn[0], node_id)) is not None
        assert b._conns.get((node_id, conn[1])) is not None
        assert len(n1) + 1 == len(n2)

    def test_main(self):
        for _ in range(1000):
            i, o = random.randrange(1, 5), random.randrange(1, 5)
            Node.reset_test()

            Options.inputs = i
            Options.outputs = o
            b = Brain()

            self.check_add_node(b)
