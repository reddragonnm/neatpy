import random
from neat import Brain, Options, Node, NodeState


def boilerplate(func):
    def wrap():
        for _ in range(1000):
            i, o = random.randrange(1, 20), random.randrange(1, 20)

            Options.set_options(i, o)
            b = Brain()

            func(i, o, b)

    return wrap


@boilerplate
def test_num(i, o, b):
    assert len(b._nodes) == i + o + 1
    assert len(b._conns) == (i + 1) * o


@boilerplate
def test_valid(i, o, b):
    for node_id in range(i + o + 1):
        assert node_id in b._nodes

    for conn in list(b._conns):
        assert 0 <= conn[0] <= i
        assert i < conn[1] <= i + o


@boilerplate
def test_add_node(i, o, b):
    n1 = b._nodes.copy()
    b._add_node()
    n2 = b._nodes.copy()

    node_id = list(n2 - n1)[0]
    conn = None
    for c in Node.history:
        if Node.history[c] == node_id:
            conn = c

    assert Node.get_state(node_id) == NodeState.hidden
    assert b._conns[conn]['enabled'] == False

    assert b._conns.get((conn[0], node_id)) is not None
    assert b._conns.get((node_id, conn[1])) is not None

    assert len(n1) + 1 == len(n2)


@boilerplate
def test_node_pos(i, o, b):
    pass
