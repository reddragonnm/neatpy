import random
from neat import Brain, Options, Node, NodeState


def boilerplate(num=1000, run_self=1, node_range=20):
    def normal(func):
        def wrap():
            for x in range(num):
                i, o = random.randrange(
                    1, node_range), random.randrange(1, node_range)

                Options.set_options(i, o)
                b = Brain()

                for y in range(run_self):
                    func(i, o, b)

        return wrap
    return normal


@boilerplate()
def test_num(i, o, b):
    assert len(b._nodes) == i + o + 1
    assert len(b._conns) == (i + 1) * o


@boilerplate()
def test_valid(i, o, b):
    for node_id in range(i + o + 1):
        assert node_id in b._nodes

    for conn in list(b._conns):
        assert 0 <= conn[0] <= i
        assert i < conn[1] <= i + o


@boilerplate(num=100, run_self=10)
def test_add_node(i, o, b):
    n1 = b._nodes.copy()
    b._add_node()
    n2 = b._nodes.copy()

    node_id = list(n2 - n1)[0]
    conn = None
    for c in Node.history:
        if Node.history[c] == node_id:
            conn = c

    check_node_pos(node_id, conn)

    assert Node.get_state(node_id) == NodeState.hidden
    assert b._conns[conn]['enabled'] == False

    assert b._conns.get((conn[0], node_id)) is not None
    assert b._conns.get((node_id, conn[1])) is not None

    assert len(n1) + 1 == len(n2)


def check_node_pos(node_id, conn):
    assert Node.pos.get(node_id) is not None

    p = Node.pos[node_id]
    p1 = Node.pos[conn[0]]
    p2 = Node.pos[conn[1]]

    assert p[0] == (p1[0] + p2[0]) / 2
    assert p[1] == (p1[1] + p2[1]) / 2


@boilerplate(num=100, run_self=10)
def test_add_conn(i, o, b):
    b._add_node()

    conns = [
        (i, j)
        for i in b._nodes
        for j in b._nodes
        if b._valid_conn(i, j)
    ]

    old = b._conns.copy()
    b._add_conn()
    new = b._conns.copy()

    new_conn = list(set(new) - set(old))

    if len(new_conn) == 1:
        assert new_conn[0] in conns
    elif len(new_conn) == 0:
        check_list = []
        for c in conns:
            if old.get(c) is not None:
                # older connections that are enabled
                check_list.append(new.get(c) is not None and new[c]['enabled'])

        assert True in check_list
    else:
        assert False
