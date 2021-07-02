import random


class Options:
    inputs = None
    outputs = None

    add_node_prob = 0.02
    add_conn_prob = 0.05


class NodeState:
    bias = 'bias'
    input = 'input'
    hidden = 'hidden'
    output = 'output'


class Node:
    pos = {}
    node_id = 0
    history = {}

    @staticmethod
    def reset_test():
        Node.node_id = 0
        Node.history = {}

        Node.setup()

    @staticmethod
    def setup():
        Node.node_id = Options.inputs + Options.outputs + 1

        for i in range(Options.inputs + 1):
            Node.pos[i] = 0, 0

        for i in range(Options.outputs):
            Node.pos[Options.inputs + 1 + i] = 0, 1

    @staticmethod
    def get_node_id(conn):
        if Node.history.get(conn) is None:
            Node.history[conn] = Node.node_id
            Node.node_id += 1

        return Node.history[conn]

    @staticmethod
    def get_state(idx):
        if idx == 0:
            return NodeState.bias
        elif idx <= Options.inputs:
            return NodeState.input
        elif idx <= Options.inputs + Options.outputs:
            return NodeState.output

        return NodeState.hidden


def new_conn(weight=None, enabled=True):
    if weight is None:
        weight = random.uniform(-1, 1)

    return {
        'weight': weight,
        'enabled': enabled
    }


class Brain:
    def __init__(self, nodes=None, conns=None):
        self._nodes = nodes
        self._conns = conns

        self._fitness = 0

        if self._nodes is None:
            self._gen_network()

    def _gen_network(self):
        self._nodes = set(i for i in range(
            Options.inputs + Options.outputs + 1))

        self._conns = dict()
        for i in range(Options.outputs):
            for j in range(Options.inputs + 1):
                self._conns[j, Options.inputs + 1 + i] = new_conn()

    def _add_node(self):
        conn = random.choice(
            [i for i in self._conns if i[0] != 0 and self._conns[i]['enabled']])

        self._conns[conn]['enabled'] = False

        node_id = Node.get_node_id(conn)
        self._nodes.add(node_id)

        self._conns[conn[0], node_id] = new_conn(1)
        self._conns[node_id, conn[1]] = new_conn(self._conns[conn]['weight'])

    def _add_conn(self):
        pass


if __name__ == '__main__':
    Options.inputs = 2
    Options.outputs = 1
    Node.setup()

    b = Brain()

    print(b._nodes, b._conns, Node.history)
    b._add_node()
    print()
    print(b._nodes, b._conns, Node.history)
