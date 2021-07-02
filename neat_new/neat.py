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
    _node_id = 0
    _history = {}

    @staticmethod
    def reset_test():
        Node._node_id = 0
        Node._history = {}

    @staticmethod
    def set_node_id(node_id):
        Node._node_id = max(Node._node_id, node_id)

    @staticmethod
    def get_node_id(conn):
        if Node._history.get(conn) is None:
            Node._history[conn] = Node._node_id
            Node._node_id += 1

        return Node._history[conn]

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

        Node.set_node_id(max(self._nodes) + 1)

    def _add_node(self):
        conn = random.choice(
            [i for i in self._conns if i[0] != 0 and self._conns[i]['enabled']])

        self._conns[conn]['enabled'] = False

        node_id = Node.get_node_id(conn)
        self._nodes.add(node_id)

        self._conns[conn[0], node_id] = new_conn(1)
        self._conns[node_id, conn[1]] = new_conn(self._conns[conn]['weight'])


if __name__ == '__main__':
    Options.inputs = 2
    Options.outputs = 1

    b = Brain()

    print(b._nodes)
    b._add_node()
    print(b._nodes)
