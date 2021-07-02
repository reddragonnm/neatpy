import random


class Options:
    inputs = None
    outputs = None


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
    def get_node_id(conn):
        if Node._history.get(conn) is None:
            Node._history[conn] = Node._node_id
            Node._node_id += 1

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
