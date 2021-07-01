import random


def new_conn(weight=None, enabled=True):
    if weight is None:
        weight = random.uniform(-1, 1)

    return {
        'weight': weight,
        'enabled': enabled
    }


class Options:
    inputs = None
    outputs = None


class Brain:
    def __init__(self, nodes=None, conns=None):
        self._nodes = nodes
        self._conns = conns

        self._fitness = 0

        if self._nodes is None:
            self.generate_network()

    def generate_network(self):
        self._nodes = set(i for i in range(
            Options.inputs + Options.outputs + 1))

        self._conns = dict()
        for i in range(Options.outputs):
            for j in range(Options.inputs + 1):
                self._conns[j, Options.inputs + 1 + i] = new_conn()
