import enum

class NodeState(enum.Enum):
    input = 'input'
    hidden = 'hidden'
    output = 'output'
    bias = 'bias'

class Node:
    def __init__(self, node_id, state, x, y):
        self.id = node_id
        self.state = state

        self.x = x
        self.y = y

        self.val = 0

        