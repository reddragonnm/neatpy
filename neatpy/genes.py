import random
import enum

class NodeState(enum.Enum):
    """Enum of all possible node states
    - input
    - hidden
    - output
    - bias
    """
    input = 'input'
    hidden = 'hidden'
    output = 'output'
    bias = 'bias'

class Node:
    def __init__(self, node_id, state, x, y):
        """Node gene

        Args:
            node_id (int): ID of the node
            state (NodeState): Enum of the node state
            x (float): X position of the node
            y (float): Y position of the node
        """
        self.id = node_id
        self.state = state

        self.x = x
        self.y = y

        self.val = 0

class Connection:
    def __init__(self, fr, to, innov, weight=None):
        """Connection gene

        Args:
            fr (int): ID of the input node
            to (int): ID of the output node
            innov (int): Innovation number of the connection
            enabled (bool, optional): Is the connection enabled. Defaults to True.
            weight (float, optional): If the weight is None then a random weight between -1 and 1 is chosen. Defaults to None.
        """
        self.fr = fr
        self.to = to

        self.weight = weight or random.uniform(-1, 1)
        self.enabled = True
        self.innov = innov