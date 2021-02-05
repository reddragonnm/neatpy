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
        