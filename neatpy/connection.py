import random

class Connection:
    def __init__(self, fr, to, innov, enabled=True, weight=None):
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
        self.enabled = enabled
        self.innov = innov