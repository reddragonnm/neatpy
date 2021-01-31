import random

class Connection:
    def __init__(self, fr, to, innov, enabled=True, weight=None):
        self.fr = fr
        self.to = to

        self.weight = weight or random.uniform(-1, 1)
        self.enabled = enabled
        self.innov = innov