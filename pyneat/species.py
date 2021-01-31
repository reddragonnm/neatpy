from .options import Options

class Species:
    def __init__(self, first_member, species_id):
        self.leader = first_member
        
        self.pool = [first_member]
        self.id = species_id
        
        self.stagnation = 0
        self.age = 0

        self.spawns_required = 0
        self.max_fitness = 0.0
        self.average_fitness = 0.0
