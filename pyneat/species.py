class Species:
    def __init__(self, species_id, member):
        self.leader = member
        
        self.pool = [member]
        self.id = species_id
        
        self.age = 0
        self.stagnation = 0

        self.spawns_required = 0

        self.max_fitness = 0.0
        self.average_fitness = 0.0

    def purge(self):
        self.age += 1
        self.pool = []
