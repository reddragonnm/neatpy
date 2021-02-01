import copy
import random

from .brain import Brain
from .species import Species
from .options import Options

class Population:
    def __init__(self):
        self.pool = [Brain(i) for i in range(Options.population_size)]
        self.species = []

        self.best = self.pool[0]

        self.gen = 0
        self.next_genome_id = len(self.pool)
        self.next_species_id = 0   

    def evaluate(self, eval_func, num_generations=float('inf')):
        while True:
            self.epoch(eval_func)

            print(self.best.fitness, len(self.species), Options.compatibility_threshold, len(self.pool))

            if self.best.fitness > Options.fitness_threshold:
                return self.best, True
            elif self.gen >= num_generations:
                return self.best, False

    def speciate(self):
        for brain in self.pool:
            added = False

            for sp in self.species:
                if sp.same_species(brain):
                    sp.pool.append(brain)
                    added = True
                    break

            if not added:
                self.species.append(Species(self.next_species_id, brain))
                self.next_species_id += 1

        self.species = [sp for sp in self.species if len(sp.pool) > 0]

    def calc_spawns(self):
        total = sum([sp.average_fitness for sp in self.species])
        for sp in self.species:
            sp.spawns_required = Options.population_size * sp.average_fitness / total

    def reproduce(self):
        new_pop = []
        for s in self.species:
            new_pool = []

            if Options.species_elitism:
                new_pool.append(s.best)

            while len(new_pool) < s.spawns_required:                
                brain1 = self.tournament_selection(s.pool)

                if random.random() < Options.crossover_rate:
                    brain2 = self.tournament_selection(s.pool)
                    child = self.crossover(brain1, brain2, self.next_genome_id)
                    self.next_genome_id += 1
                else:
                    child = copy.copy(brain1)

                child.mutate()
                new_pool.append(child)

            new_pop.extend(new_pool)
            s.purge()

        self.pool = new_pop

        while len(self.pool) < Options.population_size:
            genome = Brain(self.next_genome_id)
            self.pool.append(genome)
            self.next_genome_id += 1

    def sort_pool(self):
        self.pool.sort(key=lambda x: x.fitness, reverse=True)

        if self.best.fitness < self.pool[0].fitness:
            self.best = self.pool[0]

    def adjust_fitnesses(self):
        for s in self.species:
            s.make_leader()
            s.adjust_fitnesses()

    def change_compatibility_threshold(self):
        if len(self.species) < Options.target_species:
            Options.compatibility_threshold *= 0.95

        elif len(self.species) > Options.target_species:
            Options.compatibility_threshold *= 1.05

    def reset_and_kill(self):
        new_species = []
        
        for sp in self.species:
            if sp.stagnation > Options.dropoff_age or sp.spawns_required == 0:
                continue

            sp.pool = sp.pool[:max(1, round(len(sp.pool) * Options.survival_rate))]
            new_species.append(sp)

        self.species = new_species

    def epoch(self, evaluate):
        evaluate(self.pool)

        self.sort_pool()

        self.speciate()

        if Options.dynamic_compatibility_threshold:
            self.change_compatibility_threshold()

        self.adjust_fitnesses()
        self.calc_spawns()
        
        self.reset_and_kill()
        self.reproduce()        

        self.gen += 1

    @staticmethod
    def tournament_selection(genomes):
        champion = genomes[0]
        for _ in range(min(len(genomes), Options.tries_tournament_selection)):
            g = random.choice(genomes)
            if g.fitness > champion.fitness:
                champion = g
        return champion

    def crossover(self, mum, dad, baby_id=None):
        n_mum = len(mum.connections)
        n_dad = len(dad.connections)

        if mum.fitness == dad.fitness:
            if n_mum == n_dad:
                better = random.choice([mum, dad])
            elif n_mum < n_dad:
                better = mum
            else:
                better = dad
        elif mum.fitness > dad.fitness:
            better = mum
        else:
            better = dad

        baby_nodes = []   # node genes
        baby_connections = []     # conn genes

        # iterate over parent genes
        i_mum = i_dad = 0
        node_ids = set()
        while i_mum < n_mum or i_dad < n_dad:
            mum_gene = mum.connections[i_mum] if i_mum < n_mum else None
            dad_gene = dad.connections[i_dad] if i_dad < n_dad else None
            selected_gene = None
            if mum_gene and dad_gene:
                if mum_gene.innov == dad_gene.innov:
                    selected_gene, selected_genome = random.choice([(mum_gene, mum), (dad_gene, dad)])

                    i_mum += 1
                    i_dad += 1
                elif dad_gene.innov < mum_gene.innov:
                    if better == dad:
                        selected_gene = dad.connections[i_dad]
                        selected_genome = dad
                    i_dad += 1
                elif mum_gene.innov < dad_gene.innov:
                    if better == mum:
                        selected_gene = mum_gene
                        selected_genome = mum
                    i_mum += 1
            elif mum_gene == None and dad_gene:
                if better == dad:
                    selected_gene = dad.connections[i_dad]
                    selected_genome = dad
                i_dad += 1
            elif mum_gene and dad_gene == None:
                if better == mum:
                    selected_gene = mum_gene
                    selected_genome = mum
                i_mum += 1

            if selected_gene != None:
                # inherit conn
                baby_connections.append(copy.copy(selected_gene))

                # inherit nodes
                if not selected_gene.fr in node_ids:
                    node = selected_genome.get_node(selected_gene.fr)
                    if node != None:
                        baby_nodes.append(copy.copy(node))
                        node_ids.add(selected_gene.fr)

                if not selected_gene.to in node_ids:
                    node = selected_genome.get_node(selected_gene.to)
                    if node != None:
                        baby_nodes.append(copy.copy(node))
                        node_ids.add(selected_gene.to)

        for node in mum.nodes:
            if not node.id in node_ids:
                baby_nodes.append(copy.copy(node))
                node_ids.add(node.id)

        s = list(set([l.enabled for l in baby_connections]))
        if len(s) == 1 and not s[0]:
            random.choice(baby_connections).enabled = True

        return Brain(baby_id, baby_nodes, baby_connections)
