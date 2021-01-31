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

    def calc_spawns(self):
        total = sum([s.average_fitness for s in self.species])
        for s in self.species:
            s.spawns_required = Options.population_size * s.average_fitness / total

    def speciate(self):
        for brain in self.pool:
            added = False

            for s in self.species:
                if s.same_species(brain):
                    s.pool.append(brain)
                    added = True
                    break

            if not added:
                self.species.append(Species(self.next_species_id, brain))
                self.next_species_id += 1

        self.species = [sp for sp in self.species if len(sp.pool) > 0]

    def calc_spawns(self):
        total = sum([s.average_fitness for s in self.species])
        for s in self.species:
            s.spawns_required = Options.population_size * s.average_fitness / total

    def reproduce(self):
        for s in self.species:
            k = max(1, int(round(len(s.pool) * Options.survival_rate)))
            pool = s.pool[:k]
            s.pool[:] = []

            if Options.species_elitism:
                s.pool.append(s.leader)

            while len(s.pool) < s.spawns_required:                
                g1 = self.tournament_selection(pool)

                if random.random() < Options.crossover_rate:
                    g2 = self.tournament_selection(pool)
                    child = self.crossover(g1, g2, self.next_genome_id)
                    self.next_genome_id += 1
                else:
                    child = copy.copy(g1)

                child.mutate()
                s.pool.append(child)

        self.pool[:] = []
        for s in self.species:
            self.pool.extend(s.pool)
            s.purge()

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
            s.pool.sort(key=lambda x: x.fitness, reverse=True)
            s.leader = s.pool[0]

            if s.leader.fitness > s.max_fitness:
                s.generations_not_improved = 0
            else:
                s.generations_not_improved += 1
            s.max_fitness = s.leader.fitness

            # adjust fitness
            sum_fitness = 0.0
            for m in s.pool:
                fitness = m.fitness
                sum_fitness += fitness
                # boost young species
                if s.age < Options.young_age_threshhold:
                    fitness *= Options.young_age_fitness_bonus
                # punish old species
                if s.age > Options.old_age_threshold:
                    fitness *= Options.old_age_fitness_penalty
                # apply fitness sharing to adjusted fitnesses
                m.adjusted_fitness = fitness/len(s.pool)

            s.average_fitness = sum_fitness/len(s.pool)

    def change_compatibility_threshold(self):
        if len(self.species) < Options.target_species:
            Options.compatibility_threshold *= 0.95
        elif len(self.species) > Options.target_species:
            Options.compatibility_threshold *= 1.05

    def epoch(self, evaluate):
        evaluate(self.pool)
        self.sort_pool()

        self.speciate()
        self.change_compatibility_threshold()
        self.adjust_fitnesses()

        self.calc_spawns()
        self.species = [s for s in self.species if s.stagnation < Options.dropoff_age and s.spawns_required > 0]
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
