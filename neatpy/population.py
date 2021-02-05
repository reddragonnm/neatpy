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
        self.brain_id = len(self.pool)
        self.species_id = 0   

    def evaluate(self, eval_func, num_generations=float('inf'), report=True):
        while True:
            eval_func(self.pool)
            self.epoch()

            if report:
                print(self)

            if self.best.fitness > Options.fitness_threshold:
                return self.best, True
            elif self.gen >= num_generations:
                return self.best, False

    def _speciate(self):
        for brain in self.pool:
            added = False

            for sp in self.species:
                if sp.same_species(brain):
                    sp.pool.append(brain)
                    added = True
                    break

            if not added:
                self.species.append(Species(self.species_id, brain))
                self.species_id += 1

        self.species = [sp for sp in self.species if len(sp.pool) > 0]

    def _calc_spawns(self):
        total = max(1, sum([sp.average_fitness for sp in self.species]))
        for sp in self.species:
            sp.spawns_required = Options.population_size * sp.average_fitness / total

    def _reproduce(self):
        new_pop = []
        for s in self.species:
            new_pool = []

            if Options.species_elitism:
                new_pool.append(s.best)

            while len(new_pool) < s.spawns_required:                
                brain1 = s.get_brain()

                if random.random() < Options.crossover_rate:
                    brain2 = s.get_brain()
                    child = self.crossover(brain1, brain2, self.brain_id)
                    self.brain_id += 1
                else:
                    child = copy.copy(brain1)

                child.mutate()
                new_pool.append(child)

            new_pop.extend(new_pool)
            s.purge()

        self.pool = new_pop

        while len(self.pool) < Options.population_size:
            self.pool.append(Brain(self.brain_id))
            self.brain_id += 1

    def _sort_pool(self):
        self.pool.sort(key=lambda x: x.fitness, reverse=True)

        assert self.pool[-1].fitness >= 0, "Cannot handle negative fitness values"

        if self.best.fitness < self.pool[0].fitness:
            self.best = self.pool[0]

    def _adjust_fitnesses(self):
        for s in self.species:
            s.make_leader()
            s.adjust_fitnesses()

    def _change_compatibility_threshold(self):
        if len(self.species) < Options.target_species:
            Options.compatibility_threshold *= 0.95

        elif len(self.species) > Options.target_species:
            Options.compatibility_threshold *= 1.05

    def _reset_and_kill(self):
        new_species = []
        
        for sp in self.species:
            if sp.stagnation > Options.dropoff_age or sp.spawns_required == 0:
                continue

            sp.cull()
            new_species.append(sp)

        self.species = new_species

    def epoch(self):
        self._sort_pool()
        self._speciate()

        if Options.dynamic_compatibility_threshold:
            self._change_compatibility_threshold()

        self._adjust_fitnesses()
        self._calc_spawns()
        
        self._reset_and_kill()
        self._reproduce()        

        self.gen += 1

    @staticmethod
    def crossover(mum, dad, baby_id=None):
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

        baby_nodes = []
        baby_connections = []

        i_mum = i_dad = 0
        node_ids = set()

        while i_mum < n_mum or i_dad < n_dad:
            mum_gene = mum.connections[i_mum] if i_mum < n_mum else None
            dad_gene = dad.connections[i_dad] if i_dad < n_dad else None

            selected_gene = None
            selected_genome = None

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

            if selected_gene is not None and selected_genome is not None:
                baby_connections.append(copy.copy(selected_gene))

                if not selected_gene.fr in node_ids:
                    node = selected_genome._get_node(selected_gene.fr)
                    if node != None:
                        baby_nodes.append(copy.copy(node))
                        node_ids.add(selected_gene.fr)

                if not selected_gene.to in node_ids:
                    node = selected_genome._get_node(selected_gene.to)
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

    def __str__(self):
        b = self.best
        s = '\nGeneration %s' %(self.gen)
        s += '\nBest id %s fitness %0.5f neurons %s links %s' % (b.id, b.fitness, len(b.nodes), len(b.connections))
        s += '\nspecies_id  ' + ' '.join('%4d' %(s.id) for s in self.species)
        s += '\nspawns_req  ' + ' '.join('%4d' %(s.spawns_required) for s in self.species)
        s += '\nmembers_len ' + ' '.join('%4d' %(len(s.pool)) for s in self.species)
        s += '\nage         ' + ' '.join('%4d' %(s.age) for s in self.species)
        s += '\nnot_improved' + ' '.join('%4d' %(s.stagnation) for s in self.species)
        s += '\nmax_fitness ' + ' '.join('%0.2f' %(s.max_fitness) for s in self.species)
        s += '\navg_fitness ' + ' '.join('%0.2f' %(s.average_fitness) for s in self.species)
        s += '\nleader      ' + ' '.join('%4d' %(s.best.id) for s in self.species)
        s += '\npopulation_len %s  species_len %s  compatibility_threshold %0.2f' %(len(self.pool), len(self.species), Options.compatibility_threshold)
        s += '\n'
        return s
