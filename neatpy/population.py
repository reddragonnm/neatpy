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

            if self.best.fitness >= Options.fitness_threshold:
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

        self.species[:] = [sp for sp in self.species if len(sp.pool) > 0]

    def _calc_spawns(self):
        total = max(1, sum([sp.average_fitness for sp in self.species]))
        for sp in self.species:
            sp.spawns_required = Options.population_size * sp.average_fitness / total

    def _reproduce(self):
        self.pool[:] = []
        for s in self.species:
            new_pool = []

            if Options.species_elitism:
                new_pool.append(s.best)

            while len(new_pool) < s.spawns_required:
                brain1 = s.get_brain()

                if random.random() < Options.crossover_rate:
                    brain2 = s.get_brain()
                    child = Brain.crossover(brain1, brain2, self.brain_id)
                    self.brain_id += 1
                else:
                    # child = copy.copy(brain1)
                    child = Brain.crossover(brain1, brain1)

                child.mutate()
                new_pool.append(child)

            self.pool.extend(new_pool)
            s.purge()

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

        self.species[:] = new_species

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

    def __str__(self):
        b = self.best
        s = '\nGeneration %s' % (self.gen)
        s += '\nBest id %s fitness %0.5f neurons %s links %s' % (
            b.id, b.fitness, len(b.nodes), len(b.connections))
        s += '\nspecies_id  ' + ' '.join('%4d' % (s.id) for s in self.species)
        s += '\nspawns_req  ' + \
            ' '.join('%4d' % (s.spawns_required) for s in self.species)
        s += '\nage         ' + ' '.join('%4d' % (s.age) for s in self.species)
        s += '\nnot_improved' + \
            ' '.join('%4d' % (s.stagnation) for s in self.species)
        s += '\nmax_fitness ' + \
            ' '.join('%0.2f' % (s.max_fitness) for s in self.species)
        s += '\navg_fitness ' + \
            ' '.join('%0.2f' % (s.average_fitness) for s in self.species)
        s += '\nleader      ' + ' '.join('%4d' % (s.best.id)
                                         for s in self.species)
        s += '\npopulation_len %s  species_len %s  compatibility_threshold %0.2f' % (
            len(self.pool), len(self.species), Options.compatibility_threshold)
        s += '\n'
        return s
