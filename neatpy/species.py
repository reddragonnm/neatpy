from .options import Options
import random


class Species:
    def __init__(self, species_id, member):
        self.best = member

        self.pool = [member]
        self.id = species_id

        self.age = 0
        self.stagnation = 0

        self.spawns_required = 0

        self.max_fitness = 0.0
        self.average_fitness = 0.0

    def purge(self):
        self.age += 1
        self.stagnation += 1
        self.pool[:] = []

    def get_brain(self):
        best = self.pool[0]
        for _ in range(min(len(self.pool), Options.tries_tournament_selection)):
            g = random.choice(self.pool)
            if g.fitness > best.fitness:
                best = g

        return best

    def cull(self):
        self.pool[:] = self.pool[:max(
            1, round(len(self.pool) * Options.survival_rate))]

    def adjust_fitnesses(self):
        total = 0
        for m in self.pool:
            fitness = m.fitness

            if self.age < Options.young_age_threshhold:
                fitness *= Options.young_age_fitness_bonus

            if self.age > Options.old_age_threshold:
                fitness *= Options.old_age_fitness_penalty

            total += fitness / len(self.pool)

        self.average_fitness = total

    def make_leader(self):
        self.pool.sort(key=lambda x: x.fitness, reverse=True)
        self.best = self.pool[0]

        if self.best.fitness > self.max_fitness:
            self.stagnation = 0
            self.max_fitness = self.best.fitness

    @staticmethod
    def compat_dist(genome1, genome2):
        n_match = n_disjoint = n_excess = 0
        weight_difference = 0

        n_g1 = len(genome1.connections)
        n_g2 = len(genome2.connections)
        i_g1 = i_g2 = 0

        while i_g1 < n_g1 or i_g2 < n_g2:
            # excess
            if i_g1 == n_g1:
                n_excess += 1
                i_g2 += 1
                continue

            if i_g2 == n_g2:
                n_excess += 1
                i_g1 += 1
                continue

            conn1 = genome1.connections[i_g1]
            conn2 = genome2.connections[i_g2]

            # match
            if conn1.innov == conn2.innov:
                n_match += 1
                i_g1 += 1
                i_g2 += 1
                weight_difference += abs(conn1.weight-conn2.weight)
                continue

            # disjoint
            if conn1.innov < conn2.innov:
                n_disjoint += 1
                i_g1 += 1
                continue

            if conn1.innov > conn2.innov:
                n_disjoint += 1
                i_g2 += 1
                continue

        n_match += 1
        return (Options.excess_coeff * n_excess + Options.disjoint_coeff * n_disjoint) / max(n_g1, n_g2) + Options.weight_coeff * weight_difference / n_match

    def same_species(self, brain):
        return Species.compat_dist(brain, self.best) <= Options.compatibility_threshold
