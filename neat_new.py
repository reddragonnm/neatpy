import random
import math
import copy

random.seed(11)


def sigmoid(x):
    try:
        return (1.0 / (1.0 + math.exp(-x)))
    except OverflowError:
        return 0 if x < 0 else 1


class Options:
    @staticmethod
    def set_options(
        num_inputs,
        num_outputs,
        population_size,

        fitness_threshold=float('inf'),
        max_nodes=float('inf'),

        activation_func=sigmoid,
        aggregation_func=sum,

        excess_coeff=1,
        disjoint_coeff=1,
        weight_coeff=0.5,

        add_node_prob=0.02,
        add_conn_prob=0.05,

        weight_mutate_prob=0.1,
        new_weight_prob=0.1,
        weight_init_range=1,
        weight_mutate_power=0.5,

        feature_selection=False,

        compatibility_threshold=3,
        dynamic_compatibility_threshold=True,

        target_species=20,
        dropoff_age=15,
        survival_rate=0.2,
        species_elitism=True,

        crossover_rate=1,
        tries_tournament_selection=3,

        young_age_threshhold=10,
        young_age_fitness_bonus=1.3,
        old_age_threshold=50,
        old_age_fitness_penalty=0.7
    ):
        Options.num_inputs = num_inputs
        Options.num_outputs = num_outputs
        Options.population_size = population_size

        Options.fitness_threshold = fitness_threshold
        Options.max_nodes = max_nodes

        Options.activation_func = activation_func
        Options.aggregation_func = aggregation_func

        Options.excess_coeff = excess_coeff
        Options.disjoint_coeff = disjoint_coeff
        Options.weight_coeff = weight_coeff

        Options.add_node_prob = add_node_prob
        Options.add_conn_prob = add_conn_prob

        Options.weight_mutate_prob = weight_mutate_prob
        Options.new_weight_prob = new_weight_prob
        Options.weight_init_range = weight_init_range
        Options.weight_mutate_power = weight_mutate_power

        Options.feature_selection = feature_selection

        Options.compatibility_threshold = compatibility_threshold
        Options.dynamic_compatibility_threshold = dynamic_compatibility_threshold

        Options.target_species = target_species
        Options.dropoff_age = dropoff_age
        Options.survival_rate = survival_rate
        Options.species_elitism = species_elitism

        Options.crossover_rate = crossover_rate
        Options.tries_tournament_selection = tries_tournament_selection

        Options.young_age_threshhold = young_age_threshhold
        Options.young_age_fitness_bonus = young_age_fitness_bonus
        Options.old_age_threshold = old_age_threshold
        Options.old_age_fitness_penalty = old_age_fitness_penalty


class NodeState:
    input = 'input'
    hidden = 'hidden'
    output = 'output'
    bias = 'bias'


class Node:
    pos = {}

    @staticmethod
    def init_pos():
        for i in range(Options.num_inputs + 1):
            Node.pos[i] = 0, 0

        for i in range(Options.num_outputs):
            Node.pos[Options.num_inputs + i + 1] = 0, 1

        InnovTable.set_node_id(Options.num_inputs + Options.num_outputs + 1)

    @staticmethod
    def set_pos(node_id, fr, to):
        if Node.pos.get(node_id) is None:
            a = Node.pos[fr]
            b = Node.pos[to]

            Node.pos[node_id] = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2

    @staticmethod
    def get_state(node_id):
        if node_id == 0:
            return NodeState.bias
        elif 1 <= node_id <= Options.num_inputs:
            return NodeState.input
        elif Options.num_inputs < node_id <= Options.num_inputs + Options.num_outputs:
            return NodeState.output

        return NodeState.hidden


class Connection:
    def __init__(self, fr, to, innov, weight=None):
        self.fr = fr
        self.to = to

        self.weight = weight or random.uniform(-1, 1) * \
            Options.weight_init_range

        self.enabled = True
        self.innov = innov


class Innovation:
    def __init__(self, innov, new_conn, fr=None, to=None, node_id=None):
        self.innov = innov
        self.new_conn = new_conn
        self.fr = fr
        self.to = to
        self.node_id = node_id


class InnovTable:
    history = []
    innov = 0
    node_id = 0

    @staticmethod
    def set_node_id(node_id):
        InnovTable.node_id = max(InnovTable.node_id, node_id)

    @staticmethod
    def _create_innov(fr, to, new_conn):
        if new_conn:
            innovation = Innovation(InnovTable.innov, new_conn, fr, to)
        else:
            innovation = Innovation(
                InnovTable.innov, new_conn, fr, to, node_id=InnovTable.node_id)
            InnovTable.node_id += 1

        InnovTable.history.append(innovation)
        InnovTable.innov += 1

        return innovation

    @staticmethod
    def get_innov(fr, to, new_conn=True):
        for innovation in InnovTable.history:
            if innovation.new_conn == new_conn and innovation.fr == fr and innovation.to == to:
                return innovation

        return InnovTable._create_innov(fr, to, new_conn)


class Brain:
    def __init__(self, genome_id, nodes=None, connections=None):
        self.id = genome_id
        self.fitness = 0

        self.nodes = nodes
        self.connections = connections

        if nodes is not None:
            self.nodes.sort()
            return

        node_id = 0

        self.nodes = []

        bias_nodes = []
        input_nodes = []
        output_nodes = []

        bias_nodes.append(node_id)
        node_id += 1

        for _ in range(Options.num_inputs):
            input_nodes.append(node_id)
            node_id += 1

        for _ in range(Options.num_outputs):
            output_nodes.append(node_id)
            node_id += 1

        self.nodes = bias_nodes + input_nodes + output_nodes
        self.connections = []

        for node1 in input_nodes + bias_nodes:
            for node2 in output_nodes:
                self.connections.append(
                    Connection(
                        node1,
                        node2,
                        InnovTable.get_innov(node1, node2).innov
                    )
                )

    def _add_conn(self):
        valid = []

        for node1 in self.nodes:
            for node2 in self.nodes:
                if self._valid_conn(node1, node2):
                    valid.append((node1, node2))

        if valid:
            node1_id, node2_id = random.choice(valid)

            self.connections.append(
                Connection(
                    node1_id,
                    node2_id,
                    InnovTable.get_innov(node1_id, node2_id).innov
                )
            )

    def _add_node(self):
        valid = [
            conn for conn in self.connections if conn.enabled and conn.fr != 0]

        if valid:
            conn = random.choice(valid)
        else:
            return

        node_id = InnovTable.get_innov(conn.fr, conn.to, False).node_id
        conn.enabled = False

        Node.set_pos(node_id, conn.fr, conn.to)
        self.nodes.append(node_id)

        self.connections.append(
            Connection(
                conn.fr,
                node_id,
                InnovTable.get_innov(conn.fr, node_id).innov,
                weight=1
            )
        )

        self.connections.append(
            Connection(
                node_id,
                conn.to,
                InnovTable.get_innov(node_id, conn.to).innov,
                weight=conn.weight
            )
        )

    def mutate(self):
        if random.random() < Options.add_node_prob and len(self.nodes) < Options.max_nodes:
            self._add_node()

        if random.random() < Options.add_conn_prob:
            self._add_conn()

        for conn in self.connections:
            if random.random() < Options.weight_mutate_prob:
                if random.random() < Options.new_weight_prob:
                    conn.weight = random.uniform(-1, 1) * \
                        Options.weight_init_range
                else:
                    conn.weight += random.uniform(-1, 1) * \
                        Options.weight_mutate_power

    def _get_input_connections(self, node_id):
        """Returns all connections where the connection leads to a node with given node_id

        Args:
            node_id (int): ID of the node

        Returns:
            List[Connections]: List of the connections where connection.to is node_id
        """
        return [conn for conn in self.connections if conn.to == node_id]

    def _get_node(self, node_id):
        """Returns node with given node_id in self.nodes

        Args:
            node_id (int): ID of the node

        Returns:
            Node: The Node which has the id -> node_id
        """
        return node_id

    def _valid_conn(self, node1, node2):
        for conn in self.connections:
            if conn.fr == node1 and conn.to == node2:
                return False

        return (
            node1 != node2 and
            Node.get_state(node1) in [NodeState.input, NodeState.hidden, NodeState.bias] and
            Node.get_state(node2) in [NodeState.hidden, NodeState.output] and
            Node.pos[node1][1] < Node.pos[node2][1]
        )

    def predict(self, inputs):
        assert len(inputs) == Options.num_inputs

        depth = len(set([Node.pos[node][1] for node in self.nodes]))
        val = {}

        for node in self.nodes:
            val[node] = 0

        for _ in range(depth):
            inp_num = 0

            for node in self.nodes:
                if Node.get_state(node) == NodeState.input:
                    val[node] = inputs[inp_num]
                    inp_num += 1

                elif Node.get_state(node) == NodeState.bias:
                    val[node] = 1

                else:
                    values = []
                    for conn in self._get_input_connections(node):
                        if conn.enabled:
                            values.append(
                                conn.weight * val[conn.fr])

                    val[node] = Options.activation_func(
                        Options.aggregation_func(values))

        return [val[node] for node in self.nodes if Node.get_state(node) == NodeState.output]

    @staticmethod
    def crossover(mom, dad, baby_id=None):
        n_mom = len(mom.connections)
        n_dad = len(dad.connections)

        if mom.fitness == dad.fitness:
            if n_mom == n_dad:
                better = random.choice([mom, dad])
            elif n_mom < n_dad:
                better = mom
            else:
                better = dad
        elif mom.fitness > dad.fitness:
            better = mom
        else:
            better = dad

        baby_nodes = set()
        baby_connections = []

        i_mom = i_dad = 0

        while i_mom < n_mom or i_dad < n_dad:
            mom_gene = mom.connections[i_mom] if i_mom < n_mom else None
            dad_gene = dad.connections[i_dad] if i_dad < n_dad else None

            selected_gene = None
            selected_genome = None

            if mom_gene and dad_gene:
                if mom_gene.innov == dad_gene.innov:
                    selected_gene, selected_genome = random.choice(
                        [(mom_gene, mom), (dad_gene, dad)])

                    i_mom += 1
                    i_dad += 1

                elif dad_gene.innov < mom_gene.innov:
                    if better == dad:
                        selected_gene = dad.connections[i_dad]
                        selected_genome = dad
                    i_dad += 1

                elif mom_gene.innov < dad_gene.innov:
                    if better == mom:
                        selected_gene = mom_gene
                        selected_genome = mom
                    i_mom += 1

            elif mom_gene == None and dad_gene:
                if better == dad:
                    selected_gene = dad.connections[i_dad]
                    selected_genome = dad
                i_dad += 1

            elif mom_gene and dad_gene == None:
                if better == mom:
                    selected_gene = mom_gene
                    selected_genome = mom
                i_mom += 1

            if selected_gene is not None and selected_genome is not None:
                baby_connections.append(copy.copy(selected_gene))

                baby_nodes.add(selected_gene.fr)
                baby_nodes.add(selected_gene.to)

        if True not in [l.enabled for l in baby_connections]:
            random.choice(baby_connections).enabled = True

        return Brain(baby_id, list(baby_nodes), baby_connections)

    @staticmethod
    def distance(genome1, genome2):
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

    def same_species(self, brain):
        return Brain.distance(brain, self.best) <= Options.compatibility_threshold


class Population:
    def __init__(self):
        self.pool = [Brain(i) for i in range(Options.population_size)]
        self.species = []

        self.best = self.pool[0]

        self.gen = 0
        self.brain_id = len(self.pool)
        self.species_id = 0

        Node.init_pos()

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
        return f'{self.gen} - {b.fitness}'


xor_inp = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_out = [0, 1, 1, 0]


def evaluate(nns):
    for nn in nns:
        nn.fitness = 4

        for xi, xo in zip(xor_inp, xor_out):
            output = nn.predict(xi)[0]
            nn.fitness -= (output - xo) ** 2


if __name__ == '__main__':
    Options.set_options(2, 1, 150, 3.9, weight_mutate_prob=0.3,
                        add_node_prob=0.05, add_conn_prob=0.1)

    p = Population()
    best, solved = p.evaluate(evaluate, 400)

    c = [(i.fr, i.to) for i in best.connections]
    print([i for i in best.nodes])
    print(c)

    assert len(c) == len(set(c))
