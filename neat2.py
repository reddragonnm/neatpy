import math
import random
import enum
import copy

random.seed(10)


def sigmoid(x):
    try:
        return (1.0 / (1.0 + math.exp(-x)))
    except OverflowError:
        return 0 if x < 0 else 1


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
    def create_innov(fr, to, new_conn):
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

        return InnovTable.create_innov(fr, to, new_conn)


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
    ):
        Options.num_inputs = num_inputs
        Options.num_outputs = num_outputs
        Options.population_size = population_size

        Options.fitness_threshold = fitness_threshold
        Options.max_nodes = max_nodes

        Options.activation_func = activation_func
        Options.aggregation_func = aggregation_func

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


class NodeState(enum.Enum):
    input = 'input'
    hidden = 'hidden'
    output = 'output'
    bias = 'bias'


class Node:
    def __init__(self, node_id, state, x, y):
        self.id = node_id
        self.state = state

        self.x = x
        self.y = y


class Connection:
    def __init__(self, fr, to, innov, weight=None):
        self.fr = fr
        self.to = to

        self.weight = weight or random.uniform(-1, 1) * \
            Options.weight_init_range

        self.enabled = True
        self.innov = innov


class Brain:
    def __init__(self, genome_id, nodes=None, connections=None):
        self.id = genome_id
        self.fitness = 0

        self.nodes = nodes
        self.connections = connections

        if nodes is not None:
            self.nodes.sort(key=lambda x: x.id)
            return

        input_pos_x = 1/(Options.num_inputs+1)
        output_pos_x = 1/(Options.num_outputs)
        node_id = 0

        self.nodes = []

        bias_nodes = []
        input_nodes = []
        output_nodes = []

        bias_nodes.append(Node(node_id, NodeState.bias, 0.5*input_pos_x, 0.0))
        node_id += 1

        for i in range(Options.num_inputs):
            input_nodes.append(
                Node(node_id, NodeState.input, (i+1.5)*input_pos_x, 0.0))
            node_id += 1

        for i in range(Options.num_outputs):
            output_nodes.append(Node(node_id, NodeState.output,
                                     (i+0.5)*output_pos_x, 1.0))
            node_id += 1

        self.nodes = bias_nodes + input_nodes + output_nodes
        InnovTable.set_node_id(node_id)
        self.connections = []

        for node1 in input_nodes + bias_nodes:
            for node2 in output_nodes:
                self.connections.append(
                    Connection(
                        node1.id,
                        node2.id,
                        InnovTable.get_innov(node1.id, node2.id).innov
                    )
                )

    def add_conn(self):
        valid = []

        for node1 in self.nodes:
            for node2 in self.nodes:
                if self.valid_conn(node1, node2):
                    valid.append((node1.id, node2.id))

        if valid:
            node1_id, node2_id = random.choice(valid)

            self.connections.append(
                Connection(
                    node1_id,
                    node2_id,
                    InnovTable.get_innov(node1_id, node2_id).innov
                )
            )

    def add_node(self):
        valid = [
            conn for conn in self.connections if conn.enabled and conn.fr != 0]

        if valid:
            conn = random.choice(valid)
        else:
            return

        fr = self.get_node(conn.fr)
        to = self.get_node(conn.to)

        x = (fr.x + to.x) / 2
        y = (fr.y + to.y) / 2

        node_id = InnovTable.get_innov(conn.fr, conn.to, False).node_id
        conn.enabled = False

        self.nodes.append(
            Node(
                node_id,
                NodeState.hidden,
                x, y
            )
        )

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
            self.add_node()

        if random.random() < Options.add_conn_prob:
            self.add_conn()

        for conn in self.connections:
            if random.random() < Options.weight_mutate_prob:
                if random.random() < Options.new_weight_prob:
                    conn.weight = random.uniform(-1, 1) * \
                        Options.weight_init_range
                else:
                    conn.weight += random.uniform(-1, 1) * \
                        Options.weight_mutate_power

    def get_input_connections(self, node_id):
        return [conn for conn in self.connections if conn.to == node_id]

    def get_node(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node

    def valid_conn(self, node1, node2):
        for conn in self.connections:
            if conn.fr == node1.id and conn.to == node2.id:
                return False

        return (
            node1.id != node2.id and
            node1.state in [NodeState.input, NodeState.hidden, NodeState.bias] and
            node2.state in [NodeState.hidden, NodeState.output] and
            node1.y <= node2.y
        )

    def predict(self, inputs):
        assert len(inputs) == Options.num_inputs

        depth = len(set(nn.y for nn in self.nodes))
        val_dict = {}

        for node in self.nodes:
            val_dict[node.id] = 0

        for _ in range(depth):
            inp_num = 0

            for node in self.nodes:
                if node.state == NodeState.input:
                    val_dict[node.id] = inputs[inp_num]
                    inp_num += 1

                elif node.state == NodeState.bias:
                    val_dict[node.id] = 1

                else:
                    values = []
                    for conn in self.get_input_connections(node.id):
                        if conn.enabled:
                            values.append(
                                conn.weight * val_dict[conn.fr])

                    val_dict[node.id] = Options.activation_func(
                        Options.aggregation_func(values))

        return [val_dict[node.id] for node in self.nodes if node.state == NodeState.output]

    @staticmethod
    def crossover(a, b, baby_id=None):
        n1 = len(a.connections)
        n2 = len(b.connections)

        better = max(a, b, key=lambda x: x.fitness)

        nodes = []
        connections = []

        i_a = i_b = 0
        node_ids = set()

        while i_a < n1 or i_b < n2:
            a_gene = a.connections[i_a] if i_a < n1 else None
            b_gene = b.connections[i_b] if i_b < n2 else None

            selected_gene = None
            selected_genome = None

            if a_gene and b_gene:
                if a_gene.innov == b_gene.innov:
                    selected_gene, selected_genome = random.choice(
                        [(a_gene, a), (b_gene, b)])

                    i_a += 1
                    i_b += 1

                elif b_gene.innov < a_gene.innov:
                    if better == b:
                        selected_gene = b.connections[i_b]
                        selected_genome = b
                    i_b += 1

                elif a_gene.innov < b_gene.innov:
                    if better == a:
                        selected_gene = a_gene
                        selected_genome = a
                    i_a += 1

            elif a_gene == None and b_gene:
                if better == b:
                    selected_gene = b.connections[i_b]
                    selected_genome = b
                i_b += 1

            elif a_gene and b_gene == None:
                if better == a:
                    selected_gene = a_gene
                    selected_genome = a
                i_a += 1

            if selected_gene is not None and selected_genome is not None:
                connections.append(copy.copy(selected_gene))

                if not selected_gene.fr in node_ids:
                    node = selected_genome.get_node(selected_gene.fr)
                    if node != None:
                        nodes.append(copy.copy(node))
                        node_ids.add(selected_gene.fr)

                if not selected_gene.to in node_ids:
                    node = selected_genome.get_node(selected_gene.to)
                    if node != None:
                        nodes.append(copy.copy(node))
                        node_ids.add(selected_gene.to)

        if True not in [l.enabled for l in connections]:
            random.choice(connections).enabled = True

        return Brain(baby_id, nodes, connections)

    @staticmethod
    def distance(a, b):
        l1 = set([c.innov for c in a.connections])
        l2 = set([c.innov for c in b.connections])

        n_match = 0
        n_disjoint = len(l1 ^ l2)
        weight_diff = 0

        w1 = {c.innov: c.weight for c in a.connections}
        w2 = {c.innov: c.weight for c in b.connections}

        for m in l1 & l2:
            n_match += 1
            weight_diff += abs(w1[m] - w2[m])

        return (Options.disjoint_coeff * n_disjoint) / \
            max(len(l1), len(l2)) + Options.weight_coeff * \
            weight_diff / n_match


class Species:
    def __init__(self, species_id, member):
        self.best = member

        self.pool = [member]
        self.id = species_id

        self.age = 0
        self.stagnation = 0

        self.spawns_required = 0
        self.average_fitness = 0

    def purge(self):
        self.age += 1
        self.stagnation += 1
        self.pool[:] = []

    def get_brain(self):
        thresh = random.uniform(
            0, sum(m.fitness for m in self.pool))

        for m in self.pool:
            thresh -= m.fitness

            if thresh <= 0:
                return m

    def cull(self):
        n = round(len(self.pool) * Options.survival_rate)
        self.pool[:] = self.pool[:max(1, n)]

    def adjust_fitnesses(self):
        self.average_fitness = sum(m.fitness / len(self.pool)
                                   for m in self.pool)

    def make_leader(self):
        self.pool.sort(key=lambda x: x.fitness, reverse=True)

        if self.pool[0].fitness > self.best.fitness:
            self.best = self.pool[0]
            self.stagnation = 0

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

    def speciate(self):
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

    def calc_spawns(self):
        total = max(1, sum([sp.average_fitness for sp in self.species]))
        for sp in self.species:
            sp.spawns_required = Options.population_size * sp.average_fitness / total

    def reproduce(self):
        self.pool[:] = []
        for s in self.species:
            new_pool = [s.best]

            while len(new_pool) < s.spawns_required:
                brain1 = s.get_brain()
                brain2 = s.get_brain()

                child = Brain.crossover(brain1, brain2, self.brain_id)
                self.brain_id += 1

                child.mutate()
                new_pool.append(child)

            self.pool.extend(new_pool)
            s.purge()

        while len(self.pool) < Options.population_size:
            self.pool.append(Brain(self.brain_id))
            self.brain_id += 1

    def sort_pool(self):
        self.pool.sort(key=lambda x: x.fitness, reverse=True)

        assert self.pool[-1].fitness >= 0, "Cannot handle negative fitness values"

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

            sp.cull()
            new_species.append(sp)

        self.species[:] = new_species

    def epoch(self):
        self.sort_pool()
        self.speciate()
        self.adjust_fitnesses()

        if Options.dynamic_compatibility_threshold:
            self.change_compatibility_threshold()

        self.calc_spawns()

        self.reset_and_kill()
        self.reproduce()

        self.gen += 1

    def __str__(self):
        return f'{self.gen} - {self.best.fitness}'


def evaluate(nns):
    for nn in nns:
        nn.fitness = 4

        for xi, xo in zip(xor_inp, xor_out):
            output = nn.predict(xi)[0]
            nn.fitness -= (output - xo) ** 2


if __name__ == '__main__':
    xor_inp = [(0, 0), (0, 1), (1, 0), (1, 1)]
    xor_out = [0, 1, 1, 0]

    Options.set_options(2, 1, 150, 3.9)
    p = Population()

    p.evaluate(evaluate)
