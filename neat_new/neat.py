import random


def sigmoid(x):
    try:
        return (1.0 / (1.0 + math.exp(-x)))
    except OverflowError:
        return 0 if x < 0 else 1


class Options:
    @staticmethod
    def set_options(
        inputs,
        outputs,
        population_size=150,

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
        Options.inputs = inputs
        Options.outputs = outputs
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

        Node.setup()


class NodeState:
    bias = 'bias'
    input = 'input'
    hidden = 'hidden'
    output = 'output'


class Node:
    pos = {}
    node_id = 0
    history = {}

    @staticmethod
    def setup():
        Node.pos = {}
        Node.history = {}
        Node.node_id = Options.inputs + Options.outputs + 1

        for i in range(Options.inputs + 1):
            Node.pos[i] = 0, 0

        for i in range(Options.outputs):
            Node.pos[Options.inputs + 1 + i] = 0, 1

    @staticmethod
    def get_node_id(conn):
        if Node.history.get(conn) is None:
            Node.history[conn] = Node.node_id
            Node.node_id += 1

            p1 = Node.pos[conn[0]]
            p2 = Node.pos[conn[1]]

            Node.pos[Node.history[conn]] = (
                p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

        return Node.history[conn]

    @staticmethod
    def get_state(idx):
        if idx == 0:
            return NodeState.bias
        elif idx <= Options.inputs:
            return NodeState.input
        elif idx <= Options.inputs + Options.outputs:
            return NodeState.output

        return NodeState.hidden


def new_conn(weight=None, enabled=True):
    if weight is None:
        weight = random.uniform(-1, 1)

    return {
        'weight': weight,
        'enabled': enabled
    }


class Brain:
    def __init__(self, nodes=None, conns=None):
        self._nodes = nodes
        self._conns = conns

        self._fitness = 0

        if self._nodes is None:
            self._gen_network()

    def _gen_network(self):
        self._nodes = set(i for i in range(
            Options.inputs + Options.outputs + 1))

        self._conns = dict()
        for i in range(Options.outputs):
            for j in range(Options.inputs + 1):
                self._conns[j, Options.inputs + 1 + i] = new_conn()

    def _add_node(self):
        conn = random.choice(
            [i for i in self._conns if i[0] != 0 and self._conns[i]['enabled']])

        self._conns[conn]['enabled'] = False

        node_id = Node.get_node_id(conn)
        self._nodes.add(node_id)

        self._conns[conn[0], node_id] = new_conn(1)
        self._conns[node_id, conn[1]] = new_conn(self._conns[conn]['weight'])

    def _add_conn(self):
        pass


if __name__ == '__main__':
    Options.inputs = 2
    Options.outputs = 1
    Node.setup()

    b = Brain()

    print(b._nodes, b._conns, Node.history)
    b._add_node()
    print()
    print(b._nodes, b._conns, Node.history)
