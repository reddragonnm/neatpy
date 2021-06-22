import random
import copy

from .innovation import InnovTable
from .genes import NodeState, Node, Connection

from .options import Options


class Brain:
    def __init__(self, genome_id, nodes=None, connections=None):
        """Initialises a Brain object

        Args:
            genome_id (int): ID of the Brain object
            nodes (List[Node], optional): Contains a list of nodes generated during crossover. Defaults to None.
            connections (List[Connection], optional): Contains list of connections during crossover. Defaults to None.

        Contains:
            id (int): ID of the Brain
            fitness (float): Fitness of the Brain

            nodes (List[Node]): List of nodes. If it is None then nodes are initialised
            connections (List[Connections]): List of connections. If nodes is None then connections are initialised

        Methods:
            get_draw_info: Get the information required to draw the neural network
            predict: Predict the outputs based on the inputs
        """

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

        if Options.feature_selection:
            inp = random.choice(input_nodes + bias_nodes)
            out = random.choice(output_nodes)

            self.connections.append(
                Connection(
                    inp.id,
                    out.id,
                    InnovTable.get_innov(inp.id, out.id).innov
                )
            )
        else:
            for node1 in input_nodes + bias_nodes:
                for node2 in output_nodes:
                    self.connections.append(
                        Connection(
                            node1.id,
                            node2.id,
                            InnovTable.get_innov(node1.id, node2.id).innov
                        )
                    )

    def get_draw_info(self):
        """Can be used to get the info for drawing the neural network

        Returns:
            dict: The information for drawing

            It returns a dict in the form of:

            {
                'nodes': {
                    'input': [],
                    'hidden': [],
                    'output': [],
                    'bias': []
                },

                'connections': {
                    'enabled': [],
                    'disabled': [], 
                }
            }

            where each list in the 'nodes' dict contains the normalised x and y position of the node
            Each list in the 'connections' dict contains dicts in the form of:

            {
                'from': (),
                'to': (),
                'weight': <int>
            }

            the 'from' and 'to' keys in the dict contains the normalised x and y positions of the nodes
            the 'weight' contains the weight of the node for color coding the connections
        }
        """
        info = {
            'nodes': {
                'input': [],
                'hidden': [],
                'output': [],
                'bias': []
            },

            'connections': {
                'enabled': [],
                'disabled': [],
            }
        }

        for node in self.nodes:
            info['nodes'][node.state.value].append((node.x, node.y))

        for conn in self.connections:
            string = 'enabled' if conn.enabled else 'disabled'
            info['connections'][string].append(
                {
                    'from': (self._get_node(conn.fr).x, self._get_node(conn.fr).y),
                    'to': (self._get_node(conn.to).x, self._get_node(conn.to).y),
                    'weight': conn.weight
                }
            )

        return info

    def _filter_nodes(self, *args):
        """Filters all nodes according to states which are taken as args

        Returns:
            List[Node]: List of nodes
        """
        return [node for node in self.nodes if node.state in args]

    def _add_conn(self):
        """Adds a new connection between 2 nodes
        """
        valid = []

        for node1 in self.nodes:
            for node2 in self.nodes:
                if self._valid_conn(node1, node2):
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

    def _add_node(self):
        """Adds a new node by splitting a connection
        """
        valid = [
            conn for conn in self.connections if conn.enabled and conn.fr != 0]

        if valid:
            conn = random.choice(valid)
        else:
            return

        fr = self._get_node(conn.fr)
        to = self._get_node(conn.to)

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
        """Mutates the Brain according mutation rates defined in Options
        """
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
        for node in self.nodes:
            if node.id == node_id:
                return node

    def _valid_conn(self, node1, node2):
        """Checks if the connection between the given nodes is possible

        Args:
            node1 (Node): First node
            node2 (Node): Second node

        Returns:
            bool: Is the connection valid
        """
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
        """Predict the outputs based on the given inputs

        Args:
            inputs (List[float]): The inputs to the neural network

        Returns:
            List[float]: Outputs to the neural network
        """
        assert len(inputs) == Options.num_inputs

        depth = len(set([nn.y for nn in self.nodes]))

        for node in self.nodes:
            node.val = 0

        for _ in range(depth):
            inp_num = 0

            for node in self.nodes:
                if node.state == NodeState.input:
                    node.val = inputs[inp_num]
                    inp_num += 1

                elif node.state == NodeState.bias:
                    node.val = 1

                else:
                    values = []
                    for conn in self._get_input_connections(node.id):
                        if conn.enabled:
                            values.append(
                                conn.weight * self._get_node(conn.fr).val)

                    node.val = Options.activation_func(
                        Options.aggregation_func(values))

        return [node.val for node in self.nodes if node.state == NodeState.output]

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
                    selected_gene, selected_genome = random.choice(
                        [(mum_gene, mum), (dad_gene, dad)])

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

        if True not in [l.enabled for l in baby_connections]:
            random.choice(baby_connections).enabled = True

        return Brain(baby_id, baby_nodes, baby_connections)
