import random

from .innovation import InnovTable
from .node import NodeState, Node
from .connection import Connection
from .activations import sigmoid
from .options import Options

class Brain:
    def __init__(self, genome_id, nodes=None, connections=None):
        self.id = genome_id
        self.fitness = 0

        self.nodes = nodes
        self.connections = connections

        if nodes is not None:            
            self.nodes.sort(key=lambda x: x.id)
            return

        input_pos_x = 1./(Options.num_inputs+1)
        output_pos_x = 1./(Options.num_outputs)
        node_id = 0
        
        self.nodes = []

        self.nodes.append(Node(node_id, NodeState.bias, 0.5*input_pos_x, 0.0))
        node_id += 1

        for i in range(Options.num_inputs):
            self.nodes.append(Node(node_id, NodeState.input, (i+1+0.5)*input_pos_x, 0.0))
            node_id += 1

        for i in range(Options.num_outputs):
            self.nodes.append(Node(node_id, NodeState.output, (i+0.5)*output_pos_x, 1.0))
            node_id += 1

        InnovTable.set_node_id(node_id)

        self.connections = []

        if Options.feature_selection:
            inp = random.choice(self.filter_nodes(NodeState.input))
            out = random.choice(self.filter_nodes(NodeState.output))

            self.connections.append(
                Connection(
                    inp.id,
                    out.id,
                    InnovTable.get_innov(inp.id, out.id).innov
                )
            )
        else:
            for node1 in self.filter_nodes(NodeState.bias, NodeState.input):
                for node2 in self.filter_nodes(NodeState.output):
                    self.connections.append(
                        Connection(
                            node1.id,
                            node2.id,
                            InnovTable.get_innov(node1.id, node2.id).innov
                        )
                    )

    def filter_nodes(self, *args):
        return [node for node in self.nodes if node.state in args]

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
        valid = [conn for conn in self.connections if conn.enabled and self.get_node(conn.fr).state != NodeState.bias]

        if valid:
            conn = random.choice(valid)
        else:
            return

        fr = self.get_node(conn.fr)
        to = self.get_node(conn.to)

        x = (fr.x + to.x) / 2
        y = (fr.y + to.y) / 2

        node_id = InnovTable.get_innov(conn.fr, conn.to, False).node_id

        if self.get_node(node_id) is None:
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
                    conn.weight = random.uniform(-1, 1) * Options.weight_init_range
                else:
                    conn.weight += random.uniform(-1, 1) * Options.weight_mutate_power

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

        return node1.id != node2.id and node1.state in [NodeState.input, NodeState.hidden, NodeState.bias] and node2.state in [NodeState.hidden, NodeState.output] and node1.y <= node2.y

    def predict(self, inputs):
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
                    sum = 0
                    for conn in self.get_input_connections(node.id):
                        if conn.enabled: sum += conn.weight * self.get_node(conn.fr).val

                    node.val = sigmoid(sum)

        return [node.val for node in self.nodes if node.state == NodeState.output]