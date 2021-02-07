import json

from .genes import Connection, Node, NodeState
from .brain import Brain

def _save_node(node):
    return {
        'id': node.id,
        'state': node.state.value,
        'x': node.x,
        'y': node.y
    }

def _save_conn(conn):
    return {
        'fr': conn.fr,
        'to': conn.to,
        'weight': conn.weight,
        'enabled': conn.enabled,
        'innov': conn.innov
    }

def save_brain(brain, file_name=None):
    data = json.dumps(
        {
            'id': brain.id,
            'nodes': [_save_node(node) for node in brain.nodes],
            'connections': [_save_conn(conn) for conn in brain.connections]
        },

        indent=4
    )

    if file_name is not None and isinstance(file_name, str):
        with open(file_name, 'w') as file:
            file.write(data)

    return data

def _load_node(node):
    return Node(
        node['id'],
        NodeState(node['state']),
        node['x'],
        node['y']
    )

def _load_conn(conn):
    return Connection(
        conn['fr'],
        conn['to'],
        conn['innov'],
        conn['enabled'],
        conn['weight']
    )

def load_brain(file_name):
    data = json.load(open(file_name))

    return Brain(
        data['id'],
        [_load_node(node) for node in data['nodes']],
        [_load_conn(conn) for conn in data['connections']],
    )