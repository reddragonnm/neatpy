class Innovation:
    def __init__(self, innov, new_conn, fr=None, to=None, node_id=None):
        """Innovation details

        Args:
            innov (int): Innovation ID of the Innovation
            new_conn (bool): Is the new Innovation a innovation of a connection or a node
            fr (int, optional): Node ID of the input node. Defaults to None.
            to (int, optional): Node ID of the output node. Defaults to None.
            node_id (int, optional): ID of the node if it isn't a new_conn. Defaults to None.
        """
        self.innov = innov
        self.new_conn = new_conn
        self.fr = fr
        self.to = to
        self.node_id = node_id

class InnovTable:
    """Innovation Table of the whole NEAT algorithm. It is a static class
    
    Contains:
        history (List[Innovation]): List of all the innovations occured
        innov (int): The next innovation ID
        node_id (int): The next node ID
    """
    history = []
    innov = 0
    node_id = 0

    @staticmethod
    def set_node_id(node_id):
        """Sets the node_id if it is greater than InnovTable.node_id

        Args:
            node_id (int): The node ID
        """
        InnovTable.node_id = max(InnovTable.node_id, node_id)

    @staticmethod
    def create_innov(new_conn, fr, to):   
        """Create a new innovation

        Args:
            new_conn (bool): Is it a new connection. Other option is a node
            fr (int): Node ID of the input node
            to (int): Node ID of the output node

        Returns:
            Innovation: The new innovation
        """

        if new_conn:
            innovation = Innovation(InnovTable.innov, new_conn, fr, to)
        else:
            innovation = Innovation(InnovTable.innov, new_conn, fr, to, node_id=InnovTable.node_id)
            InnovTable.node_id += 1

        InnovTable.history.append(innovation)
        InnovTable.innov += 1

        return innovation

    @staticmethod
    def get_innov(fr, to, new_conn=True):
        """Gets a innovation with given args. Returns the innovation if present or else creates a new innovation

        Args:
            fr (int): Node ID of the input node
            to (int): Node ID of the output node
            new_conn (bool, optional): Does the innovation contain a new connection or node. Defaults to True.

        Returns:
            Innovation: The innovation
        """
        for innovation in InnovTable.history:
            if innovation.new_conn == new_conn and innovation.fr == fr and innovation.to == to:
                return innovation

        return InnovTable.create_innov(new_conn, fr, to)