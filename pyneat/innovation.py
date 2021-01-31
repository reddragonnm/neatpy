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
    def create_innov(new_conn, fr, to):            
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
        for innovation in InnovTable.history:
            if innovation.new_conn == new_conn and innovation.fr == fr and innovation.to == to:
                return innovation

        return InnovTable.create_innov(new_conn, fr, to)