
class AbstractGraph:
    def __init__(self, graphs={}, **kwargs):
        pass
    #Add node for graph, multigraph, dynamic graph

    def add_node(self, node):
        pass
    #add edge for graph, multigraph, dynamic graph

    def add_edge(self, inp, goal):
        pass

    def add_nodes(self, nodes):
        pass

    def delete_node(self, node):
        pass

    def has_node(self, node):
        pass
    #матрица смежности

    def matrix(self, node):
        pass  # Это ещё нужно сделать