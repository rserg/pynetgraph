import graph

class BuildGraph:
    def __init__(self,*args,**kwargs):
        self.graph = {}
        self.graph_type = kwargs.get('gtype')

    def add_algorithm(self, *args,**kwargs):
        pass

    def add_type(self, *args,**kwargs):
        self.graph_type = kwargs.get('gtype')
        if(self.graph_type =='dual'):
            raise NotImplemented

    #Show Current Graph
    def output(self):
        return self.graph

    def _make_dual_graph(self):
        pass

class MakeGraph:
    def __init__(self, nodes):
        self.count_nodes = nodes

    def _addNodes(self, item):
        self.nodes.append(item)

    def create(self):
        pass

class Shape:
    def __init__(self, graph):
        self.graph = graph

    def rectangle(self, *args, **kwargs):
        israndom = kwargs.get('rand')
        if len(self.graph) >= 4:
            rect = self.randsample(4)
            temp = rect
            graph = {i: None for i in rect}
            while len(rect) > 0:
                first = rect.pop()
                rect.reverse()
                last =rect.pop()
                rect.reverse()
                graph[first] = [last]
                graph[last] = [first]
            keys = list(graph.keys())
            graph[keys[0]].append(keys[1])
            graph[keys[1]].append(keys[0])
            graph[keys[2]].append(keys[3])
            graph[keys[3]].append(keys[2])
        return graph


    def trangle(self):
        pass

    def randsample(self, nums):
        return random.sample(self.graph,nums)

class BullGraph(MakeGraph, Shape):
    MAX=5
    def __init__(self):
        super(MakeGraph, self).__init__()
        self.graph={i: i for i in range(self.MAX)}
        self.numbers = self.rand()
        self.endnumbers = self.rand()
        self.make_edges(2, [self.numbers[0], self.numbers[1]])
        self.make_edges(self.numbers[0], [2, self.numbers[1]])
        self.make_edges(self.numbers[1], [2, self.numbers[0]])
        self.make_edges(self.rand()[0], self.rand()[1])
        self.make_edges(self.rand()[0], self.rand()[1])
        super(Shape, self).__init__(self.graph)
        print(self.graph)

    def rand(self):
        return random.sample([0,1,3,4],2)

    def make_edges(self, main, output):
        self.graph[main] = output


class Tree(list):
    def __init__(self, num,**kwargs):
        self.num = num
        self.attributes=kwargs
        self.name=kwargs.get('name', 'tree')

