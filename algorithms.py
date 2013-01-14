import graph
import json
import threading
import functools
from itertools import *
from collections import defaultdict
from queue import Queue, LifoQueue
from heapq import heappush, heappop, heapify
from math import pow
import math
import random

class GraphAlgorithms:
    def __init__(self, graph):
        if(not isinstance(graph, Graph)):
            raise GraphException("This is not graph Class")
        self.graph = graph

    '''Example nodes of Graph
    Input: ['A','B], ['C','B'],['B','D']

    Output: ['A','B], ['C','B'],['B','D'],
    ['C','D'],['A','D'],['A','C']
    docs
    http://en.wikipedia.org/wiki/Transitive_closure
    http://www.boost.org/doc/libs/1_51_0/boost/graph/transitive_closure.hpp
    http://stackoverflow.com/questions/8673482/transitive-closure-python-tuples'''
    def transitive_closure(self, elements):
        elements = set([(x, y) if x < y else (y, x) for x, y in elements])
        relations = {}
        for x, y in elements:
            if x not in relations:
                relations[x] = []
            relations[x].append(y)
        closure = set()

        def build_closure(n):
            def f(k):
                for y in relations.get(k, []):
                    yield (n, y)
                    #closure.add((n, y))
                    f(y)
            f(n)

        for k in relations.keys():
            build_closure(k)
        return closure

    '''def transitive_closure2(self,elements):
        edges = defaultdict(set)
        for x, y in elements: edges[x].add(y)
        for _ in range(len(elements) - 1):
            edges = defaultdict(set, (
                (k, v.union(*(edges[i] for i in v)))
                for (k, v) in edges.items()))

        return set((k, i) for (k, v) in edges.items() for i in v)'''

    #Нвходит кратчайший путь нужно с помощью дерева
    #Ван Эдме Боаса http://habrahabr.ru/post/125499/
    def dijkstra_short_path(self, fnode):
        pass

    #Топологическая сортировка
    #Нужно проверять!
    #habrahabr.ru/post/100953
    def topological_sort(self):
        for node in self.graph:
            result = GraphSearch.dfs(self.graph, node)
            if result:
                return False

        return true

    def dfs(self, node, path=[]):
        return GraphSearch.dfs(self.graph, node, path)

    #multithread ?
    def bsf(self, start, path=[], multithread=False):
        marked = []
        que = Queue()
        que.put(start)
        while not que.empty():
            v = que.get()
            if v in path:
                path = path + [v]
                que.put(v)
        return path

    def _bsf(self, start, path=[], multithread=False):
        if not self.graph.has_node(start):
            raise GraphException("This not has not found")

        if multithread == False:
            que = Queue()
            que.put(start)
            while not que.empty():
                v = que.get()
                if v in self.graph.get_graph():
                    path = path + [v]
                    que.put(v)
        else:
            def bfs_thread():
                t = threading.Thread(name='bfs_search', target=_bsf)
                t.start()

    #Код пруфера
    #http://e-maxx.ru/algo/prufer_code_cayley_formula
    #http://r-aa.livejournal.com/13495.html
    def prufer_code(self):
        #Если в дереве меньше 2 веришин, то выход
        if len(self.graph.get_graph()) < 2:
            raise ValueError("Length of Grapg is too short")

        knowgraph = self.graph
        prqueue = []
        #print(knowgraph.get_graph())
        while len(knowgraph.get_graph()) > 2:
            minvalue = min(filterfalse(lambda x: self.graph.adjacent(x),
                                       self.graph.get_graph()))
            #Переделать граф, чтобы небыло пустых нодов
            knowgraph.delete_node(minvalue)
            heappush(prqueue, minvalue)
            print(minvalue)

        return prqueue

    #Floyd-Warshall algorithm
    #http://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
    def fwa(self):
        def init_array(self, wgraph):
            return {i: float('inf') for i in wgraph}
        for i in self.graph.size():
            for j in self.graph.size():
                for k in self.graph.size():
                    pass
                    #if(self.graph[i][k] + )

    #Belmann Ford Shortest Path
    #Взвешанный граф
    #wgraph - weight graph, goal - find node
    #!Note сделать отдельный класс!
    def BF_shortest_path(self, _wgraph, goal):
        def init_array(_wgraph):
            return {i: float('inf')for i in _wgraph},\
                {i: None for i in _wgraph}

        if(not isinstance(self.graph, WeightGraph)):
            raise GraphException("This is not weight graph")
        wgraph = _wgraph
        arr, pinit = init_array(_wgraph)
        arr[goal] = 0
        for v in range(len(wgraph) - 1):
            for node in wgraph:
                for neighbour in wgraph[node]:
                    if arr[neighbour] > arr[node] + wgraph[node][neighbour]:
                        arr[neighbour] = arr[node] + wgraph[node][neighbour]
                        pinit[neighbour] = node

        return arr, pinit

    def another_short_path(self, node):
        pass

    #A* алгоритм
    #http://code.activestate.com/recipes/577519-a-star-shortest-path-algorithm/
    #http://stackoverflow.com/questions/4159331/python-speed-up-an-a-star-pathfinding-algorithm
    def AStar(self, start, goal):
        astar = AStar([1, 2, 3])
        astar.run(1, 3)
        print (astar.cost())

    #Первое наилучшее совпадение
    def BestFirstSearch(self, start, goal):
        bfs = BestFirstSearch([(0, 1), (1, -1), (0, 0)])
        bfs.run()
        print(bfs.cost())

    #Приблизительно для этого алгоритма
    def CostSearch(self, limit, goal):
        q = Queue.Queue()
        q.put(goal)
        while not q.isEmpty:
            current = self.graphbase[goal]
            if current.cost > limit:
                CostSearch(limit, self.graphbase[current])

    def kruskal(self):
        return Kruskal(self.graph).run()

    #Гамельтоновый цикл
    #http://www.boriel.com/2011/03/22/camino-hamiltoniano/?lang=en
    #http://research.cyber.ee/~peeter/teaching/graafid08s/previous/loeng3eng.pdf
    def hamiltonian_cycle(self):
        if(not isinstance(self.graph, WeightGraph)):
            raise GraphException("this is not Weight Graph")

    #Меняем местами
    def swap(self, inp, outp):
        assert self.graph.has_node(inp) and self.graph.has_node(outp),\
            "This graph not exists some of nodes"

        node1 = self.graph[inp]
        node2 = self.graph[outp]
        swap = node2
        node2 = node1
        node1 = swap

        self.graph[inp] = node1
        self.graph[outp] = node2

    #Сумма весов
    def sum(self):
        suma = 0
        itertools.count(lambda x: x.attribute['width'], self.graph)
        for node in self.graph.get_graph:
            suma += self.graph[node].attribute['width']
        return suma

    def betweenness_centrality(self):
        pass

    #Применить событие
    #action
    def action_nodes(self, action):
        size = self.graph.length()
        for node in self.graph.get_graph():
            action(node)

    def test_render(self, action):
        pass

    #Minimum Spanning tree
    def MST(self, graph):
        pass

    #Метод Шульце
    #def Schulze_method
    #Алгоритм Прима в остовном дереве
    #graph - Граф с весами (взвешанный граф?)
    #start - вершина, откуда начинаем
    #http://urban-sanjoo.narod.ru/prim.html
    #http://e-maxx.ru/algo/mst_prim
    def prime(self, graph, gr, start):
        def init_array(_wgraph):
            return {i: float('inf')for i in _wgraph},\
                {i: None for i in _wgraph}

        Q = []
        key, prev = init_array(graph)
        heapq.heappush(Q, graph[start])
        while not Q.empty():
            d = heapq.heappop()
            for node in self.graph.adjacent(d):
                if(gr[d][node] < key[node]):
                    prev[node] = d
                    key[node] = graph[d][node]


    #http://en.wikipedia.org/wiki/Connected_dominating_set
    #http://www.cis.upenn.edu/~sudipto/mypapers/cds_algorithmica.pdf
    #def connecting_dominating_set(self,graph):
#Специальный класс для поиска кратчайших путей
class ShortestPath:
    def __init__(self):
        pass
    #Запуск алгоритма

    def run(self, start, goal):
        pass
    #Стоимость алгоритма

    def cost(self):
        pass
    #Количество действий


#Алгоритм Краскала
#https://raw.github.com/israelst/Algorithms-Book--Python/master/5-Greedy-algorithms/kruskal.py
class Kruskal:
    def __init__(self, graph):
        self.rank = dict()
        self.parent = dict()
        self.graph = graph
        self._append()

    def run(self):
        parent = {}
        rank = {}
        spanning_tree = set()
        edges = self.graph.edges()
        if not self.graph.is_weight():
            raise Exception("Graph is not weight")

        edges = self.graph.sort_edges_by_weight()
        for node in edges:
            weight = node.weight
            edge = node.node
            inp = node.value
            if edge not in inp:
                self.comparerank(inp, edge)
                spanning_tree.add(node)
        return spanning_tree

    def _newset(self, node):
        self.parent[node] = node
        self.rank[node] = 0

    def _append(self):
        for node in self.graph.edges():
            self._newset(node.node)

    def find(self, edge):
        if isinstance(edge, list):
            for node in edge:
                if node != self.parent[node]:
                    self.parent[node] = find(self.parent[node])
                    result = node
        else:
            if self.parent[edge] != edge:
                self.parent[edge] = find(self.parent[edge])
        return self.parent[edge]  # Пере

    def comparerank(self, edgein, edgeout):
        root1 = self.find(edgein)
        root2 = self.find(edgeout)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1
#http://www.policyalmanac.org/games/aStarTutorial_rus.htm
#matrix [(0,1),(-1,0),(1,1),(0,0),(1,0)]


class AStarNode(Node):
    def __init__(self, node):
        #Node.__init__(node,[])
        self.node = node
        self.last = None
        self.cost = 0
        self.G = 0
        self.F = self.G + self.cost

    def update(self):
        return


class AStar(ShortestPath):
    def __init__(self, graph, **kwargs):
        ShortestPath.__init__(graph)
        self.graph = graph
        self.rememberPaths = []  # Уже просчитанные пути
        self.minnode = AStarNode(0)
        self.distance_func = kwargs.get('distance_cost', self._maddim)

    def _getmin(self, node):
        for minf in openSet:
            if minf.F < node.F:
                node.F = minf.F

    #http://scriptogr.am/jdp/post/pathfinding-with-python-graphs-and-a-star
    #http://www.oriontransfer.co.nz/research/Inverse%20Kinematics%20A-star.pdf
    #Тестировать
    def run(self, start, goal):
        closeSet = set()  # Обработанные вершины
        openSet = []  # Необработанные вершины
        pathmap = dict()
        openSet.append(start)
        #q.put(self.minnode)
        #heapify(openSet)
        #heappush(openSet, start)
        while len(openSet) > 0:
            x = min(openSet, key=lambda curr: self.distance_func(curr, goal))
            #x = heappop(openSet)
            #Ищем в открытом списке клетку с наименьшим F
            for minf in openSet:
                if minf.F < x.F:
                    x.F = minf.F

            print(x.node)
            if x.node == goal.node:
                return self.reconstruct_path(start, goal)

            openSet.remove(x)
            closeSet.add(x)

            for ngnode in self.graph[x.node]:
                #print(ngnode.node)
                if ngnode in closeSet:
                    continue

                need_update = True
                score = x.G + self.dist(x, ngnode)
                if ngnode in openSet:
                    if ngnode.G > score:
                        ngnode.G = score
                        ngnode.last = x.node

                else:
                    ngnode.G = score
                    ngnode.H = self.update(ngnode, goal)
                    ngnode.last = x.node
                    openSet.append(ngnode)

            g_sore = 1  # Вычисляем g(x) обрабатываемого соседа'''

    #Манхеттеновское расстояние. Вычислени от старта до цели
    def _maddim(self, start, goal):
        return 10 * abs(start.F - start.G) + abs(goal.F - goal.G)

    #Евклидово расстояние
    def _euqlide(self, start, goal):
        return math.sqrt(start.x ** 2 - start.y ** 2) - (goal.x ** 2 - goal.y ** 2)

    def reconstruct_path(self, start, goal):
        pass

    #Соседи
    def neighbor_nodes(self, x):
        pass

    #Расстояние
    def dist(self, x, y):
        return x.F + y.F * 2

    def update(self, x, y):
        return x.F * y.F


#http://www.optimization-online.org/DB_FILE/2008/11/2154.pdf
#http://stackoverflow.com/questions/10783659/is-there-any-implementation-of-bidirectional-search-for-dijkstra-algorithm
#http://stackoverflow.com/questions/3641741/bidirectional-a-a-star-search?rq=1
class BidirectionalSearch:
    def __init__(self, graph):
        self.graph = graph

    def run(self, start, target):
        for node in self.graph:
            pass

#Первый лучший
#http://en.wikipedia.org/wiki/Best-first_search


class BestFirstSearch(ShortestPath):
    def __init__(self, matrix):
        pass

    def run(self, start, goal):
        q = Queue.Queue()
        q.put(self.graph[start])
        while not q.Empty:
            current = q.get()
            if current == goal:
                return self._trace_back(start, goal)

    def _traceback(self, start, goal):
        pass


#http://habrahabr.ru/post/162915/
class JumpPointSearch:
    def __init__(self, matrix):
        self.matrix = matrix

#Туры по графам


class Tour:
    def __init__(self, matrixGraph):
        self.matrixGraph = matrixGraph

    def run(self, start):
        pass

    def cost(self):
        pass


#Эйлеров цикл - путь, проходящий по всем рёбрам графа один раз
#http://code.activestate.com/recipes/498243-finding-eulerian-path-in-undirected-graph/
#Проверить
class EulerianPath(Tour):
    def __init__(self, matrixGraph):
        super(EulerianPath, self).__init__(matrixGraph)
        self.matrixGraph = matrixGraph
        self.costnum = 0

    def run(self, pos):
        path=[]
        #q = Queue()
        q = [pos]
        lastnodes = None
        while q:
            node = q[-1]
            if self.matrixGraph.edge(node) != None and self.matrixGraph.edge(node).value != []:
                currentnode = self.matrixGraph.edge(node).value[0]
                q.append(currentnode)
                if lastnodes == currentnode:
                    #Blocking nodes
                    return []
                lastnodes = currentnode
                self.matrixGraph.delete_edge(node,currentnode)
            else:
                path.append(q.pop())
        print(path)
        return path

    def cost(self):
        return self.costnum


class RandomGraph:
    def __init__(self, nodes=100, edhes=100):
        self.nodes = [0] * round(random() * 100)
        self.edges = round(random() * 100)

    #Абсодютно случайное связывание
    def generate(self):
        for node in self.nodes:
            print(node)


#МультиГраф
#Может хранить несколько рёбер к одной вершине
class MultiGraph(AbstractGraph):
    def __init__(self, graphs={}, **kwargs):
        AbstractGraph().__init__(graphs, kwargs)
        self.direction = kwargs.get('direction')
        self.graph = {}

    def add_node(self, node, **kwargs):
        self.graph[node] = StructNode(node, [], [])

    def add_edge(self, inc, out, nodes):
        self.graph[inc] = Edge(inc, out, nodes)

    def has_node(self, node):
        return node in self.graphbase

    def delete_node(self, node):
        if node in self.graph:
            #Добавляем предыдущее состояние
            self.paststates.append(PastState(node, self.graph))
            del self.graph[node]


class GraphFlow:
    def __init__(self):
        pass
#Matching area
#http://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm


#Кластеризация графа
#http://geza.kzoo.edu/~erdi/patent/Schaeffer07.pdf
#http://cs.nyu.edu/shasha/papers/GraphClust.html
class GraphClustering:
    def __init__(self, graph, **kwargs):
        self.graph = graph
    #Алгоритм кластеризации

    def rnsc(self):
        clusters = {}
        bestcost = 0
        for i in self.graph:
            if self.graph[i].weight > bestcost:
                bestcost = self.graph[i].weight


#Динамические графы
#http://codeforces.ru/blog/entry/5500
#http://www.cs.cmu.edu/~sleator/papers/dynamic-trees.pdf
#http://e-maxx.ru/algo/heavy_light
#http://wcipeg.com/wiki/Heavy-light_decomposition
#http://e-maxx.ru/algo/sqrt_decomposition
#http://apps.topcoder.com/forums/;jsessionid=6B12CEFE53BB57063C58C5869608AF00?module=Thread&threadID=621514&start=0&mc=15#1012829
class TournamentGraph:
    def __init__(self, graph):
        self.graph = graph

    def heavylightdecomposition(self):
        for node in self.graph:
            prrint(node)