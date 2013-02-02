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
import unittest
import time
from abstract import AbstractGraph

#Дуги между нодами, меняющиеся в зависимости от времени


class GraphDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        # self.graph = graph

    def __setitem__(self, node, value):
        super(GraphDict, self).__setitem__(node, value)


class GraphError(Exception):
    def __init__(self, value):
        Exception.__init__(self)


class StructNode:
    def __init__(self, node, value, attribute=[], **kwargs):
        self.node = node
        self.value = value
        self.attribute = attribute
        self.index = kwargs.get('index', [])
        self.x = kwargs.get('x')
        self.y = kwargs.get('y')
        self.edges = kwargs.get('edges')  # for multigraph
        self.weight = kwargs.get('weight')
        self.const = kwargs.get('cost')  # Minimal cost
        self.connected = []
        self.simple_connected = []
        self.checks = []
        if 'check' not in self.__dict__:
            self.check = 0

    '''def __new__(self,*args,**kwargs):
        self.check = 12'''

    # Index exist
    def isIndex(self, idx):
        return idx in self.index

    #Добавить связь
    def add_connect(self, edge):
        if(isinstance(edge, Edge)):
            self.simple_connected.append(edge.outedge)
            self.connected.append(edge)

    #На одну больше в счётчике обращений
    def inc(self):
        selr.check += 1

    def node(self):
        self.inc()
        return self.node

    def add_weight(self, weight):
        self.weight = weight

    def set_edge(self, edge):
        if(isinstance(edge, Edge)):
            self.connected.append(edge)

    def get_edges(self):
        return self.connected


#Пользавательский класс
class Node:
    def __init__(self, node, *args, **kwargs):
        self.node = node
        self.weight = kwargs.get('weight')
        self.structnode = None

    #Добавить случайный нод
    def add_random(self):
        self.node = self.node.__hash__()

    def add_weight(self, weight):
        self.weight = weight

    def add_advanced_node(self, structnode):
        if(isinstance(structnode, StructNode)):
            self.structnode = structnode

    def get_node(self):
        return self.node


class NodePosition(StructNode):
    def __init__(self, x, y, priority, distance):
        StructNode.__init__(None, None, x=x, y=y)
        self.x = priority
        self.y = distance
        self.priority = 0
        self.distance = 0
        self.cost = self.G + self.H  # Оценка пути

    def estimate(self, xOther, yOther):
        goalx = xOther - self.x
        goaly = xOther - self.y
        # Manhattan distance
        d = math.abs(goalx) + math.abs(goaly)

        return (d)


class Edge:
    def __init__(self, inedge, outedge, **kwargs):

        if not isinstance(outedge, list):
            self.outedge = [outedge]
        else:
            self.outedge = outedge
        self.value = self.outedge
        self.inedge = inedge
        self.weight = kwargs.get('weight')
        self.action = kwargs.get('action')
        self.label = kwargs.get('label')
        self.password = kwargs.get('password')

    def getLabel(self):
        return self.label

    def edge(self):
        return (self.inedge, self.outedge)

    def add(self, outedge):
        self.outedge.append(outedge)

    def change(self, inedge, outedge, **kwargs):
        self.inedge = inedge
        self.weight = kwargs.get('weight')
        self.action = kwargs.get('action')
        self.label = kwargs.get('label')
        self.password = kwargs.get('password')
        self.outedge.append(outedge)

#Вспомогательные функции для графа


class HelpGraph:
    def __init__(self, hgrapg):
        self.hgrapg = hgrapg

    def chesk_type(self):
        if not isinstance(self.hgrapg, list):
            return [self.hgrapg]
        return self.hgrapg


#Разные типы графов, где нужны проверки
class OtherGraph:
    def __init__(self, graph):
        self.graph = graph

    def check(self):
        pass


# Save past state of graph
class PastState:
    def __init__(self, node, edges, connectivity=[]):
        self.pastnode = node
        self.pastedge = edges
        self.pastconnection = connectivity


class Graph(AbstractGraph):
    def __init__(self, graphs={}, **kwargs):
        super(Graph, self).__init__(graphs, **kwargs)
        self.graphbase = GraphDict()
        self.paststates = []
        self.last_results = []
        if graphs != None:
            for node, edge in graphs.items():
                # self.checknodes(edge)
                self.append_c(node, edge)

    def Zipped(zipfunc):
        nodes = {}
        for node1, node2 in zipfunc:
            nodes[node1] = [node2]
        return Graph(nodes)
    # Check all nodes for exists

    def checknodes(self, nodes):
        for node in nodes:
            if not self.has_node(node):
                self.add_node(node)

    '''add Graph looks like
       A:[1,2,3]'''
    def append(self, graph):
        if len(graph) == 1:
            key = list(graph.keys())[0]
            if key not in self.graphbase:
                self.graphbase[key] = StructNode(key, self.has_nodes(graph[key]))

    def append_c(self, node, edge, attribute=[]):
        if node not in self.graphbase:
            self.graphbase[node] = StructNode(node, edge, attribute)

    def add_vertix(self, node):
        # main represent of graph
        self.graphbase = {}

    def add_edge(self, inedge, outedge, **kwargs):
        self.check_and_create(inedge)
        self.check_and_create(outedge)
        # assert self.has_nodes(HelpGraph(outedge).chesk_type())]
        if inedge in self.graphbase:
            self.graphbase[inedge].add_connect(Edge(inedge, outedge, **kwargs))

    '''Add random connectuon between all nodes E
    count - Number of iters
    arguments: max_weight - maximum random weight

    Add test for the case with similar nodes
    '''
    def add_edge_random(self, count, **kwargs):
        maxweight = kwargs.get('weight',10)
        from random import choice,randint
        for node in range(count):
            def chice():
                return choice(self.edges())
            self.add_edge(chice().node, chice().node,
                weight =randint(0,maxweight))

    # Connection between edges (from two sides)
    def connect(self, inedge, outedge, **kwargs):
        self.add_edge(inedge, outedge)
        self.add_edge(outedge, inedge)

    # StructNode is connection
    def add_node(self, node, **kwargs):
        if(not isinstance(node, StructNode)):
            self._add_nodeh(node, attribute=kwargs.get('attribute'), index=kwargs.get('index', []))
        else:
            newnode = node.get_node()
            self.graphbase[newnode] = StructNode(newnode, [])

    def _add_nodeh(self, node, **kwargs):
        if node in self.graphbase:
            count = self.graphbase[node].check
            self.graphbase[node] = StructNode(node, [],
                                              attribute=kwargs.get('attribute'), index=kwargs.get('index', []),
                                              checks=count + 1)
        else:
            print('NA :', node)
            self.graphbase[node] = StructNode(node, [],
                                              attribute=kwargs.get('attribute'), index=kwargs.get('index', []),
                                              checks=0)

    def add_node_from(self, imps):
        yield from imps

    # Add node and in the the absence case, raise Exception
    def add_node_exc(self, node, exception="StructNode already in graph"):
        if node in self.graphbase:
            raise GraphError(exception)
        self.add_node(node)

    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)

    def delete_node(self, node):
        if node in self.graphbase:
            self.paststates.append(PastState(node, self.graphbase))
            del self.graphbase[node]

    def delete_edge(self, nodein, nodeout):
        if nodein in self.graphbase and nodeout in self.graphbase:
            self.graphbase[nodein].value.remove(nodeout)

    def show_graph(self):
        for graph, node in self.graphbase.items():
            print(graph, node.value)

    # high ordered func check
    def has_node_bind(self, check, node):
        return check(node)

    def has_node(self, node):
        return node in self.graphbase

    def has_nodes(self, nodes):
        return [node for node in nodes if self.has_node(node)]

    def get_graph(self):
        return self.graphbase

    def edges(self):
        return list(self.graphbase.values())

    #Соседние вершины у ноды соединённые отрезком
    #!Дописать
    def neighbors(self, node):
        return self.graphbase[node]

    def edge(self, node):
        print(node, node in self.graphbase)
        if node in self.graphbase:
            return self.graphbase[node]
        else:
            return None

    #Взвешанный ли граф
    def is_weight(self):
        return len([node for node in self.graphbase if self.graphbase[node].weight == None]) == 0

    #Сортировка связей
    def sort_edges_by_weight(self):
        return sorted(self.edges(), key=lambda x: x.weight)

    # Oprional. Восстановить если возможно
    #Нужен тест
    def recovery(self, key):
        result = list(filter(lambda x: x.pastnode, self.paststates))
        if len(result) > 0:
            self.graphbase[result[0].pastnode] = StructNode(result[0].pastnode, [])

    #Поиск по индексу
    def findIndex(self, idx):
        return [node for node in self.graphbase
                if idx in self.graphbase[node].index]

    # Size of Graph
    def size(self):
        return len(self.graphbase)

    #Проверить, существует ли нода и если нет, то создать
    def check_and_create(self, node):
        # print(HelpGraph(node).chesk_type())
        self.add_nodes(HelpGraph(node).chesk_type())
        '''if isinstance(node,list):
            self.add_nodes(node)
        else:
            if node not in self.graphbase:
                print(node)
                self.add_node(node)'''
        '''if node not in self.graphbase:
            self.add_node(node)'''

    #Возвращает уникальные ноды
    def unique_nodes(self):
        return (set(self.get_graph()))

    def adjacent(self, node):
        if not self.has_node(node):
            raise Exception("StructNode {0} if not found"
                            .format(node))
        return self.graphbase[node].value

    def nodeinfo(self, node):
        print(node)
        if node in self.graphbase:
            return self.graphbase[node]

    def set_weight(self, edge_in, ed, wid):
        self.graphbase[edge_in].set_edge(Edge(edge_in, ed, weight=wid))

    # Проверка на циклы
    # http://code.google.com/p/python-graph/source/browse/trunk/core/pygraph/algorithms/cycles.py
    # nodes - {'A':['B','C']}
    # http://en.wikipedia.org/wiki/Tarjan%E2%80%99s_strongly_connected_components_algorithm
    def has_cyclic(self, nodes, another):
        spanning_tree = []
        if not has_nodes(nodes):
            while nodes != another:
                if nodes is None:
                    return []
                spanning_tree.append(nodes)
        return spanning_tree

    def isBipartite(self):
        q = Queue()
        key = self.graphbase.keys()
        start = list(key)[random.randint(0, len(self.graphbase) - 1)]
        for node in self.graphbase[start]:
            print(node)

    #Добавить веса графа
    def add_weight(self, node, weight):
        if node in self.graphbase:
            self.graphbase[node].add_weight(weight)

    def filter_node(self, pattern):
        return list(filter(lambda x: pattern(x), self.graphbase))

    def __str__(self):
        return "This graph has {0} ver and {1} nodes"\
            .format(len(self.graphbase), len(self.edges()))

    def connected(self, node):
        return self.graphbase[node].connected

    def clear(self):
        self.graphbase.clear()

    # Query for graph attributes
    def query(self, **kwargs):
        kwargs.get('select')

# Hypergraph area


class HyperGraph(Graph):
    def __init__(self):
        self.node_neighboards = {}
        self.graph = []

    def neighboards(self, node):
        self.node_neighboards[node] = {}

    def add_node(self, node):
        if len(node) <= 1:
            raise GraphException('THis is not Hypergraph')
        self.graph.append(node)

    def del_node(self, node):
        if node in self.graph:
            del self.graph[node]


class GraphException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
