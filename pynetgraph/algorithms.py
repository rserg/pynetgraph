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
import graph
from abstract import ShortestPath, Codes


class GraphAlgorithms:
    @classmethod
    def easy_short_path(self, graph, start, end):
        visited = set()
        path = []
        for node in graph.edges():
            if node.node not in visited:
                visited.add(node.node)
                if node.node == end:
                    return None
            value = node.get_edges()
            minnode = min(value,
                          key=lambda x: x.weight)
            path.append(minnode.inedge)
        return path

    #Breadth-First Search
    '''start - the root node
       graph - current graph
       multithread - enable multithread way for this algorithm (not yet)
    '''
    def bsf(self, graph, start, end, multithread=False):
        return BreadthFirstSearch(graph).run()


class BreadthFirstSearch(ShortestPath):
    def __init__(self, graph):
        super(BreadthFirstSearch, self).__init__(graph)
        self.path = {}

    def run(self, start, end):
        que = Queue()
        que.put(start)
        while que:
            v = que.get()
            for node in self.graph.get(v, []):
                self.path[node] = v
                que.put(node)
            if end == v:
                return self._final_path(start, [end])

    def _final_path(self, start, result):
        if result[-1] == start:
            result.reverse()
            return result
        result.append(self.path[result[-1]])
        return self._final_path(start, result)


class DepthFirstSearch(ShortestPath):
    def __init__(self, graph):
        super(DepthFirstSearch, self).__init__(graph)
        self.visited = set()
        self.result = []

    def run(self, start):
        self.result.append(start)
        self.visited.add(start)
        for node in self.graph:
            if node not in self.visited:
                self.run(node)
        return self.result


class PruferCode(Codes):
    def __init__(self, graph):
        super(PruferCode, self).__init__(graph)

    def tocode(self):
        if self.graph.size() < 2:
            raise ValueError("Length of Graph is too short")

        knowgraph = self.graph
        prqueue = []
        while len(knowgraph.get_graph()) > 2:
            minvalue = filterfalse(lambda x: self.graph.adjacent(x),
                                       self.graph.get_graph())
            print(list(minvalue))
            knowgraph.delete_node(minvalue)
            heappush(prqueue, minvalue)

        return prqueue


class Prim:
    #graph = Tree in this case
    def __init__(self, sgraph):
        self.sgraph = sgraph
        self.prev=[None]*self.sgraph.size()
        self.key=[float("inf")]*self.sgraph.size()

    def run(self,root):
        #Nodes init to nil
        #self.key[root] = 0
        self.key[root] = 0
        S=[]
        dist={}
        prev={}
        heapify(S)
        heappush(S,root)


g = graph.Graph()
g.add_node(1)
g.add_node(2)
g.add_node(3)
g.add_node(4)
g.add_node(5)
g.add_node(6)
g.add_node(7)
g.add_node(8)
#g.add_edge('a')
g.add_edge(1,3,weight=7)
g.add_edge(1,4,weight=4)
g.add_edge(1,6,weight=3)
g.add_edge(2,3,weight=2)
g.add_edge(3,4,weight=3)
g.add_edge(3,6,weight=6)
g.add_edge(4,5,weight=1)
g.add_edge(5,6,weight=3)
g.add_edge(5,7,weight=2)
g.add_edge(6,7,weight=4)
g.add_edge(7,1,weight=1)
g.add_edge(7,8,weight=4)
g.add_edge(8,1,weight=2)
#g.add_edge_random(4,max_weight=50)
pr = Prim(g)
pr.run(1)

