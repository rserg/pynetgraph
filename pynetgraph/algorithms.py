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
from abstract import ShortestPath

class GraphAlgorithms:
    @classmethod
    def easy_short_path(self, graph, start, end):
        visited=set()
        path=[]
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

    def run(self, start, end):
        marked = []
        path=defaultdict()
        que = Queue()
        que.put(start)
        while not que.empty():
            v = que.get()
            for node in self.graph[v]:
                    print(node)
                    path[node] = v
                    que.put(node)
            if end == v:
                return self._final_path(path, start, end)

    def _final_path(self, resultpath, start, end):
        print(resultpath)
        result=[end]
        while start != end:
            result.append(start)
            start = resultpath[start]
        return result
