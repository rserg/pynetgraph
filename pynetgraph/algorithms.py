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

    def djestra(self):
        pass



class Transform:
    def unconnect(self, graph):
        pass