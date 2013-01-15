
import inittest
import unittest
import graph
import algorithms
import random

class InitGraph(unittest.TestCase):
    """docstring for InitGraph"""
    def testopen(self):
        gr = graph.Graph()
        grp = graph.Graph({'a':[1,2,3], 'b':[7,8,9]})
        lis1 = ['a', 'b']
        lis2 = ['b', 'c']
        grzip = graph.Graph.Zipped(zip(lis1, lis2))

    def test_addnodes(self):
        gr = graph.Graph()
        gr.add_node(78)
        gr.add_node("SUPERNODE")
        gr.add_node("NEXTNODE")
        def add_test_size():
            for x in range(1000):
                gr.add_node(x)
            return gr.size();

        self.assertEqual(add_test_size(),1002)

    def test_append(self):
        pass


class ShortPathAlgorithms(unittest.TestCase):
    def test_easy_short_path(self):
        gr = graph.Graph({'s':['u', 'x'], 'u':['v', 'x'], 'v':['y'], 'x':['u', 'v', 'y'], 'y':['s', 'v']})
        gr.set_weight('s','u',5)
        gr.set_weight('s','x',10)
        gr.set_weight('u','v',7)
        gr.set_weight('u','x',4)
        gr.set_weight('v','y',13)
        gr.set_weight('x','u',8)
        gr.set_weight('x','v',6)
        gr.set_weight('x','y',10)
        gr.set_weight('y','s',18)
        gr.set_weight('y','s',11)
        algo = algorithms.GraphAlgorithms.easy_short_path(gr,'s', 'y')


if __name__ == "__main__":
    unittest.main()