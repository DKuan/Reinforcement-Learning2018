"""
this file is for the struct of the mcts's tree using the language of Python 
for the program of mcts_gobang
author: Zachary
rights reserved
"""

class Node():
    """
    This class for mcts_tree's node
    it has three attributes:
    father children value
    """
    def __init__(self, father=None, player=None, action=None):
        self.father = father
        self.children = []
        self.value = 0
        self.simulation_node = False
        self.player = player
        self.visit = 0
        self.action = action 
