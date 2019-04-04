"""
This file is the sub of base pakage's module class: Agent_Base
rewrite the method of three
for the mcts_agent
Author: Zachary
Reserved Rights. 
"""
import sys
sys.path.append('../')

from base.Agent_Base import Agent_Base 
from mcts_tree import Node
import numpy as np
from tqdm import tqdm
import pickle
import math

class MCTS_Agent(Agent_Base):
    """
    this class is the running-time mcts agent to find the good action for the state now
    """
    def __init__(self, env, load_model_flag=False, discount=1.0, max_mcts_simulation=20, max_mcts_episode=1000):
        """
        init the data 
        arg:
        |env: instance of the env, for the use in simulation to step the action
        |load_model_flag: bool, set true to load model from the model file
        |discount: float, to discount the reward backward
        |max_mcts_episode: int, the max num of the complete mcts run
        |max_mcts_simulation: int, the max num of the simulation run in one mcts
        """
        Agent_Base.__init__(self)
        self.c = 2.46 # experienced value
        self.env = env # for the use of bellow
        self.terminate = False # show if the game is over
        self.mcts_tree = [] # store the tree in the trajectory
        self.my_action = None # action choose
        self.decision_node = None # the node to make decision this time
        self.max_mcts_simulation = max_mcts_simulation 
        self.max_mcts_episode = max_mcts_episode


    def mcts_selection(self):
        """
        this func is in mcts program, the first method
        select an action that can be expanded from the root
        in policy_tree
        arg:
        |state_now: list, the state now we should do dicision
        """
        root = self.mcts_tree # first from the self.mcts_tree
        select_success = False # if the select is over
        while(select_success != True):
            action_done_space = [child.action for child in root.children] # get the actions have done

            """ span the action space if the node is not full """
            for action in self.trace_neighbor_place(): # create the children nodes
                if action in action_done_space: continue # skip this action
                node_ = Node(father=root, player=self.env.current_player_id, action=action)
                root.children.append(node_)

            """ select the child node """
            children_visits = [child.visit for child in root.children] # get the children visit
            root_visit = np.sum(children_visits) # cal the root visit
            
            if  all(children_visits):

                """ cal the UCT of all the children nodes """
                children_UCT = [child.value/child.visit + self.c*math.sqrt(root.visit/child.visit) for child in root.children] # cal the children value by UCT
                child_choose_id = np.where(np.max(children_UCT)==children_UCT)[0][0] # choose the child by the UCT
                self.node_selected = root.children[child_choose_id] # choose the max UCT child

                if  self.node_selected.children.__len__() == 0: # the child is a leaf
                    select_success = True
                else:

                    """ if terminate , then return """ 
                    data_return = self.env.step(self.node_selected.action)['terminal'] # step env,if have child node,then mustn't be terminate
                    root = self.node_selected # choose another way, to a deeper layer
                    if data_return['terminal'] == True: # if the game end in the selection func
                        return data_return['reward'] 
            else:

                """ the children nodes haven't been explored completly """
                children_no_visit = np.where(np.array(children_visits) == 0)[0] # random choose a child
                child_choose_id = children_no_visit[np.random.randint(children_no_visit.__len__())] # random
                self.node_selected = root.children[child_choose_id] # get the next node
                select_success = True

        data_return = self.env.step(self.node_selected.action) # update the state
        if data_return['terminal'] == True: # if the game end in the selection func
                return data_return['reward'] 
        

    def mcts_simulation(self):
        """
        this func is in mcts program, the third method
        from the node to simulation the spisode
        in policy_default
        arg:
        |depth: int, the depth of the tree, which is the start of our simulation
        """
        self.simulation_steps = 0 # discount the reward
        reward = 0 # init the reward
        self.node_selected.simulation_node = True # simulation start from this node
        terminate, reward = self.env._is_terminal() # check the env

        for step in range(self.max_mcts_simulation):

            """ check if the state is end """
            if terminate == True: # return the reward
                return reward 

            """ no place can be choosed """
            if self.env.available_action_space.__len__() == 0: # punish the agent
                return 0

            self.simulation_steps += 1 # update the simulation

            """ choose the action and update the selected node """
            action = self.policy_default() # get action by the default policy
            actions = [child.action for child in self.node_selected.children] # get all the actions
            node_ids = np.where(np.array(actions) == action)[0] # exist nodes

            if node_ids.__len__() == 0: # there is no action
                node_ = Node(father=self.node_selected, player=self.env.current_player_id, action=action) # add the node to the tree
                self.node_selected.children.append(node_) # add the node to the self selectd
                self.node_selected = node_ # update the node selected
            else:
                self.node_selected = self.node_selected.children[node_ids[0]] # get the new node
            
            """interact with the env"""
            data_return = self.env.step(action) # interact with the env
            terminate, reward = data_return['terminal'], data_return['reward'] # check if the game is over

        return 0 


    def mcts_backup(self, reward):
        """
        this func is in mcts program, the final method
        from the terminate to backup to the root, back the rewards 
        in policy_default
        arg:
        |reward: int, the reward that received in this simulation
        |node_new: Node class, the 
        """
        while(self.node_selected.father != None):
            self.node_selected.value += self.node_selected.player*self.env.win_player*abs(reward) # update the value 
            self.node_selected.visit += 1 # update the visit number
            
            """ just leave the expanded node, all others should be threw """
            if self.node_selected.simulation_node == True:
                self.node_selected.simulation_node = False # reset the flag
                self.node_selected.children = [] # clear the children
            self.node_selected = self.node_selected.father # backup from this node

        """ update the root node and reset the env """
        self.node_selected.visit += 1 # add the root
        self.deep_copy(store=False) # restore the env to the start of the simulation


    def choose_neighbor(self):
        """
        choose action from the nearest position
        """
        if self.env.available_action_space.__len__() == 0: # check if there is any position can done
            self.env.terminal = True # set the flag
            return None

        min_num, max_num = 0, self.env.board_size**2-1 # the min and max action
        action = None # show the possible action
        action_available = set() # store the available action
        attention_place = self.trace_neighbor_place() #attention the place players have done
        
        return attention_place.pop()


    def trace_neighbor_place(self, delta=1):
        """
        return the neighbor place that the chess should be done first
        arg:
        |trace: set, the action done from the game start until now
        |delta: int, the range you want the return be
        """
        board_size = self.env.board_size # get the data from the env
        left_board = [i*board_size for i in range(board_size)] # get the left board position
        right_board = [board_size-1 + i*board_size for i in range(board_size)] # get the right board
        up_board = [i for i in range(board_size)] # get the up board
        down_board = [board_size**2-1-i for i in range(board_size)] # get the down board
        available_place = set() # for return, store the available places
        trace = set() # store the places have done

        """ find the trace """
        trace_x, trace_y = np.where(self.env.board_state != self.env.blank_id) # find the place of one 
        for idx in range(trace_x.__len__()):
            trace.add((trace_x[idx], trace_y[idx], trace_x[idx]*board_size+trace_y[idx])) # x,y,action

        """ use the trace to find the good place for simulation"""
        for x, y, last_move in trace:
            if last_move in left_board:
                neighbor_places = np.array([last_move, last_move+delta]) # no left element
            elif last_move in right_board:
                neighbor_places = np.array([last_move-delta, last_move]) # no right element
            else: 
                neighbor_places = np.array([last_move-delta, last_move, last_move+delta]) # row elements

            if last_move in up_board:
                neighbor_places = np.hstack((neighbor_places, neighbor_places.copy()+board_size*delta))
            elif last_move in down_board:
                neighbor_places = np.hstack((neighbor_places.copy()-board_size*delta,neighbor_places))
            else:
                neighbor_places = np.hstack((neighbor_places.copy()-board_size*delta,neighbor_places, neighbor_places.copy()+board_size*delta))

            for place in neighbor_places: # for every place to check if the place is available
                if place in self.env.available_action_space: 
                    available_place.add(place) # add the place to the set

        return available_place


    def deep_copy(self, store=True):
        """
        reset the env to the state that mcts simulation starts
        arg:
        |store: bool, true:store the state else: reset the state
        |available_action_space: list, the set for available action
        |invalid_action_space: list, the set for nonavailable actions
        |current_player_id: int, the player id
        |last_move: int, the number of the action
        """
        if  store == True:
            self.restore_data = (self.env.board_state.copy(), self.env.available_action_space.copy(), self.env.invalid_action_space.copy(), self.env.current_player_id, self.env.last_move) # store the data to reset
        else:
            self.env.reset() # use the env func to reset first
            self.env.board_state = self.restore_data[0].copy() # reset the board state
            self.env.available_action_space = self.restore_data[1].copy() # reset the action space back
            self.env.invalid_action_space = self.restore_data[2].copy() # reset the in~ action space bacl
            self.env.current_player_id = self.restore_data[3] # reset the player id
            self.env.last_move = self.restore_data[4] # reset the last move


    def policy_tree(self):
        """
        this func is for the selection and expassion in the mcts to choose action
        par:
        |state_now: array, the array of the state now
        """
        places = list(self.trace_neighbor_place()) # change to list
        return places[np.random.randint(places.__len__())]


    def policy_default(self):
        """
        this func is for the selection and expassion in the mcts to choose action
        par:
        |state_now: array, the array of the state now
        """
        places = list(self.trace_neighbor_place()) # change to list
        return places[np.random.randint(places.__len__())]


    def act(self):
        """
        use the random rule to make the choice
        arg:
        """
        self.mcts_tree = Node(player=self.env.current_player_id) # create the tree now
        self.deep_copy(store=True) # store the state

        """ do complete mcts in the limited max_num """
        for step in range(self.max_mcts_episode): # do the complete mcts
            reward = self.mcts_selection()
            if reward != None: # end in the selection func
                self.mcts_backup(reward)
                continue

            reward = self.mcts_simulation()
            self.mcts_backup(reward)

        """ get data from the children of the root, and choose the action by cnt of the visit """ 
        visits_children = [child.visit for child in self.mcts_tree.children] # get values of the firdt layer out
        values_children = [child.value for child in self.mcts_tree.children] # get values of the firdt layer out
        ratios_children = [child.value/child.visit for child in self.mcts_tree.children] # get values of the firdt layer out
        actions_children = [child.action for child in self.mcts_tree.children] # get values of the firdt layer out
        max_children_idx = np.where(np.max(values_children) == values_children)[0] # get the max value
        self.my_action = self.mcts_tree.children[max_children_idx[0]].action # return the first action

        return self.my_action
