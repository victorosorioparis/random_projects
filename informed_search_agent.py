# INFORMED SEARCH AGENT WITH BFS AND DFS CAPABILITIES
# IMPLEMENTATION TO SOLVE 8-PUZZLES

import time
import resource
start_time = time.time()

def id_generator():
    first_id = 0
    while True:
        first_id += 1
        yield first_id
        
new_id = id_generator()
new_level = id_generator()

class state(object):
    '''
    Object that retrieves a state of the puzzle
    '''
    def __init__(self,puzzle,id_nr,parent,level,residual_cost,parent_action=None):
        self.id_nr = id_nr
        self.parent = parent
        self.level = level
        self.residual_cost = residual_cost
        self.puzzle = puzzle
        self.parent_action = parent_action
        self.flag = False
        
    def __str__(self):
        return str(self.puzzle[0:3]) + "\n" + str(self.puzzle[3:6]) + "\n" + str(self.puzzle[6:9])
    
    def get_puzzle(self):
        '''
        returns a 2 position tuple with: a) 2 position tuple of integers of sum of first half of puzzle
                                         and sum of second half of puzzle, b) puzzle itself
        '''
        return (sum(self.puzzle[0:4]),sum(self.puzzle[4:])),self.puzzle[:]
    
    def available_movements(self):
        '''
        returns a list of available swap options of 0
        '''
        pos0 = self.puzzle.index(0)
        available_mov = ['Up','Down','Left','Right']
        if pos0 in [6,7,8]:
            available_mov.remove('Down')
        if pos0 in [0,3,6]:
            available_mov.remove('Left')
        if pos0 in [0,1,2]:
            available_mov.remove('Up')
        if pos0 in [2,5,8]:
            available_mov.remove('Right')
        return available_mov
    
    def imagine_state(self,action):
        '''
        returns a new state object by applying the requested action to the current state
        '''
        a, b = self.get_puzzle()
        new_state = state(b,next(new_id),self.id_nr,self.level + 1,self.residual_cost + 1,action)
        pos0 = new_state.puzzle.index(0)
        if action == 'Down':
            pos_swap = pos0 + 3
        elif action == 'Up':
            pos_swap = pos0 - 3
        elif action == 'Left':
            pos_swap = pos0 - 1
        else:
            pos_swap = pos0 + 1
        new_state.puzzle[pos0], new_state.puzzle[pos_swap] = new_state.puzzle[pos_swap], new_state.puzzle[pos0]
        return new_state
    
    def is_goal(self):
        for a, b in zip(self.puzzle,[0,1,2,3,4,5,6,7,8]):
            if a == b:
                pass
            else:
                return False
        return True
    

class Mauritius(object):
    '''
    agent to solve 8-puzzle problems
    '''
    def __init__(self,method,initial_state):
        self.method = method
        self.fringe = [initial_state]
        in_key, b = initial_state.get_puzzle()
        self.fringe_puzzles = {}
        self.fringe_puzzles[in_key] = [b]
        self.explored = {}
        self.explored_dict = {}
        self.satisfied = False
        self.path = []
        self.m = 1
        self.d = 1
    
    def get_path(self,state):
        '''
        INPUT: a state
        OUTPUT: a list of all vertical actions from initial state up until that state
        '''
        actions_sequence = [self.explored_dict[state.id_nr][0]]
        current_id = state.parent
        while current_id != 1:
            actions_sequence.append(self.explored_dict[current_id][0])
            current_id = self.explored_dict[current_id][1]
        actions_sequence.reverse()
        return actions_sequence
    
    def get_cost(self):
        '''
        RETURNS cost of solution path
        '''
        return len(self.path)
    
    def get_nodes(self):
        '''
        RETURNS number of nodes expanded
        '''
        # It removes the initial node which is not explored
        return len(self.explored_dict.keys()) - 1
    
    def explore(self):
        if self.method == 'bfs':
            while not self.satisfied:
                current_state = self.fringe[0]
                self.fringe.remove(current_state)
                state_key, state_val = current_state.get_puzzle()
                self.fringe_puzzles[state_key].remove(state_val)
                if state_key in self.explored.keys():
                    self.explored[state_key] += [state_val]
                else:
                    self.explored[state_key] = [state_val]
                self.explored_dict[current_state.id_nr] = (current_state.parent_action,current_state.parent)
                if current_state.is_goal():
                    self.path = self.get_path(current_state)
                    self.d = current_state.level - 1
                    self.m = self.fringe[-1].level - 1
                    self.satisfied = True
                else:
                    for i in current_state.available_movements():
                        child_state = current_state.imagine_state(i)
                        child_key, child_val = child_state.get_puzzle()
                        if child_key in self.fringe_puzzles.keys():
                            if child_key not in self.explored.keys():
                                self.explored[child_key] = [1]
                            if child_val not in self.explored[child_key] and child_val not in self.fringe_puzzles[child_key]:
                                self.fringe.append(child_state)
                                self.fringe_puzzles[child_key] += [child_val]
                        else:
                            self.fringe.append(child_state)
                            self.fringe_puzzles[child_key] = [child_val]
        if self.method == 'dfs':
            while not self.satisfied:
                current_state = self.fringe.pop()
                self.explored.append(current_state.get_puzzle())
                self.explored_dict[current_state.id_nr] = (current_state.parent_action,current_state.parent)
                if current_state.is_goal():
                    self.path = self.get_path(current_state)
                    self.d = current_state.level - 1
                    self.m = self.fringe[-1].level - 1
                    self.satisfied = True
                else:
                    mov_list = current_state.available_movements()
                    mov_list.reverse()
                    for i in mov_list:
                        child_state = current_state.imagine_state(i)
                        if child_state.get_puzzle() not in self.explored and child_state.get_puzzle() not in self.fringe_puzzles:
                            self.fringe.append(child_state)
                            self.fringe_puzzles.append(child_state.get_puzzle())
                            
    def see_explored(self):
        for i in self.explored:
            print(i[0:3],i[3:6],i[6:9],sep="\n")
            print("\n\n")
            

        
    
test = state([1,2,5,3,4,0,6,7,8],next(new_id),0,1,1)
new_agent = Mauritius('bfs',test)
new_agent.explore()

print(f"SOLUTION: {new_agent.path}")

# METRICS
print("--- %s seconds ---" % (time.time() - start_time))
print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000000)