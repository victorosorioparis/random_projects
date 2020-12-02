'''
Minimax game agent for playing 2048
It is to be executed in the framework of the 2048 from Columbia University
'''

from BaseAI import BaseAI
import time

timeLimit = 0.19

def get_children(grid):
    '''
    Input: a grid
    Output: a list of children grids
    '''
    children = []
    for i in (0,2,1,3):
        new_grid = grid.clone()
        new_grid.move(i)
        new_grid.parent = i
        children += [new_grid]
    return children

def get_children_mini(grid):
    '''
    Input: a grid
    Output: a list of children grids base on possible moves of min
    '''
    children = []
    empty_tyles = grid.getAvailableCells()
    for i in empty_tyles:
        new_grid = grid.clone()
        new_grid.insertTile(i,2)
        new_grid.parent = i
        children += [new_grid]
    for i in empty_tyles:
        new_grid = grid.clone()
        new_grid.insertTile(i,4)
        new_grid.parent = i
        children += [new_grid]
    return children

class PlayerAI(BaseAI):

    def try_move(self,grid,move):
        new_grid = grid.clone()
        new_grid.move(move)
        return new_grid

    # heuristic 1
    def check_max_values(self,grid,move):
        new_grid = self.try_move(grid,move)
        return new_grid.getMaxTile()

    def try_all_possible_moves(self,grid):
        dict_moves = {0:1,1:1,2:1,3:1}
        for i in dict_moves.keys():
            dict_moves[i] = self.check_max_values(grid,i)
        return dict_moves

    # heuristic 2
    def estimate_average_value(self,grid,move):
        new_grid = grid.clone()
        new_grid.move(move)
        total_val = 0
        count_val = 0
        for i in new_grid.map:
            for j in i:
                if j != 0:
                    count_val += 1
                    total_val += j
        return total_val / count_val

    def estimate_average_h(self,grid):
        dict_moves = {0:0,1:0,2:0,3:0}
        for i in dict_moves.keys():
            dict_moves[i] = self.estimate_average_value(grid,i)
        return dict_moves

    # heuristic 3
    def estimate_average(self,grid):
        total_val = 0
        count_val = 0
        for i in grid.map:
            for j in i:
                if j != 0:
                    count_val += 1
                    total_val += j
        mix_points = 0
        mono_points = 0
        for i in grid.map:
            for j in range(3):
                if i[j] == i[j+1]:
                    mix_points += i[j]
                if i[j] < i[j+1]:
                    mono_points += i[j+1]
        for i in range(3):
            for j in range(4):
                if grid.map[i][j] == grid.map[i+1][j]:
                    mix_points += grid.map[i][j]
                if grid.map[i][j] > grid.map[i+1][j]:
                    mono_points += grid.map[i][j]
        # smoothness
        roughness = 0
        for i in grid.map:
            for j in range(3):
                roughness += (i[j+1] - i[j])
        # free_tiles_penalty
        used_cells = 16 - len(grid.getAvailableCells())
        # max tiles
        max_tile = grid.getMaxTile()
        return (4 * total_val / count_val) + (0.5 * mix_points) + (4 * max_tile) - (0.05 * roughness) - (6 * used_cells)
   

    # minimax algorithm
    def get_max(self,grid,alpha,beta,initial_time):
        if time.clock() - initial_time > 0.019:
            return None, self.estimate_average(grid)

        maxChild, maxUtility = None, float('-inf')

        for child in get_children(grid):
            subchild, utility = self.get_mini(child, alpha, beta, initial_time)

            if utility > maxUtility:
                maxChild, maxUtility = child, utility

            if utility >= beta:
                break

            if maxUtility > alpha:
                alpha = maxUtility

        return maxChild, maxUtility

        dict_moves = self.estimate_average_h2(grid)
        max_move = max(dict_moves, key=dict_moves.get)
        max_average = dict_moves[max_move]
        return max_move, max_average

    def get_mini(self, grid, alpha, beta, initial_time):
        if time.clock() - initial_time > 0.019:
            return None, self.estimate_average(grid)

        minChild, minUtility = None, float('inf')

        for child in get_children(grid):
            subchild, utility = self.get_max(child, alpha, beta, initial_time)

            if utility < minUtility:
                minChild, minUtility = child, utility

            if utility <= alpha:
                break

            if minUtility < beta:
                beta = minUtility

        return minChild, minUtility

    def getMove(self,grid):
        child, utility = self.get_max(grid,float('-inf'),float('inf'),time.clock())
        return child.parent


    def getMoveOriginal(self, grid):
        dict_moves = self.estimate_average_h(grid)
        return max(dict_moves, key=dict_moves.get)