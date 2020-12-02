#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:33:13 2020

It solves Sudoku problems by trying with an AC3 and then a BTS approach

It takes as input a string of 81 numbers corresponding to the original Sudoku problem (empty = 0)

It outputs a text file with the solution and the approached used

@author: victorosorio
"""

import sys
import time
from itertools import product

try:
    text = sys.argv[1]
except:
    text = '000000000000942080160000029000000008906000001400250000004000000020008090050000700'

variables = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9']

starting_t = time.process_time()

class sudoku_state(object):
    def __init__(self, text):
        self.text = text
        self.zeros_list = []
        for a, b in zip(variables,text):
            self.vars_map = {a : int(b)}
            if b == '0':
                self.zeros_list += [a]
        self.vars_map = {a : int(b) for a, b in zip(variables,text)}
        self.variables = variables
    
    def check_binary_constraint(self,position_value1,position_value2):
        original_map = self.vars_map.copy()
        self.vars_map[position_value1[0]] = position_value1[1]
        self.vars_map[position_value2[0]] = position_value2[1]
        letter = position_value1[0][0]
        column = position_value1[0][1]
        good_2go = all([self.check_row(letter),self.check_column(column),self.check_box(position_value1[0])])
        self.vars_map = original_map
        return good_2go
    
    def return_zeros(self):
        '''
        Returns
        -------
        A list of variables that have zero in them.
        '''
        variables_list = []
        for i in range(81):
            if int(self.text[i]) == 0:
                variables_list += [self.variables[i]]
        return variables_list
    
    def check_row(self,row):
        first_cell = {'A':0,'B':9,'C':18,'D':27,'E':36,'F':45,'G':54,'H':63,'I':72}
        new_list = []
        for i in range(first_cell[row],first_cell[row]+9):
            new_list += [self.vars_map[variables[i]]]
        frequency = {i:new_list.count(i) for i in new_list if i != 0}
        if len(frequency) == 0:
            return True
        if max(frequency.values()) > 1:
            return False
        else:
            return True
    
    def check_column(self,row):
        first_cell = {'1':0,'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8}
        new_list = []
        for i in range(first_cell[row],81,9):
            new_list += [self.vars_map[variables[i]]]
        frequency = {i:new_list.count(i) for i in new_list if i != 0}
        if len(frequency) == 0:
            return True
        if max(frequency.values()) > 1:
            return False
        else:
            return True
    
    def check_box(self,cell):
        boxes = [('ABC','123'),('ABC','456'),('ABC','789'),('DEF','123'),('DEF','456'),('DEF','789'),('GHI','123'),('GHI','456'),('GHI','789')]
        letter, column, correct_box = cell[0], cell[1], None
        for box in boxes:
            if letter in box[0] and column in box[1]:
                correct_box = box
                break
        new_list = []
        for letter in correct_box[0]:
            for column in correct_box[1]:
                new_list += [self.vars_map[letter + column]]
        frequency = {i:new_list.count(i) for i in new_list if i != 0}
        if len(frequency) == 0:
            return True
        if max(frequency.values()) > 1:
            return False
        else:
            return True
        
    def check_constraints(self):
        letters = 'ABCDEFGHI'
        rows_check = [self.check_row(i) for i in letters] 
        columns_check = [self.check_column(str(i)) for i in range(1,10)]
        boxes_check = [self.check_box(a + str(b)) for a, b in zip(letters,[1,4,7,1,4,7,1,4,7])]
        if all(rows_check+columns_check+boxes_check):
            return True
        else:
            return False

class AC3_bot(object):
    def __init__(self,problem):
        self.problem = problem
        self.arcs_queue = []
        self.variables = self.problem.return_zeros()
        self.var_vals = {x:[1,2,3,4,5,6,7,8,9] for x in self.variables}
        self.solved = False
    
    def fill_queue(self):
        self.arcs_queue = [i for i in list(product(self.variables,self.variables,repeat=1)) if i[0] != i[1]]
        
    def replenish_queue(self,X):
        self.arcs_queue += [i for i in list(product((X,),self.variables,repeat=1)) if i[0] != i[1] and i not in self.arcs_queue]
        
    def check_arc(self,arc):
        X = arc[0]
        Y = arc[1]
        values_pos1 = [*self.var_vals[X]]
        values_pos2 = [*self.var_vals[Y]]
        revision_X = False
        revision_Y = False
        for x in values_pos1:
            consistent = False
            for y in values_pos2:
                if self.problem.check_binary_constraint((X,x),(Y,y)):
                    consistent = True
                    break
            if not consistent:
                self.var_vals[X].remove(x)
                revision_X = True
        values_pos1 = [*self.var_vals[X]]
        values_pos2 = [*self.var_vals[Y]]
        for y in values_pos2:
            consistent = False
            for x in values_pos1:
                if self.problem.check_binary_constraint((Y,y),(X,x)):
                    consistent = True
                    break
            if not consistent:
                self.var_vals[Y].remove(y)
                revision_Y = True
        return revision_X, revision_Y
    
    def check_next_arc(self):
        new_arc = self.arcs_queue[0]
        revised_X, revised_Y = self.check_arc(new_arc)
        if any((revised_X, revised_Y)):
            if revised_X:
                self.replenish_queue(new_arc[0])
            elif revised_Y:
                self.replenish_queue(new_arc[1])
            self.check_next_arc()
        del(self.arcs_queue[0])
    
    def unsolvable(self):
        for i in self.var_vals.values():
            if len(i) == 0:
                return True
            else:
                return False
            
    def find_solution(self):
        while len(self.arcs_queue) > 0:
            self.check_next_arc()
            if self.unsolvable() or (time.process_time() - starting_t > 2):
                return
        temp_list = list(map(len,self.var_vals.values()))
        if max(temp_list) != 1:
            self.fill_queue()
            self.find_solution()
        else:
            print("SOLVED WITH AC3!")
            self.solved = True
            
    def get_solution(self):
        new_text = ''
        solution = list(self.var_vals.values())
        count = 0
        for i in self.problem.text:
            if i == '0':
                new_text += str(solution[count][0])
                count += 1
            else:
                new_text += i
        return new_text

class BTS_bot(object):
    def __init__(self,problem):
        self.problem = problem
        self.children_queue = []
        self.variables = self.problem.zeros_list
        self.var_vals = {x:[1,2,3,4,5,6,7,8,9] for x in self.variables}
        self.var_vals_values = []
    
    # O**n
    def update_values(self):
        for var in self.var_vals.keys():
            values = [*self.var_vals[var]]
            for val in values:
                original_dict = self.problem.vars_map.copy()
                self.problem.vars_map[var] = val
                if not self.problem.check_constraints():
                    self.var_vals[var].remove(val)
                self.problem.vars_map = original_dict
                self.var_vals_values = list(self.var_vals.values())
                
    # O*n
    def find_next_zero(self,text,iterations):
        index = None
        iteration = 0
        for i in range(len(text)):
            if text[i] == '0': 
                if iteration == iterations:
                    index = i
                    break
                else:
                    iteration += 1
        return index
      
    # O*n
    def generate_children(self,parent_problem,variable,values_list):
        base_text = parent_problem.problem.text
        variable_dict = parent_problem.variables
        variable_i = variable_dict.index(variable)
        index = self.find_next_zero(base_text, variable_i)
        children_list = []
        valslength_list = []
        for value in values_list:
            new_text = base_text[:index] + str(value) + base_text[index + 1:]
            new_problem = sudoku_state(new_text)
            new_child = BTS_bot(new_problem)
            new_child.update_values()
            if new_problem.check_constraints() and not new_child.contains_empty_variables():
                children_list += [new_child]
                valslength_list += [self.get_lcv(new_child)]
        while len(children_list) > 0:
            most_constraining = min(valslength_list)
            most_constraining = valslength_list.index(most_constraining)
            self.children_queue.insert(0,children_list[most_constraining])
            del(children_list[most_constraining])
            del(valslength_list[most_constraining])
    
    # O*n
    def contains_empty_variables(self):
        for i in self.var_vals_values:
            if len(i) == 0:
                return True
        return False
    
    # O*n
    def get_lcv(self,new_problem):
        new_bot_values = [len(i) for i in new_problem.var_vals_values]
        return sum(new_bot_values)
        
    # O*n
    def choose_best_var(self,parent_problem):
        lengths = [len(i) for i in parent_problem.var_vals_values]
        min_length = min(lengths)
        index = lengths.index(min_length)
        variable = parent_problem.variables
        variable = variable[index]
        return variable
    
    # O*n
    def explore_child(self,child):
        while len(child.problem.zeros_list) > 0:
            variable = self.choose_best_var(child)
            self.generate_children(child,variable,child.var_vals[variable])
            child = self.children_queue.pop(0)
        print("SOLVED WITH BTS!")
        with open('output.txt', 'w', newline='') as output:
            output.write(child.problem.text + " BTS")
    
    def start_search(self):
        problem = self.problem
        root = BTS_bot(problem)
        root.update_values()
        self.explore_child(root)

if __name__ == '__main__':

    print("TRYING WITH AC3 (max 5 seconds)")
    my_state = sudoku_state(text)
    
    my_bot = AC3_bot(my_state)
    
    my_bot.fill_queue()
    
    my_bot.find_solution()
    
    if my_bot.solved:
        print()
        with open('output.txt', 'w') as output:
            output.write(my_bot.get_solution() + " AC3")
    else:
            
        AC3_pruned_p, AC3_pruned_v = my_bot.problem, my_bot.var_vals.copy()
        
        print("BETTER TRY WITH BTS!")
        
        my_bot = BTS_bot(AC3_pruned_p)
        
        my_bot.var_vals = AC3_pruned_v
        
        my_bot.start_search()
