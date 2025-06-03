
from board import Board

from helpers import place_goat, move_goat, move_tiger

class TIGER_ENV():     
    def __init__(self,size,max_number_of_turns,reward_scheme,goat_agent = None):
        self.size = size
        self.max_number_of_turns = max_number_of_turns
        self.goat_agent = goat_agent
        self.board = Board(size) 
        self.turn = 1
        self.board = place_goat(self.board, self.goat_agent)
        self.reward_scheme = reward_scheme
    
    def return_state(self):
        return self.board.state

    def step(self,action):
        self.board,rewards = move_tiger(self.board, action,self.reward_scheme)
        #CHECK TIGER WIN
        tiger_won = self.board.check_tiger_win()
        if tiger_won:
            reward = self.reward_scheme["winning"]
            return reward
        self.turn += 1
        if  int((self.turn + 1) / 2) < self.board.max_number_of_goats:
            self.board = place_goat(self.board, self.goat_agent)
            self.turn += 1
        else:
            self.board,reward = move_goat(self.board, self.goat_agent,self.reward_scheme,None)      
            if reward == self.reward_scheme["winning"]:
                return reward
            self.turn += 1 
        goats_won = self.board.check_goat_win(self.board)
        if goats_won:
            reward = self.reward_scheme["losing"]
        else:
            reward = rewards[0] 
        return reward
    

    def reset(self):
        self.board = Board(self.size)
        self.turn = 1
        self.board = place_goat(self.board, self.goat_agent)