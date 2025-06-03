
from board import Board
    
from helpers import place_goat, move_goat, move_tiger

class GOAT_ENV():
    def __init__(self,size,max_number_of_turns,goat_reward_scheme,tiger_reward_scheme,tiger_agent = None):
        self.size = size
        self.max_number_of_turns = max_number_of_turns
        self.tiger_agent = tiger_agent
        self.board = Board(size) 
        self.turn = 0
        self.goat_reward_scheme = goat_reward_scheme
        self.tiger_reward_scheme = tiger_reward_scheme

    def return_state(self):
        return self.board.state
    
    def return_goat_placement_flag(self):
        #print(self.board.eaten_goats,self.board.max_number_of_goats)
        #print(int((self.turn + 1) / 2))
        return int(int((self.turn + 1) / 2) >= self.board.max_number_of_goats) #0 if not (not all goats placed), 1 if yes (all goats placed)
    
    def step(self,action):
        reward = 0
        if  int((self.turn + 1) / 2) < self.board.max_number_of_goats:
            self.board = place_goat(self.board, action)
            self.turn += 1
        else:
            self.board = move_goat(self.board, action)
            self.turn += 1 
        goats_won = self.board.check_goat_win(self.board)
        if goats_won:
            reward = self.goat_reward_scheme["winning"]
            return reward
        self.board,tiger_reward = move_tiger(self.board, None,self.tiger_reward_scheme)
        self.turn += 1
        tiger_won = self.board.check_tiger_win()
        if tiger_won:
            reward = self.goat_reward_scheme["losing"]
        elif tiger_reward == self.tiger_reward_scheme["eating"]:
            reward = self.goat_reward_scheme["eaten"]
        return reward

    
    def reset(self):
        self.board = Board(self.size)
        self.turn = 0

