import numpy as np

import os 

import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helpers.helpers import create_adjacent_mask_goat, create_adjacent_mask_tiger

class GoatIllegalPlacementError(Exception):
    pass

class GoatIllegalMoveError(Exception):
    pass

class SpaceOccupiedError(Exception):
    pass

class TigerIllegalMoveError(Exception):
    pass

class NotAGoatError(Exception):
    pass

class NotATigerError(Exception):
    pass


class Board():
    def __init__(self,size):
        #size of the board
        self.size = size
        #intialize the board
        self.state = np.zeros(((size,size)),dtype = int)
        #max number of goats
        self.max_number_of_goats  = self.size**2 - 2
        self.eaten_goats = 0

        self.place_tigers_in_corners()

    def place_tigers_in_corners(self):
        corners = [(0, 0), (0, self.state.shape[1] - 1),
                   (self.state.shape[0] - 1, 0), (self.state.shape[0] - 1, self.state.shape[1] - 1)]
        for x, y in corners:
            self.state[x, y] = 1  # Tiger

    def goat_placement(self,x,y):
        #if the new spot is free, place the goat anf increase the counter
        if self.state[x,y] == 0:
            self.state[x,y] = 2
        else:
            raise(GoatIllegalPlacementError)

    def goat_move(self,x,y,new_x,new_y):
        #if something other than a goat is selected raise an error
        if self.state[x,y] != 2:
            raise(NotAGoatError)
        #if the new position is further than one away from the goat, raise an error
        elif self.state[new_x,new_y] == 0:
            if abs(new_x - x) > 1 or abs(new_y - y) > 1:
                raise(GoatIllegalMoveError)
            else:
                #move the goat and free up the space
                self.state[x,y] = 0
                self.state[new_x,new_y] = 2
        else:
            raise(SpaceOccupiedError)

    
    def tiger_move(self, x, y, new_x, new_y, reward_scheme):
        tiger_reward, goat_reward = reward_scheme["no score"], 0

        # Ensure selected position is a tiger
        if self.state[x, y] != 1:
            raise NotATigerError

        try:
            if self.state[new_x, new_y] == 0:
                # Check for eating move
                if abs(new_x - x) == 2 or abs(new_y - y) == 2:
                    mid_x = int(x + (new_x - x) / 2)
                    mid_y = int(y + (new_y - y) / 2)
                    if self.state[mid_x, mid_y] == 2:
                        # Remove the goat
                        self.state[mid_x, mid_y] = 0
                        self.state[x, y] = 0
                        self.state[new_x, new_y] = 1
                        self.eaten_goats += 1
                        tiger_reward, goat_reward = reward_scheme["eating"], -2
                    else:
                        raise TigerIllegalMoveError
                # Normal move
                elif abs(new_x - x) <= 1 and abs(new_y - y) <= 1:
                    self.state[x, y] = 0
                    self.state[new_x, new_y] = 1
                else:
                    raise TigerIllegalMoveError
            else:
                raise SpaceOccupiedError
        except IndexError:
            raise TigerIllegalMoveError

        return tiger_reward, goat_reward

    def check_goat_win(self):
            done = True  # Assume all tigers are blocked

            tiger_positions = np.argwhere(self.state == 1)

            for x, y in tiger_positions:
                mask = create_adjacent_mask_tiger(self.state, x, y)
                if np.sum(mask) > 0:
                    done = False  # Found at least one tiger that can move
                    break

            return done

    def check_tiger_win(self, turn):
        if self.eaten_goats == self.size:
            return True
        check_for_goats_movement_availability = int((turn + 1) / 2) >= self.max_number_of_goats
        if check_for_goats_movement_availability:
            goat_positions = np.argwhere(self.state == 2)
            for x, y in goat_positions:
                if np.any(create_adjacent_mask_goat(self.state, x, y)):
                    return False
            return True
        return False



