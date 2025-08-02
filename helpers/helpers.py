 
import numpy as np

 
def create_adjacent_mask_goat(state, x, y):
    board_size = state.shape[0]
    # Create an empty mask of zeros
    mask = np.zeros((board_size, board_size), dtype=int)
    # Define the range for rows and columns, ensuring boundaries
    x_min = max(0, x - 1)
    x_max = min(board_size, x + 2)  # x + 2 because the upper bound is exclusive
    y_min = max(0, y - 1)
    y_max = min(board_size, y + 2)  # y + 2 because the upper bound is exclusive
    # Set the mask for the adjacent positions
    mask[x_min:x_max, y_min:y_max] = 1 

    # Exclude the center position (x, y) itself
    mask[x, y] = 0

    # Make occupied spaces illegal
    mask[state != 0] = 0

    return mask

def create_adjacent_mask_tiger(state, x, y):
    board_size = state.shape[0]
    mask = np.zeros((board_size, board_size), dtype=int)

    # All 8 directions: vertical, horizontal, and diagonal
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    for dx, dy in directions:
        adj_x, adj_y = x + dx, y + dy

        # Check simple adjacent move (1 step)
        if 0 <= adj_x < board_size and 0 <= adj_y < board_size:
            if state[adj_x, adj_y] == 0:
                mask[adj_x, adj_y] = 1

        # Check eating move (2 steps over a goat)
        next_x, next_y = x + 2 * dx, y + 2 * dy
        if (0 <= adj_x < board_size and 0 <= adj_y < board_size and
            0 <= next_x < board_size and 0 <= next_y < board_size):
            if state[adj_x, adj_y] == 2 and state[next_x, next_y] == 0:
                mask[next_x, next_y] = 1

    return mask


def place_goat(board, agent_prediction):
    """Place a goat on the board if the limit has not been exceeded."""
    legality_matrix = np.ones(board.state.shape)
    legality_matrix[board.state != 0] = 0
    # Randomly place the goat (if goat_agent is None)
    if agent_prediction is None:
        possible_moves = np.random.random(board.state.shape)
        free_spots = possible_moves * legality_matrix
        max_index = np.argmax(free_spots)
        x, y = np.unravel_index(max_index, board.state.shape)
    else:
        #print('agent_prediction',agent_prediction,board.state.shape[0])
        x,y = action_converter(agent_prediction,board.state.shape[0])
    # Place the goat and get reward
    board.goat_placement(x, y)
    return board

def move_goat(board, agent_prediction):
    """Move an already placed goat on the board."""
    # Randomly pick a goat and move (if goat_agent is None)
    if agent_prediction is None:
        available_goats = np.random.random(board.state.shape)
        available_goats[board.state != 2] = 0
        max_index_goat = np.argmax(available_goats)
        x, y = np.unravel_index(max_index_goat, board.state.shape)
        legal_moves_mask = create_adjacent_mask_goat(board.state, x, y)
        max_index_move = np.argmax(np.random.random(board.state.shape) * legal_moves_mask)
        new_x, new_y = np.unravel_index(max_index_move, board.state.shape)
    else:
        # sample the first size**2 number of actions, which determine which goat will be moved
        x,y = action_converter(agent_prediction[0], board.state.shape[0])
        # sample the last size**2 number of actions, which determine where the goat will be moved
        new_x,new_y = action_converter(agent_prediction[1], board.state.shape[0])
    # Move the goat 
    board.goat_move(x, y, new_x, new_y)
    return board

def action_converter(sampled_index, board_dimension):
        one_hot_action = np.zeros((1, board_dimension**2))
        one_hot_action[0][sampled_index] = 1
        one_hot_action = one_hot_action.reshape(board_dimension, board_dimension)
        # Find the index where the value is 1
        x, y = np.where(one_hot_action == 1)
        # Append the index as a tuple
        return x[0], y[0]


def move_tiger(board, agent_prediction, reward_scheme):
    """Move a tiger on the board."""
    if agent_prediction is None:
        # Randomly choose one of the existing tigers with possible legal moves
        tiger_positions = np.argwhere(board.state == 1)
        random_scores = np.random.random(board.state.shape)
        available_tigers = np.zeros(board.state.shape)

        for x, y in tiger_positions:
            if create_adjacent_mask_tiger(board.state, x, y).sum() > 0:
                available_tigers[x, y] = random_scores[x, y]

        max_index_tiger = np.argmax(available_tigers)
        x, y = np.unravel_index(max_index_tiger, board.state.shape)

        legal_moves_mask = create_adjacent_mask_tiger(board.state, x, y)
        move_scores = np.random.random(board.state.shape) * legal_moves_mask
        max_index_move = np.argmax(move_scores)
        new_x, new_y = np.unravel_index(max_index_move, board.state.shape)
    else:
        # Agent provides action: first selects tiger, then move
        x, y = action_converter(agent_prediction[0], board.state.shape[0])
        new_x, new_y = action_converter(agent_prediction[1], board.state.shape[0])

    rewards = board.tiger_move(x, y, new_x, new_y, reward_scheme)
    return board, rewards



def print_board_pretty(board):
    """
    Prints the game board (numpy array) in a clean, aligned format.
    Replaces 0, 1, 2 with symbols for better readability if desired.

    Example mapping:
        0 -> .
        1 -> T
        2 -> G

    Args:
        board (np.ndarray): The game board.
    """
    # Define symbols for each state (optional)
    symbol_map = {0: ".", 1: "T", 2: "G"}

    # Convert the board elements to strings using the mapping
    str_board = np.vectorize(lambda x: f" {symbol_map.get(x, str(x))} ")(board)

    # Build a single string for the board
    board_str = "\n".join(["".join(row) for row in str_board])

    print(board_str)




    
    
        

         