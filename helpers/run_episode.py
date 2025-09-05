
from board import Board

from helpers import place_goat, move_goat, move_tiger


def run_episode(size, max_number_of_turns, tiger_reward_scheme, tiger_agent=None, goat_agent=None, verbose=True):
    board = Board(size)
    turn = 1
    states = []
    goat_actions = []
    goat_rewards = []
    tiger_rewards = []
    victor = None

    while turn < max_number_of_turns:
        if turn % 2 != 0:
            # --- GOAT TURN ---
            all_goats_placed_flag = int((turn + 1) / 2) >= board.max_number_of_goats

            if goat_agent is None:
                # No agent: random placement or movement
                if not all_goats_placed_flag:
                    board = place_goat(board, None)
                    if verbose:
                        print('goats placed\n', board.state, '  move number: ', str(turn))
                    goat_actions.append("random_placement")
                else:
                    board = move_goat(board, None)
                    if verbose:
                        print('goats moved\n', board.state, '  move number: ', str(turn))
                    goat_actions.append("random_move")
            else:
                # Get goat agent prediction
                goat_or_spot_action, goat_move_action, _, _ = goat_agent.predict_action(board.state, all_goats_placed_flag)

                if not all_goats_placed_flag:
                    board = place_goat(board, goat_or_spot_action.item())
                    if verbose:
                        print('goats placed\n', board.state, '  move number: ', str(turn))
                    goat_actions.append(goat_or_spot_action.item())
                else:
                    board = move_goat(board, (goat_or_spot_action.item(), goat_move_action.item()))
                    if verbose:
                        print('goats moved\n', board.state, '  move number: ', str(turn))
                    goat_actions.append((goat_or_spot_action.item(), goat_move_action.item()))

            turn += 1

            if board.check_goat_win(board):
                if verbose:
                    print('goats won')
                tiger_rewards.append(tiger_reward_scheme["losing"])
                goat_rewards.append(3)
                victor = 'goats'
                break

        else:
            # --- TIGER TURN ---
            if tiger_agent:
                tiger_action, move_action, _,_= tiger_agent.predict_action(board.state)
                board, rewards = move_tiger(board, (tiger_action.item(), move_action.item()), tiger_reward_scheme)
            else:
                board, rewards = move_tiger(board, None, tiger_reward_scheme)

            states.append(board.state)
            tiger_rewards.append(rewards[0])
            goat_rewards.append(rewards[1])

            if board.check_tiger_win():
                goat_rewards[-1] = -3
                tiger_rewards[-1] = tiger_reward_scheme["winning"]
                victor = 'tigers'
                if verbose:
                    print('tiger won')
                break

            if verbose:
                print('tiger moved\n', board.state, '  move number: ', str(turn))

            turn += 1

    return states, goat_actions, goat_rewards, tiger_rewards, victor
