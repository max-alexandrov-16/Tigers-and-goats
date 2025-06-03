import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
import torch.nn.init as init
from helpers import create_adjacent_mask_goat,action_converter

class ReinforceGoatAgent():

    def __init__(self, model,goat_env,board_dimension,reward_scheme,max_number_of_turns = 40):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-7,weight_decay=1e-6)
        self.gamma = 0.99
        self.board_dimension = board_dimension
        self.goat_env = goat_env
        self.max_number_of_turns = max_number_of_turns
        self.reward_scheme = reward_scheme
        self.monitoring_state = None
        self.monitoring_distributions = []
    

    def update_policy(self, log_probs, rewards):
        """
        Updates the policy by applying the REINFORCE algorithm.
        
        Args:
            log_probs (Tensor): Tensor of log probabilities of taken actions.
            rewards (Tensor): Tensor of cumulative discounted rewards.
        """
        self.optimizer.zero_grad()  # Clear previous gradients

        # Normalize rewards for stability in training
        normalized_rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)

        # Compute the loss (negative expected reward)
        loss = - torch.sum(normalized_rewards * log_probs)
        loss.backward()  # Backpropagate the loss

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()  # Update model parameters


    def calculate_cum_reward(self, rewards):
        """
        Calculates cumulative discounted rewards for each timestep using the discount factor gamma.
        
        Args:
            rewards (list): A list of rewards received during an episode.

        Returns:
            Tensor: A tensor of discounted cumulative rewards.
        """
        G = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            G.insert(0, R)
        return torch.tensor(G, dtype=torch.float32)



    def learn(self):
        """
        Runs one training episode using the current policy. Collects states, actions, and rewards,
        then updates the policy based on the outcomes.

        Returns:
            float: The average reward received during the episode.
        """
        rewards = []
        log_probs = []

        for _ in range(self.max_number_of_turns):
            # Get current environment state and placement phase flag
            current_state = self.goat_env.return_state()
            all_goats_placed_flag = self.goat_env.return_goat_placement_flag()

            # Predict the action using the policy
            goat_or_spot_action, goat_move_action, goat_or_spot_dist, goat_move_dist = self.predict_action(current_state, all_goats_placed_flag)

            # Combine log probabilities depending on phase
            if goat_move_dist is not None:
                # Moving a goat: two-stage action
                log_prob = goat_or_spot_dist.log_prob(goat_or_spot_action) + goat_move_dist.log_prob(goat_move_action)
                action = (goat_or_spot_action.item(), goat_move_action.item())  # Tuple for movement
            else:
                # Placing a goat: single action
                log_prob = goat_or_spot_dist.log_prob(goat_or_spot_action)
                action = goat_or_spot_action.item()

            # Take action in environment and receive reward
            reward = self.goat_env.step(action)

            # Log reward and log probability
            rewards.append(reward)
            log_probs.append(log_prob)

            # Check for terminal condition
            if reward == self.reward_scheme["losing"] or reward == self.reward_scheme["winning"]:
                    break

        # Compute cumulative rewards and update policy
        cum_rewards = self.calculate_cum_reward(rewards)
        log_probs = torch.stack(log_probs)
        self.update_policy(log_probs, cum_rewards)

        # Reset environment for the next episode
        self.goat_env.reset()

        return np.mean(rewards)  # Return average reward of the episode
      
    def check_output(self):
        current_state = self.tiger_env.return_state()
        action_probs = self.model.predict_action(current_state)
        print('raw action probs',action_probs)

    def _prepare_model_input(self, state, all_goats_placed_flag):
        """
        Flattens the board state and appends the goat placement flag to form the input for the model.
        """
        flattened_state = np.reshape(state, (1, state.shape[0] ** 2))  # Flatten 2D state to 1D
        all_goats_placed_flag = torch.tensor(all_goats_placed_flag, dtype=torch.float32).view(1, 1)  # Make flag a 2D tensor
        flattened_state = torch.tensor(flattened_state, dtype=torch.float32)  # Convert numpy to tensor
        return torch.cat((flattened_state, all_goats_placed_flag), dim=1)  # Concatenate along the feature axis


    def _split_action_probs(self, action_probs, size):
        """
        Splits the model output into two distributions:
        - One for selecting a goat or a placement spot
        - One for selecting a destination move for a goat
        """
        goat_or_spot_selection = F.softmax(action_probs[0, 0:size ** 2], dim=0)  # First half of logits
        goat_move_selection = F.softmax(action_probs[0, size ** 2:], dim=0)  # Second half of logits
        return goat_or_spot_selection, goat_move_selection


    def _get_available_spots(self, state, all_goats_placed_flag):
        """
        Returns a mask indicating which spots are valid for placing or selecting goats.
        If all goats are placed, this also checks if the goat can actually move (not stuck).
        """
        available_spots = state == 0 if not all_goats_placed_flag else state == 2  # Determine base eligibility

        if all_goats_placed_flag:
            size = state.shape[0]
            for idx in range(size * size):
                if available_spots.flatten()[idx] == 1:
                    x, y = divmod(idx, size)  # Convert flat index to coordinates
                    legality = create_adjacent_mask_goat(state, x, y)  # Check move legality for goat
                    if np.sum(legality) == 0:  # Goat is stuck
                        available_spots[x, y] = 0  # Mark goat as unavailable

        return torch.tensor(available_spots.flatten(), dtype=torch.float32)  # Return mask as flat tensor


    def _filter_goat_moves(self, state, goat_or_spot_action, goat_move_selection):
        """
        Filters the goat move probabilities to only include valid destination spots for the selected goat.
        """
        x, y = action_converter(goat_or_spot_action, state.shape[0])  # Get coordinates of selected goat
        move_legality_matrix = create_adjacent_mask_goat(state, x, y)  # Create legality matrix for its moves
        legality_tensor = torch.tensor(move_legality_matrix.flatten(), dtype=torch.float32)  # Flatten and convert
        #print('legality of move tensor', legality_tensor.view(state.shape[0], state.shape[0]))

        goat_move_selection =goat_move_selection * legality_tensor  # Apply legality mask
        #print('filtered goat move selection', goat_move_selection.view(state.shape[0], state.shape[0]))
        return goat_move_selection
    
    def predict_action(self, state, all_goats_placed_flag):
        # Prepare input for the neural network
        model_input = self._prepare_model_input(state, all_goats_placed_flag)
        #print('state',state)
        #print('all_goats_placed_flag',all_goats_placed_flag)
        # Get action probabilities from the model
        #print('model input',model_input)
        action_probs = self.model.predict_probabilities(model_input)
        #print('action probs',action_probs)

        # Separate probabilities into goat/spot selection and movement
        goat_or_spot_selection, goat_move_selection = self._split_action_probs(action_probs, state.shape[0])

        # Get mask of valid goat/spot selections depending on game phase
        available_spots = self._get_available_spots(state, all_goats_placed_flag)

        # Filter selection probabilities using the mask
        goat_or_spot_selection = goat_or_spot_selection * available_spots
        #print('filtered goat or spot selection', goat_or_spot_selection.view(state.shape[0], state.shape[0]))

        # Normalize the distribution
        if goat_or_spot_selection.sum() > 0:
            goat_or_spot_selection /= goat_or_spot_selection.sum()
        else:
            return ValueError("Goat or spot selection probabilities sum to 0")

        # Sample a spot or goat to move
        goat_or_spot_dist = Categorical(goat_or_spot_selection)
        goat_or_spot_action = goat_or_spot_dist.sample()
        #if not all_goats_placed_flag:
            #print('spot selected', goat_or_spot_action)

        # If in movement phase, process movement distribution
        if all_goats_placed_flag:
            #print('goat chosen to move', goat_or_spot_action)
            goat_move_selection = self._filter_goat_moves(state, goat_or_spot_action, goat_move_selection)
            #print('filtered goat move selection', goat_move_selection.view(state.shape[0], state.shape[0]))
            # Normalize move distribution
            if goat_move_selection.sum() > 0:
                goat_move_selection /= goat_move_selection.sum()
            else:
                return ValueError("Goat move selection probabilities sum to 0")

            # Sample move action
            goat_move_dist = Categorical(goat_move_selection)
            goat_move_action = goat_move_dist.sample()

            return goat_or_spot_action, goat_move_action, goat_or_spot_dist, goat_move_dist

        return goat_or_spot_action, None, goat_or_spot_dist, None
    

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))












'--------------------------------------------------------------------------------------------------------------------------------------'
class ActorCriticGoatAgent():

    def __init__(self, actor_model,critic_model,goat_env,board_dimension,reward_scheme,max_number_of_turns = 40):
        self.actor = actor_model
        self.critic = critic_model
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-5,weight_decay=1e-6)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-5,weight_decay=1e-6)
        self.gamma = 0.99
        self.board_dimension = board_dimension
        self.goat_env = goat_env
        self.max_number_of_turns = max_number_of_turns
        self.reward_scheme = reward_scheme
        self.monitoring_state = None
        self.monitoring_distributions = []
    

    def update_actor(self, log_prob, advantage):
        """
        Performs a per-step actor update using the policy gradient and advantage estimate.
        
        Args:
            log_prob (Tensor): Log probability of the action taken.
            advantage (float or Tensor): Advantage estimate (e.g., from critic).
        """
        self.actor_optimizer.zero_grad()

        # Policy gradient loss using advantage
        loss = -log_prob * advantage
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

        self.actor_optimizer.step()



    def update_critic(self, predicted_value,target_value):
        """
        Updates the critic (value network) by minimizing the squared TD error.

        Args:
            state: The current environment state.
            all_goats_placed_flag (bool): Flag indicating the placement phase.
            target_value (float): The TD target value (r + γ * V(s')).
        """
        # Convert target to tensor if not already
        if not isinstance(target_value, torch.Tensor):
            target_value = torch.tensor([target_value], dtype=torch.float32)

        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(predicted_value, target_value)

        # Update critic
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()


    def learn(self):
        """
        Runs one training episode using the current policy. Collects states, actions, and rewards,
        and updates both actor and critic per step using TD-based advantage.

        Returns:
            float: The average reward received during the episode.
        """
        total_reward = 0
        step_count = 0

        for _ in range(self.max_number_of_turns):
            # Get current environment state and placement phase flag
            current_state = self.goat_env.return_state()
            all_goats_placed_flag = self.goat_env.return_goat_placement_flag()

            # Predict the action using the policy
            goat_or_spot_action, goat_move_action, goat_or_spot_dist, goat_move_dist = self.predict_action(current_state, all_goats_placed_flag)
            
            # Estimate value of current state
            value = self.predict_value(current_state, all_goats_placed_flag)

            # Combine log probabilities depending on phase
            if goat_move_dist is not None:
                log_prob = goat_or_spot_dist.log_prob(goat_or_spot_action) + goat_move_dist.log_prob(goat_move_action)
                action = (goat_or_spot_action.item(), goat_move_action.item())
            else:
                log_prob = goat_or_spot_dist.log_prob(goat_or_spot_action)
                action = goat_or_spot_action.item()

            # Take action and receive reward
            reward = self.goat_env.step(action)
            total_reward += reward
            step_count += 1

            # Get next state value (for TD target)
            next_state = self.goat_env.return_state()
            next_all_goats_placed_flag = self.goat_env.return_goat_placement_flag()
            next_value = self.predict_value(next_state, next_all_goats_placed_flag)

            # If terminal, set next_value to 0 (no future value)
            done = reward == self.reward_scheme["losing"] or reward == self.reward_scheme["winning"]
            if done:
                next_value = torch.tensor([0.0], dtype=torch.float32)

            # Compute advantage: TD error
            advantage = reward + self.gamma * next_value.detach() - value.detach()

            # Update actor using advantage
            self.update_actor(log_prob, advantage)

            # Update critic to fit target value
            target_value = reward + self.gamma * next_value.detach()
            self.update_critic(value, target_value)

            if done:
                break

        self.goat_env.reset()
        return total_reward if step_count > 0 else 0.0


      
    def check_output(self):
        current_state = self.tiger_env.return_state()
        action_probs = self.model.predict_action(current_state)
        print('raw action probs',action_probs)

    def _prepare_model_input(self, state, all_goats_placed_flag):
        """
        Flattens the board state and appends the goat placement flag to form the input for the model.
        """
        flattened_state = np.reshape(state, (1, state.shape[0] ** 2))  # Flatten 2D state to 1D
        all_goats_placed_flag = torch.tensor(all_goats_placed_flag, dtype=torch.float32).view(1, 1)  # Make flag a 2D tensor
        flattened_state = torch.tensor(flattened_state, dtype=torch.float32)  # Convert numpy to tensor
        return torch.cat((flattened_state, all_goats_placed_flag), dim=1)  # Concatenate along the feature axis


    def _split_action_probs(self, action_probs, size):
        """
        Splits the model output into two distributions:
        - One for selecting a goat or a placement spot
        - One for selecting a destination move for a goat
        """
        goat_or_spot_selection = F.softmax(action_probs[0:size ** 2], dim=0)  # First half of logits
        goat_move_selection = F.softmax(action_probs[size ** 2:], dim=0)  # Second half of logits
        return goat_or_spot_selection, goat_move_selection


    def _get_available_spots(self, state, all_goats_placed_flag):
        """
        Returns a mask indicating which spots are valid for placing or selecting goats.
        If all goats are placed, this also checks if the goat can actually move (not stuck).
        """
        available_spots = state == 0 if not all_goats_placed_flag else state == 2  # Determine base eligibility

        if all_goats_placed_flag:
            size = state.shape[0]
            for idx in range(size * size):
                if available_spots.flatten()[idx] == 1:
                    x, y = divmod(idx, size)  # Convert flat index to coordinates
                    legality = create_adjacent_mask_goat(state, x, y)  # Check move legality for goat
                    if np.sum(legality) == 0:  # Goat is stuck
                        available_spots[x, y] = 0  # Mark goat as unavailable

        return torch.tensor(available_spots.flatten(), dtype=torch.float32)  # Return mask as flat tensor


    def _filter_goat_moves(self, state, goat_or_spot_action, goat_move_selection):
        """
        Filters the goat move probabilities to only include valid destination spots for the selected goat.
        """
        x, y = action_converter(goat_or_spot_action, state.shape[0])  # Get coordinates of selected goat
        move_legality_matrix = create_adjacent_mask_goat(state, x, y)  # Create legality matrix for its moves
        legality_tensor = torch.tensor(move_legality_matrix.flatten(), dtype=torch.float32)  # Flatten and convert
        #print('legality of move tensor', legality_tensor.view(state.shape[0], state.shape[0]))

        goat_move_selection =goat_move_selection * legality_tensor  # Apply legality mask
        #print('filtered goat move selection', goat_move_selection.view(state.shape[0], state.shape[0]))
        return goat_move_selection
    
    def predict_action(self, state, all_goats_placed_flag):
        # Prepare input for the neural network
        model_input = self._prepare_model_input(state, all_goats_placed_flag)
        #print('state',state)
        #print('all_goats_placed_flag',all_goats_placed_flag)
        # Get action probabilities from the model
        #print('model input',model_input)
        action_probs = self.actor.predict_probabilities(model_input)
        #print('action probs',action_probs)

        # Separate probabilities into goat/spot selection and movement
        goat_or_spot_selection, goat_move_selection = self._split_action_probs(action_probs, state.shape[0])

        # Get mask of valid goat/spot selections depending on game phase
        available_spots = self._get_available_spots(state, all_goats_placed_flag)

        # Filter selection probabilities using the mask
        goat_or_spot_selection = goat_or_spot_selection * available_spots
        #print('filtered goat or spot selection', goat_or_spot_selection.view(state.shape[0], state.shape[0]))

        # Normalize the distribution
        if goat_or_spot_selection.sum() > 0:
            goat_or_spot_selection /= goat_or_spot_selection.sum()
        else:
            return ValueError("Goat or spot selection probabilities sum to 0")

        # Sample a spot or goat to move
        goat_or_spot_dist = Categorical(goat_or_spot_selection)
        goat_or_spot_action = goat_or_spot_dist.sample()
        #if not all_goats_placed_flag:
            #print('spot selected', goat_or_spot_action)

        # If in movement phase, process movement distribution
        if all_goats_placed_flag:
            #print('goat chosen to move', goat_or_spot_action)
            goat_move_selection = self._filter_goat_moves(state, goat_or_spot_action, goat_move_selection)
            #print('filtered goat move selection', goat_move_selection.view(state.shape[0], state.shape[0]))
            # Normalize move distribution
            if goat_move_selection.sum() > 0:
                goat_move_selection /= goat_move_selection.sum()
            else:
                return ValueError("Goat move selection probabilities sum to 0")

            # Sample move action
            goat_move_dist = Categorical(goat_move_selection)
            goat_move_action = goat_move_dist.sample()

            return goat_or_spot_action, goat_move_action, goat_or_spot_dist, goat_move_dist

        return goat_or_spot_action, None, goat_or_spot_dist, None

    def predict_value(self, state, all_goats_placed_flag):
        model_input = self._prepare_model_input(state, all_goats_placed_flag)
        return self.critic.predict_value(model_input)
    

    def actor_load_model(self, model_path):
        self.actor.load_state_dict(torch.load(model_path))

    def critic_load_model(self, model_path):
        self.critic.load_state_dict(torch.load(model_path))
    
    




        