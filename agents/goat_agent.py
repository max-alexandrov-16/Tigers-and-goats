import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
import torch.nn.init as init
from helpers.helpers import create_adjacent_mask_goat,action_converter

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
        self.monitoring_state_1 = None
        self.monitoring_state_2 = None
        self.monitoring_state_3 = None
    

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

    def __init__(self, actor_model,critic_model,goat_env,board_dimension,reward_scheme,actor_optimizer,critic_optimizer,max_number_of_turns = 40):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.actor = actor_model.to(self.device)
        self.critic = critic_model.to(self.device)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = 0.99
        self.board_dimension = board_dimension
        self.goat_env = goat_env
        self.max_number_of_turns = max_number_of_turns
        self.reward_scheme = reward_scheme
        self.monitoring_state_and_flag_1 = None
        self.monitoring_state_and_flag_2 = None
        self.monitoring_state_and_flag_3 = None
        self.monitoring_distributions_1 = []
        self.monitoring_distributions_2 = []
        self.monitoring_distributions_3 = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.probs = []
    
    def normalize_advantage(self, advantage: torch.Tensor) -> torch.Tensor:
        """
        Normalize a scalar advantage tensor using running mean/std.
        Returns a 0-dim tensor on the same device, detached from autograd.
        """
        # Initialize stats on first call
        if not hasattr(self, "_adv_stats"):
            self._adv_stats = {
                "n": 0,
                "mean": 0.0,
                "M2": 0.0,
                "eps": 1e-8,
                "warmup": 32,
                "clip": 5.0,
            }

        stats = self._adv_stats
        device = advantage.device

        def std():
            return ((stats["M2"] / (stats["n"] - 1)) + stats["eps"]) ** 0.5 if stats["n"] >= 2 else 1.0

        # Convert to Python float for running stats update
        x = float(advantage.detach().cpu().item())

        # Normalize using current running mean/std
        if stats["n"] < stats["warmup"]:
            norm_x = x
        else:
            norm_x = (x - stats["mean"]) / std()
            if stats["clip"] is not None:
                norm_x = max(min(norm_x, stats["clip"]), -stats["clip"])

        # Update running stats
        stats["n"] += 1
        delta = x - stats["mean"]
        stats["mean"] += delta / stats["n"]
        delta2 = x - stats["mean"]
        stats["M2"] += delta * delta2

        # Return as a detached tensor on the same device
        return torch.tensor(norm_x, dtype=torch.float32, device=device, requires_grad=False)


    def update_actor(self, log_prob, advantage, entropy, entropy_coef=0.025):
        """
        Performs a per-step actor update using policy gradient, advantage, and precomputed entropy.

        Args: 
            log_prob (Tensor): Log probability of the action taken.
            advantage (float or Tensor): Advantage estimate (single-step).
            entropy (Tensor): Entropy of the policy at this step.
            entropy_coef (float): Coefficient for entropy regularization.
        """
        self.actor_optimizer.zero_grad()

        # Normalize advantage
        norm_advantage = self.normalize_advantage(advantage)

        # Policy gradient loss
        loss = -log_prob * norm_advantage

        # Add entropy penalty
        loss = loss - entropy_coef * entropy

        # Backpropagate and clip gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
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

        # Return loss value
        return loss


    def learn(self,record_monitoring_states = False):
        """
        Runs one training episode using the current policy. Collects states, actions, and rewards,
        and updates both actor and critic per step using TD-based advantage.

        Returns:
            float: The average reward received during the episode.
        """
        total_reward = 0
        total_critic_loss = 0
        step_count = 0
        entropies = []

        for _ in range(self.max_number_of_turns):
            # Get current environment state and placement phase flag
            current_state = self.goat_env.return_state()
            self.states.append(np.copy(current_state))
            all_goats_placed_flag = self.goat_env.return_goat_placement_flag()
            if record_monitoring_states:
                self.collect_monitor_states(step_count, current_state, all_goats_placed_flag)

            # Predict the action using the policy
            goat_or_spot_action, goat_move_action, goat_or_spot_dist, goat_move_dist = self.predict_action(current_state, all_goats_placed_flag)
            #print('forward prop actor')
            
            # Estimate value of current state
            value = self.predict_value(current_state, all_goats_placed_flag)
            self.values.append(value.clone().detach().cpu().numpy())
            #print('forward prop critic')
            step_entropy = goat_or_spot_dist.entropy()
            # Combine log probabilities depending on phase
            if goat_move_dist is not None:
                step_entropy += goat_move_dist.entropy() 
                log_prob = goat_or_spot_dist.log_prob(goat_or_spot_action) + goat_move_dist.log_prob(goat_move_action)
                action = (goat_or_spot_action.item(), goat_move_action.item())
            else:
                log_prob = goat_or_spot_dist.log_prob(goat_or_spot_action)
                action = goat_or_spot_action.item()
            entropies.append(step_entropy.mean().detach().cpu().numpy())
            self.actions.append(action)
            log_prob = log_prob.to(self.device)
            self.log_probs.append(log_prob.clone().detach().cpu().numpy())
            # Take action and receive reward
            reward = self.goat_env.step(action)
            self.rewards.append(reward)
            reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
            total_reward += reward.cpu().item()
            step_count += 1

            # Get next state value (for TD target) 
            next_state = self.goat_env.return_state()
            next_all_goats_placed_flag = self.goat_env.return_goat_placement_flag()
            next_value = self.predict_value(next_state, next_all_goats_placed_flag)

            # If terminal, set next_value to 0 (no future value)
            done = reward == self.reward_scheme["losing"] or reward == self.reward_scheme["winning"]
            if done:
                next_value = torch.tensor([0.0], dtype=torch.float32).to(self.device)

            # Compute advantage: TD error
            advantage = reward + self.gamma * next_value.detach() - value.detach()

            # Update actor using advantage
            self.update_actor(log_prob, advantage, step_entropy)
            #print('updated actor')

            # Update critic to fit target value
            target_value = reward + self.gamma * next_value.detach()
            # ensuring the size of input matches (0) the size of target (1)
            value = torch.reshape(value, (1,))
            critic_loss = self.update_critic(value, target_value)
            total_critic_loss += critic_loss
            #print('updated critic')
            if done:
                break

        self.goat_env.reset()
        return total_reward,(total_critic_loss/step_count).item(),np.mean(entropies),step_count if step_count > 0 else (0.0,0.0,0.0,0.0)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.probs = [] 

    def get_memory(self):
        return self.states, self.actions, self.rewards, self.log_probs,self.probs, self.values

    def _prepare_model_input(self, state, all_goats_placed_flag):
        """
        Flattens the board state and appends the goat placement flag to form the input for the model.
        """
        flattened_state = np.reshape(state, (1, state.shape[0] ** 2))  # Flatten 2D state to 1D
        all_goats_placed_flag = torch.tensor(all_goats_placed_flag, dtype=torch.float32).view(1, 1)  # Make flag a 2D tensor
        flattened_state = torch.tensor(flattened_state, dtype=torch.float32)  # Convert numpy to tensor
        return torch.cat((flattened_state, all_goats_placed_flag), dim=1).to(self.device)  # Concatenate along the feature axis


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
        self.probs.append(action_probs)
        #print('action probs',action_probs)

        # Separate probabilities into goat/spot selection and movement
        goat_or_spot_selection, goat_move_selection = self._split_action_probs(action_probs.cpu(), state.shape[0])
        # Get mask of valid goat/spot selections depending on game phase
        available_spots = self._get_available_spots(state, all_goats_placed_flag)
        #print('available spots', available_spots.view(state.shape[0], state.shape[0]))
        # Filter selection probabilities using the mask
        goat_or_spot_selection = goat_or_spot_selection * available_spots
        #print('filtered goat or spot selection', goat_or_spot_selection.view(state.shape[0], state.shape[0]))

        # Normalize the distribution
        if goat_or_spot_selection.sum() > 0:
            goat_or_spot_selection /= goat_or_spot_selection.sum()
        else:
            print('state',state)
            print('action probs',action_probs)
            print('goat or spot selection', goat_or_spot_selection.view(state.shape[0], state.shape[0]))
            print('filtered goat or spot selection', goat_or_spot_selection.view(state.shape[0], state.shape[0]))
            print('available spots', available_spots.view(state.shape[0], state.shape[0]))
            raise ValueError("Goat or spot selection probabilities sum to 0")

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

    def collect_monitor_states(self, step_count, state, all_goats_placed_flag):
        if 5<step_count<10 and self.monitoring_state_and_flag_1 is None:
            self.monitoring_state_and_flag_1 = (state, all_goats_placed_flag)    

        if 10<step_count<15 and self.monitoring_state_and_flag_2 is None:
            self.monitoring_state_and_flag_2 = (state, all_goats_placed_flag)
        
        if 23<step_count<30 and self.monitoring_state_and_flag_3 is None:
            self.monitoring_state_and_flag_3 = (state, all_goats_placed_flag)
        
        else:
            pass

    def monitor_KL_divergence_multiple_states(self, epsilon=1e-8):
        """
        Tracks KL divergence across 3 fixed (state, flag) pairs with epsilon smoothing to avoid infinite KL.
        Assumes two categorical distributions per state.
        Returns dictionary of KL divergence values per monitoring state.
        """
        kl_dict = {}

        for i in range(1, 4):
            state, flag = getattr(self, f"monitoring_state_and_flag_{i}")
            _, _, goat_or_spot_dist, dist_move = self.predict_action(state, flag)

            # Store distributions as a tuple (goat_or_spot_dist, dist_move)
            dist_list = getattr(self, f"monitoring_distributions_{i}")
            dist_list.append((goat_or_spot_dist, dist_move))

            if len(dist_list) > 1:
                prev_goat_or_spot_dist, prev_dist_move = dist_list[-2]
                curr_goat_or_spot_dist, curr_dist_move = dist_list[-1]

                # Smooth probabilities by clamping to [epsilon, 1.0]
                def smooth_dist(dist):
                    if dist is None:
                        return None
                    probs = torch.clamp(dist.probs, epsilon, 1.0)
                    # Re-normalize to ensure sum to 1 after clamping
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    return torch.distributions.Categorical(probs=probs)

                prev_goat_or_spot_dist_smooth = smooth_dist(prev_goat_or_spot_dist)
                curr_goat_or_spot_dist_smooth = smooth_dist(curr_goat_or_spot_dist)

                kl_or_spot = torch.distributions.kl.kl_divergence(curr_goat_or_spot_dist_smooth, prev_goat_or_spot_dist_smooth).mean()

                if curr_dist_move is not None and prev_dist_move is not None:
                    prev_dist_move_smooth = smooth_dist(prev_dist_move)
                    curr_dist_move_smooth = smooth_dist(curr_dist_move)
                    kl_move = torch.distributions.kl.kl_divergence(curr_dist_move_smooth, prev_dist_move_smooth).mean()
                    kl_total = kl_or_spot + kl_move
                else:
                    kl_total = kl_or_spot

                kl_dict[f"monitoring_state_{i}_KL"] = kl_total.item()

        return kl_dict




                