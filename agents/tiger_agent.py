import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from helpers import create_adjacent_mask_tiger,action_converter,print_board_pretty

class TigerAgent():

    def __init__(self, model,tiger_env,board_dimension,reward_scheme,max_number_of_turns = 40):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5,weight_decay=1e-6)
        self.gamma = 0.99
        self.board_dimension = board_dimension
        self.tiger_env = tiger_env
        self.max_number_of_turns = max_number_of_turns
        self.reward_scheme = reward_scheme
        self.monitoring_state = None
        self.monitoring_distributions = []
    
    def update_policy(self,log_probs,rewards):
        self.optimizer.zero_grad()
        #print('rewards',rewards)
        normalized_rewards = (rewards-torch.mean(rewards))/torch.std(rewards)
        #print('normalized rewards',normalized_rewards)
        loss = - torch.sum(normalized_rewards * log_probs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        

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


    def monitor_KL_divergence(self): 
        _,action_dist = self.predict_action(self.monitoring_state)
        self.monitoring_distributions.append(action_dist)
        if len(self.monitoring_distributions) > 1:
            kl_div = torch.distributions.kl.kl_divergence(self.monitoring_distributions[-1],self.monitoring_distributions[-2])
            return kl_div.detach().numpy()
        
    def learn(self):
        rewards = []
        log_probs = []
        for _ in range(self.max_number_of_turns):
            current_state = self.tiger_env.return_state()
            action,action_dist = self.predict_action(current_state)
            #print('action',action)
            log_prob = action_dist.log_prob(action) 
            #print('log prob',log_prob)
            reward = self.tiger_env.step(action.item())
            #print('reward',reward)
            rewards.append(reward)
            log_probs.append(log_prob)
            if reward == self.reward_scheme["losing"] or reward == self.reward_scheme["winning"]:
                if reward == self.reward_scheme["losing"]:
                    #print(current_state)
                    break
        #print('raw rewards',rewards)
        cum_rewards = self.calculate_cum_reward(rewards)
        log_probs = torch.stack(log_probs) 
        self.update_policy(log_probs,cum_rewards)
        self.tiger_env.reset()
        return np.mean(rewards)
            
    def check_output(self):
        current_state = self.tiger_env.return_state()
        action_probs = self.model.predict_action(current_state)
        print('raw action probs',action_probs)

    def predict_action(self, state):
        flattened_state = np.reshape(state, (1, state.shape[0] ** 2))
        flattened_state = torch.tensor(flattened_state, dtype=torch.float32)
        #print('flattened state',flattened_state)
        action_probs = self.model.predict_probabilities(flattened_state)
        #print('action probs',action_probs)
        legality_matrix = create_adjacent_mask_tiger(state)
        #print('legality matrix',legality_matrix)
        action_probs  = action_probs * torch.tensor(legality_matrix.flatten(),dtype=torch.float32)
        action_probs = action_probs / action_probs.sum()  # Normalize probabilities
        if action_probs.sum() == 0:
            return ValueError("Action probabilities sum to 0")
        #print('filtered action probs',action_probs)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action,action_dist
     
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
  
    
    




        







'-------------------------------------------------------------------------------------------------------------------------------------------------------'

class ActorCriticTigerAgent():
    def __init__(self, actor_model, critic_model, tiger_env, board_dimension, reward_scheme, max_number_of_turns=40):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = actor_model.to(self.device)
        self.critic = critic_model.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=5e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4,weight_decay=1e-6)
        self.gamma = 0.99

        self.board_dimension = board_dimension
        self.tiger_env = tiger_env
        self.max_number_of_turns = max_number_of_turns
        self.reward_scheme = reward_scheme

        # For monitoring (e.g. storing logits, entropy, etc.)
        self.monitoring_state = None
        self.monitoring_distributions = []

    def update_actor(self, log_prob, advantage):
        """
        Update actor using policy gradient with advantage.
        """
        self.actor_optimizer.zero_grad()
        loss = -log_prob * advantage
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

    def update_critic(self, predicted_value, target_value):
        """
        Update critic using MSE loss between predicted and TD target.
        """
        if not isinstance(target_value, torch.Tensor):
            target_value = torch.tensor([target_value], dtype=torch.float32)

        loss = torch.nn.functional.mse_loss(predicted_value, target_value)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

    def learn(self):
        """
        Runs one training episode for the tiger agent using actor-critic with TD advantage.

        Returns:
            float: The total reward obtained in this episode.
        """
        total_reward = 0
        step_count = 0

        advantages = []
        values = []
        states= []

        for _ in range(self.max_number_of_turns):
            current_state = self.tiger_env.return_state()
            states.append(current_state)

            # Get actions and action distributions from actor
            tiger_select_action, move_action, tiger_dist, move_dist = self.predict_action(current_state)

            # Estimate value of current state from critic
            value = self.predict_value(current_state)
    

            values.append(value)

            # Combine log probs for both actions
            log_prob = tiger_dist.log_prob(tiger_select_action) + move_dist.log_prob(move_action)
            log_prob = log_prob.to(self.device)
            action = (tiger_select_action.item(), move_action.item())


            # Execute action in environment and collect reward
            reward = self.tiger_env.step(action)
            reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
            total_reward += reward.cpu().item()
            step_count += 1
            #print_board_pretty(current_state)
            #print('reward',reward)
            # Estimate next state value
            next_state = self.tiger_env.return_state()
            next_value = self.predict_value(next_state)

            # If terminal state, next value is 0
            done = reward.cpu().item() == self.reward_scheme["losing"] or reward.cpu().item() == self.reward_scheme["winning"]

            if done:
                next_value = torch.tensor([0.0], dtype=torch.float32).to(self.device)

            # TD Error = Advantage
            advantage = reward + self.gamma * next_value.detach() - value.detach()

            advantages.append(advantage)
            # Update actor and critic
            self.update_actor(log_prob, advantage)
            target_value = reward + self.gamma * next_value.detach()
            self.update_critic(value, target_value)

            if done:
                break

        self.tiger_env.reset()
        return total_reward if step_count > 0 else 0.0,advantages,values,states
    

    def _prepare_model_input(self, state):
        """
        Flattens the board state to form the input for the actor/critic models.
        """
        flattened_state = np.reshape(state, (1, state.shape[0] ** 2))
        return torch.tensor(flattened_state, dtype=torch.float32).to(self.device)

    def _get_movable_tigers_mask(self, state):
        """
        Returns a flat mask (tensor of shape [25]) indicating which tigers are valid to select.
        A tiger is valid if it has at least one legal move (standard or capture).
        """
        size = state.shape[0]
        movable_mask = np.zeros((size, size), dtype=np.float32)

        for x in range(size):
            for y in range(size):
                if state[x, y] == 1:  # Tiger present
                    legal_moves = create_adjacent_mask_tiger(state, x, y)
                    if np.sum(legal_moves) > 0:
                        movable_mask[x, y] = 1.0  # Tiger can move

        return torch.tensor(movable_mask.flatten(), dtype=torch.float32)


    def _filter_tiger_moves(self, state, selected_tiger_action, tiger_move_selection):
        """
        Filters the tiger move probabilities to only include valid destination spots
        for the selected tiger.
        """
        x, y = action_converter(selected_tiger_action, state.shape[0])  # Coordinates of selected tiger
        move_legality_matrix = create_adjacent_mask_tiger(state, x, y)  # Get legal moves
        legality_tensor = torch.tensor(move_legality_matrix.flatten(), dtype=torch.float32)

        # Apply legality mask
        filtered_move_selection = tiger_move_selection * legality_tensor
        return filtered_move_selection


    def _split_action_probs(self, action_probs, size):
        """
        Splits the model output into two distributions:
        - One for selecting a tiger
        - One for selecting a destination to move it to
        """
        tiger_selection = F.softmax(action_probs[0:size ** 2], dim=0)     # First 25: pick a tiger
        tiger_move_selection = F.softmax(action_probs[size ** 2:], dim=0) # Last 25: pick where to move it
        return tiger_selection, tiger_move_selection

    def predict_action(self, state):
        """
        Predicts a tiger move using the actor model.
        Always returns two actions: which tiger to move, and where to move it.
        """
        # Prepare input for the neural network (no flag needed)
        model_input = self._prepare_model_input(state)

        # Get action probabilities from the actor
        action_probs = self.actor.predict_probabilities(model_input)

        # Split output into two distributions
        tiger_select_probs, tiger_move_probs = self._split_action_probs(action_probs.cpu(), state.shape[0])
        # Mask for valid tigers to move (i.e., tigers that can move somewhere)
        available_tigers = self._get_movable_tigers_mask(state)
        tiger_select_probs = tiger_select_probs * available_tigers

        # Normalize
        if tiger_select_probs.sum() > 0:
            tiger_select_probs /= tiger_select_probs.sum()
        else:
            raise ValueError("Tiger selection probabilities sum to 0")
        # Sample a tiger
        tiger_dist = Categorical(tiger_select_probs)
        tiger_action = tiger_dist.sample()

        # Get legal moves for the selected tiger
        tiger_move_probs = self._filter_tiger_moves(state, tiger_action, tiger_move_probs)
        # Normalize move probabilities
        if tiger_move_probs.sum() > 0:
            tiger_move_probs /= tiger_move_probs.sum()
        else:
            raise ValueError("Tiger move selection probabilities sum to 0")
        # Sample move
        move_dist = Categorical(tiger_move_probs)
        move_action = move_dist.sample()

        return tiger_action, move_action, tiger_dist, move_dist

    def predict_value(self, state):
        model_input = self._prepare_model_input(state)
        return self.critic.predict_value(model_input)
    
    def actor_load_model(self, model_path):
        self.actor.load_state_dict(torch.load(model_path))

    def critic_load_model(self, model_path):
        self.critic.load_state_dict(torch.load(model_path))
