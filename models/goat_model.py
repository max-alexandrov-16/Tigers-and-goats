import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init


class ReinforceModel(nn.Module):
    def __init__(self, size):
        super(ReinforceModel, self).__init__()
        self.size = size
        self.input_size = self.size**2 + 1
        self.output_size = 2 * self.size**2

        self.fc1 = nn.Linear(self.input_size, 2 * self.input_size)
        self.fc2 = nn.Linear(2 * self.input_size, 4 * self.input_size)
        self.ln = nn.LayerNorm(4 * self.input_size)  

        self.fc3 = nn.Linear(4 * self.input_size, 6 * self.input_size)
        self.fc4 = nn.Linear(6 * self.input_size, 4 * self.input_size)

        self.output = nn.Linear(4 * self.input_size, self.output_size)

        self.activation = F.relu

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.ln(x)

        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.ln(x)

        x = self.output(x)
        x = x - x.max(dim=-1, keepdim=True).values  # Stabilize logits
        return x


    def predict_probabilities(self, model_input):
        model_input = np.reshape(model_input, (1, model_input.shape[1]))     
        model_input = torch.tensor(model_input, dtype=torch.float32)
        probabilities = self.forward(model_input)
        return probabilities
    
def init_weights_kaiming(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)



class ActorModel(nn.Module):
    def __init__(self, size):
        super(ActorModel, self).__init__()
        self.size = size
        self.input_size = self.size**2 + 1
        self.output_size = 2 * self.size**2

        self.fc1 = nn.Linear(self.input_size, 2 * self.input_size)
        self.fc2 = nn.Linear(2 * self.input_size, 4 * self.input_size)
        self.fc3 = nn.Linear(4 * self.input_size, 6 * self.input_size)
        self.fc4 = nn.Linear(6 * self.input_size, 2 * self.input_size)

        self.layer_norm1 = nn.LayerNorm(2 * self.input_size)
        self.layer_norm2 = nn.LayerNorm(6 * self.input_size) 

        self.output = nn.Linear(2 * self.input_size, self.output_size)

        self.activation = F.relu

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.layer_norm1(x)

        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))

        x = self.layer_norm2(x)
        x = self.activation(self.fc4(x))

        x = self.output(x)
        x = x - x.max(dim=-1, keepdim=True).values  # Stabilize logits
        return x


    def predict_probabilities(self, model_input):
        probabilities = self.forward(model_input)
        return probabilities.squeeze()
    

class CNNActorModel(nn.Module):
    def __init__(self, size):
        super(CNNActorModel, self).__init__()
        self.size = size
        self.board_size = size ** 2
        self.input_size = self.board_size + 1  # +1 for all_goats_placed_flag
        self.output_size = 2 * self.board_size  # same as original actor

        # Convolutional layers (for board only)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Output: (16, 5, 5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Output: (32, 5, 5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: (64, 5, 5)

        # Fully connected layers
        conv_output_flat_size = 64 * size * size  # 64 feature maps of 5x5
        fc_input_size = conv_output_flat_size + 1  # add goat flag here

        self.fc1 = nn.Linear(fc_input_size, 2 * self.input_size)
        self.fc2 = nn.Linear(2 * self.input_size, 4 * self.input_size)
        self.fc3 = nn.Linear(4 * self.input_size, 2 * self.input_size)
        self.output = nn.Linear(2 * self.input_size, self.output_size)

        self.layer_norm1 = nn.LayerNorm(2 * self.input_size)
        self.layer_norm2 = nn.LayerNorm(4 * self.input_size)

        self.activation = F.relu

    def forward(self, x):
        # x: (batch_size, 26) = (25 board + 1 flag)
        board, flag = x[:, :-1], x[:, -1:]

        # Reshape board to (batch_size, 1, 5, 5)
        board = board.view(-1, 1, self.size, self.size)

        # Convolutional layers
        x = self.activation(self.conv1(board))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))

        # Flatten and concatenate the goat flag
        x = x.view(x.size(0), -1)  # (batch_size, flattened conv features)
        x = torch.cat([x, flag], dim=1)  # append the flag

        # Fully connected layers
        x = self.activation(self.fc1(x))
        x = self.layer_norm1(x)

        x = self.activation(self.fc2(x))
        x = self.layer_norm2(x)

        x = self.activation(self.fc3(x))
        x = self.output(x)

        # Logit stabilization
        x = x - x.max(dim=-1, keepdim=True).values
        return x

    def predict_probabilities(self, model_input):
        probabilities = self.forward(model_input)
        return probabilities.squeeze()
    

class CriticModel(nn.Module):
    def __init__(self, size):
        super(CriticModel, self).__init__()
        self.size = size
        self.input_size = self.size**2 + 1

        self.fc1 = nn.Linear(self.input_size, 2 * self.input_size)
        self.fc2 = nn.Linear(2 * self.input_size, 4 * self.input_size)
        self.fc3 = nn.Linear(4 * self.input_size, 6 * self.input_size)
        self.fc4 = nn.Linear(6 * self.input_size, 8 * self.input_size)
        self.fc5 = nn.Linear(8 * self.input_size, 4 * self.input_size)

        self.output = nn.Linear(4 * self.input_size, 1)

        self.layer_norm1 = nn.LayerNorm(2 * self.input_size)
        self.layer_norm2 = nn.LayerNorm(4 * self.input_size)
        self.layer_norm3 = nn.LayerNorm(8 * self.input_size)

        self.activation = F.relu

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.layer_norm1(x)

        x = self.activation(self.fc2(x))
        x = self.layer_norm2(x)

        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.layer_norm3(x)

        x = self.activation(self.fc5(x))
        x = self.output(x)
        return x

    def predict_value(self, model_input):
        """
        Predicts the state value for a given environment state.
        
        Args:
            model_input (np.ndarray): The state representation as a NumPy array.

        Returns:
            torch.Tensor: The predicted value as a tensor of shape [1].
        """
        value = self.forward(model_input)
        return value.squeeze()  # Remove batch dimension