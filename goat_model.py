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
        self.ln = nn.LayerNorm(4 * self.input_size)  

        self.output = nn.Linear(4 * self.input_size, self.output_size)

        self.activation = F.relu

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.ln(x)

        x = self.output(x)
        x = x - x.max(dim=-1, keepdim=True).values  # Stabilize logits
        return x


    def predict_probabilities(self, model_input):
        model_input = np.reshape(model_input, (1, model_input.shape[1]))     
        model_input = torch.tensor(model_input, dtype=torch.float32)
        probabilities = self.forward(model_input)
        return probabilities.squeeze()
    

class CriticModel(nn.Module):
    def __init__(self, size):
        super(CriticModel, self).__init__()
        self.size = size
        self.input_size = self.size**2 + 1  # Same input format as actor

        hidden_size_1 = 2 * self.input_size
        hidden_size_2 = 4 * self.input_size

        self.fc1 = nn.Linear(self.input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.output = nn.Linear(hidden_size_2, 1)  # Single scalar output

        self.activation = F.relu

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.output(x)
        return x  # No activation on output (regression)

    def predict_value(self, model_input):
        """
        Predicts the state value for a given environment state.
        
        Args:
            model_input (np.ndarray): The state representation as a NumPy array.

        Returns:
            torch.Tensor: The predicted value as a tensor of shape [1].
        """
        model_input = np.reshape(model_input, (1, model_input.shape[1]))     
        model_input = torch.tensor(model_input, dtype=torch.float32)
        value = self.forward(model_input)
        return value.squeeze()  # Remove batch dimension