import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self, size):
        super(Model, self).__init__()
        self.size = size
        
        # Sequentially renamed layers
        self.fc1 = nn.Linear(self.size ** 2, 2 * self.size ** 2)
        self.fc2 = nn.Linear(2 * self.size ** 2, 4 * self.size ** 2)
        self.fc3 = nn.Linear(4 * self.size ** 2, 6 * self.size ** 2)
        self.fc4 = nn.Linear(6 * self.size ** 2, 8 * self.size ** 2)
        self.fc5 = nn.Linear(8 * self.size ** 2, 6 * self.size ** 2)  # Adjusted to match next layer
        self.fc6 = nn.Linear(6 * self.size ** 2, 4 * self.size ** 2)
        self.fc7 = nn.Linear(4 * self.size ** 2, 2 * self.size ** 2)
        self.fc8 = nn.Linear(2 * self.size ** 2, self.size ** 2)
        self.output = nn.Linear(self.size ** 2, self.size ** 2)

        self.dropout = nn.Dropout(p=0.2)  # Dropout with a probability of 0.2
        self.activation = F.relu

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.activation(self.fc6(x))
        x = self.activation(self.fc7(x))
        x = self.activation(self.fc8(x))
        x = self.output(x)
        x = x - x.max(dim=-1, keepdim=True).values  # Stabilize logits
        x = F.softmax(x, dim=-1)
        return x


    def predict_probabilities(self, state):
        state = np.reshape(state, (1, self.size ** 2))     
        state = torch.tensor(state, dtype=torch.float32)
        probabilities = self.forward(state)
        return probabilities
    



class ActorModel(nn.Module):
    def __init__(self, size):
        super(ActorModel, self).__init__()
        self.size = size
        self.input_size = self.size**2
        self.output_size = 2 * self.size**2

        # Expanded layers
        self.fc1 = nn.Linear(self.input_size, 2 * self.input_size)
        self.fc2 = nn.Linear(2 * self.input_size, 4 * self.input_size)
        self.fc3 = nn.Linear(4 * self.input_size, 6 * self.input_size)

        self.output = nn.Linear(6 * self.input_size, self.output_size)

        self.layer_norm1 = nn.LayerNorm(2 * self.input_size)
        self.layer_norm2 = nn.LayerNorm(4 * self.input_size)
        self.activation = F.relu

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.layer_norm1(x)
        x = self.activation(self.fc2(x))
        x = self.layer_norm2(x)
        x = self.activation(self.fc3(x))

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
        self.input_size = self.size**2

        # Expanded layers
        hidden_size_1 = 2 * self.input_size
        hidden_size_2 = 4 * self.input_size
        hidden_size_3 = 6 * self.input_size

        self.fc1 = nn.Linear(self.input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.fc4 = nn.Linear(hidden_size_3, hidden_size_2)

        self.output = nn.Linear(hidden_size_2, 1)  # Single scalar output

        self.layer_norm1 = nn.LayerNorm(hidden_size_1)
        self.layer_norm2 = nn.LayerNorm(hidden_size_2)
        self.layer_norm3 = nn.LayerNorm(hidden_size_3)

        self.activation = F.relu

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.layer_norm1(x)
        x = self.activation(self.fc2(x))
        x = self.layer_norm2(x)
        x = self.activation(self.fc3(x))
        x = self.layer_norm3(x)
        x = self.activation(self.fc4(x))
        x = self.output(x)
        return x

    def predict_value(self, model_input):
        model_input = np.reshape(model_input, (1, model_input.shape[1]))     
        model_input = torch.tensor(model_input, dtype=torch.float32)
        value = self.forward(model_input)
        return value.squeeze()  # Remove batch dimension