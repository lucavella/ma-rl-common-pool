import torch.nn as nn
import torch.nn.functional as F



class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_size, hidden_size)    # Input Layer
        self.layer2 = nn.Linear(hidden_size, hidden_size)   # Hidden layer 1
        self.layer3 = nn.Linear(hidden_size, hidden_size)   # Hidden Layer 2
        self.layer4 = nn.Linear(hidden_size, output_size)   # Output Layer

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)