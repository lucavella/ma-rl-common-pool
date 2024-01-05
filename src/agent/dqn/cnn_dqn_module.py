import torch.nn as nn
import torch.nn.functional as F



class CnnDQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32, kernel_size=5, padding=0, stride=1):
        super().__init__()
        conv1_out = (input_size - kernel_size + 2 * padding) / stride + 1
        conv2_out = (conv1_out - kernel_size + 2 * padding) / stride + 1

        self.conv1 = nn.Conv2d(3, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding)

        # flatten
        self.layer1 = nn.Linear(conv2_out, hidden_size)    # Input Layer
        self.layer2 = nn.Linear(hidden_size, hidden_size)   # Hidden layer 1
        self.layer3 = nn.Linear(hidden_size, hidden_size)   # Hidden Layer 2
        self.layer4 = nn.Linear(hidden_size, output_size)   # Output Layer

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)