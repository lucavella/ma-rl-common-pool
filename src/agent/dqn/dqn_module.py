import torch.nn as nn



class DQN(nn.Module):
    def __init__(self, input_shape, output_size, hidden_size=32):
        super().__init__()
        depth, h, w = input_shape
        flatten_out = depth * h * w

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_out, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, output_size)
        )


    def forward(self, x):
        x = self.fc(x)
        return x