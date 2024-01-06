import torch.nn as nn



class CnnDQN(nn.Module):
    def __init__(self, input_shape, output_size, hidden_size=32):
        super().__init__()
        depth, h, w = input_shape
        conv1_filters = 16
        conv1_kernel_size = 5
        conv1_stride = 2
        conv2_filters = 32
        conv2_kernel_size = 3
        conv2_stride = 1

        conv_out_h = (((h - conv1_kernel_size) // conv1_stride + 1) - conv2_kernel_size) // conv2_stride + 1
        conv_out_w = (((w - conv1_kernel_size) // conv1_stride + 1) - conv2_kernel_size) // conv2_stride + 1
        flatten_out = conv2_filters * conv_out_h * conv_out_w

        self.conv = nn.Sequential(
            nn.Conv2d(depth, conv1_filters, kernel_size=conv1_kernel_size, stride=conv1_stride),
            nn.ReLU(),
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=conv2_kernel_size, stride=conv2_stride),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_out, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )


    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x