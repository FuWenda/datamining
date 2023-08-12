import torch.nn as nn
import torch

dropout_rate=0.2
class Net1(nn.Module):
    def __init__(self):
        super(Net1,self).__init__()

        self.work = nn.Sequential(
            nn.Linear(278, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16))
    def forward(self, x):
        out = self.work(x)
        return out


class Net2(nn.Module):
    def __init__(self, input_size=278, output_size=16):
        super(Net2, self).__init__()
        self.tree = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        x = self.tree(x)
        return x

#

class Net3(nn.Module):
    def __init__(self, input_size=278, output_size=16):
        super(Net3, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Calculate the size of the fully connected layer input
        conv_output_size = self._get_conv_output_size(input_size)#4224


        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a dimension for the channel
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x

    def _get_conv_output_size(self, input_size):
        input_tensor = torch.zeros(1, 1, input_size)
        output = self.conv_layers(input_tensor)
        conv_output_size = output.view(1, -1).size(1)
        return conv_output_size

# if __name__ == '__main__':
#     model=Net3()
#     t=torch.zeros(1,278)
#     out=model(t)
#     print(out.shape)