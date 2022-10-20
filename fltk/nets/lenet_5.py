from torch import nn


class LeNet5(nn.Module):
    """
    LeNe5 network class that inherits the nn.Module as it provides many of the needed methods.
    """

    def __init__(self):
        """
        Initialize the LeNet5 class.
        """
        super(LeNet5, self).__init__()

        # Define convolutional layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Define convolutional layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Connect layers
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()

        # Output layer
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Define sequence of operations to process the data.

        :param x: The input data.
        :return: The processed data.
        """
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc(out)
        out = self.relu(out)

        out = self.fc1(out)
        out = self.relu1(out)

        out = self.fc2(out)

        return out
