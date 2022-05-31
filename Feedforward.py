import torch.nn as nn
import StepModule


class FFNetwork(StepModule.StepModule):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # input [in_channels, out_channels, width, heigth]
            nn.Conv2d(3, 4, kernel_size=9, padding=4, stride=2),
            # conv2d output als specificatie [channels=4, width=64, heigth=64], geen kleinere dimensie door 4 padding,
            # wel door stride = 2
            # layernorm verwacht als input vorm [channels, height, width]
            nn.LayerNorm([4, 64, 64]),
            nn.ReLU(),
            # noch layernorm, noch relu passen de vorm aan
            nn.MaxPool2d(kernel_size=2, stride=2),
            # maxpool met stride twee halveert de dimensies naar 32x32

            nn.Conv2d(4, 8, kernel_size=9, padding=4, stride=2),
            # conv2d output als specificatie [channels=8, width=16, heigth=16], geen kleinere dimensie door 4 padding,
            # wel door stride = 2
            nn.LayerNorm([8, 16, 16]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # maxpool met stride 2 halveert de dimensies naar 8x8

            # de eerste fully connected layer krijgt dus 8 channels x 8 breed x 8 hoog binnen
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 16),
            nn.ReLU(),

            # de tweede fully connected layer vergroot van 16 naar 32
            nn.Linear(16, 32),
            nn.ReLU())

        self.znetwork = nn.Linear(32, 2)

        self.lnetwork = nn.Linear(32, 3)

    def forward(self, ot):
        enc = self.network(ot)
        return self.znetwork(enc), self.lnetwork(enc)

