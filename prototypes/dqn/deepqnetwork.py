import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from util import plot_learning_curve
import os

class DeepQNetwork(nn.Module):
    def __init__(self, lr, name, n_actions, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        #self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        #fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        #conv3 = F.relu(self.conv3(conv2))

        conv_state = conv3.view(conv2.size()[0], -1)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions

    def save_checkpoint(self):
        print('.....saving checkpoint.....')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('.....loading checkpoint....')
        self.load_state_dict(T.load(self.checkpoint_file))