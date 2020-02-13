import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim

from tqdm import tqdm

import map_data

class DatasetTilemap(Dataset):

    def __init__(self, context, number_of_data_points):

        self.training_map = context.tilemap

        self.X, self.y = self.generate_data_points(number_of_data_points)


    def generate_data_points(self, count):
        x = []  # maps
        y = []  # costs

        print("Generating data points..")
        for i in tqdm(range(count)):
            self.training_map.randomize_start_goal() # randomize new positions

            # take new positions and put them in empty map of same size
            (x1, y1) = self.training_map.custom_start
            (x2, y2) = self.training_map.custom_goal

            tmp_map = []
            for j in range(self.training_map.map_height):
                tmp_map.append([0] * self.training_map.map_width)
            tmp_map[x1][y1] = 2
            tmp_map[x2][y2] = 2
            
            x.append(tmp_map)

            y.append(int(self.training_map.astar_cost())) # cost of new path

        return (x, y)

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, index):
        return torch.Tensor(self.X[index]), self.y[index]

class Net(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

    @property
    def input(self):
        return self.fc1.in_features

    @property
    def output(self):
        return self.fc4.out_features
        
class InstructionSet:

    def __init__(self, generations=2000, data_size=20000, set_size=100, test_size=100):
        self.generations = generations
        self.data_size = data_size
        self.set_size = set_size
        self.test_size = test_size


class NeuralNetwork:

    def __init__(self, context, instruction_set=None):
        self.context = context
        self.instruction_set = instruction_set if instruction_set else InstructionSet()
        self.net = Net(context.tilemap.map_width * context.tilemap.map_height, context.tilemap.map_width * context.tilemap.map_height * 2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.eval()

    def train(self):

        train_data = DatasetTilemap(self.context, self.instruction_set.data_size)
        test_data = DatasetTilemap(self.context, self.instruction_set.data_size)

        train_set = torch.utils.data.DataLoader(train_data, self.
        instruction_set.set_size, shuffle=True)
        test_set = torch.utils.data.DataLoader(test_data, self.instruction_set.test_size, shuffle=False)

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        self.net = self.net.to(self.device)

        correct = 0
        total = 0

        print("Training network..")
        for gen in tqdm(range(self.instruction_set.generations)):
            #print("Generation #", gen)
            for data in train_set:
                X, y = data  # X is the batch of features, y is the batch of targets.
                self.net.zero_grad()  # sets gradients to 0 before loss calc.
                X = X.to(self.device)
                output = self.net(X.view(-1, self.net.input))
                output = output.cpu()
                loss = loss_function(output, y)  # calc and grab the loss value
                loss.backward()  # apply this loss backwards thru the network's parameters
                optimizer.step()  # attempt to optimize weights to account for loss/gradients
            self.net = self.net.to(self.device)
            with torch.no_grad():
                for data in test_set:
                    X, y = data
                    X = X.to(self.device)
                    output = self.net(X.view(-1, self.net.input))
                    output = output.cpu()
                    for idx, i in enumerate(output):
                        #print(torch.argmax(i), y[idx])
                        if torch.argmax(i) == y[idx]:
                            correct += 1
                        total += 1

        print("Accuracy: ", round((correct/total)*100, 3))