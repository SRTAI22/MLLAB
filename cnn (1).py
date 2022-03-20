import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import requests
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
batch_size = 300
input_size = 784
hidden_size = 200       
num_classes = 10
num_epoch = 120


train_dataset = torchvision.datasets.KMNIST(root='./dataKM', train=True, download=True, transform=transforms.ToTensor())

test_dataset = torchvision.datasets.KMNIST(root='./dataKM', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.fc = nn.Linear(26*26*32, 128)
        self.fc1 = nn.Linear(128, 10)
       
        

    def forward(self, x):
       x = self.conv1(x)
       x = F.relu(x)
       x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
       x = self.fc(x)
       x = F.relu(x)
       lg = self.fc1(x)

       return x
       print(x.shape)

model = ConvNet()

l = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_t_s = len(train_loader)

for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader, 0):
        images = images.to(device)
        labels = labels.to(device)

        ot = model(images)
        loss = l(ot, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if(i+1) % 1 == 0:
        print(f' epoch : {epoch+1}/{num_epoch}, step : {i+1}/{n_t_s}, loss : {loss.item():.4f}')

print('Finished Training')
