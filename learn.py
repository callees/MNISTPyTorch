import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="data", download=True, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#Create a class NeuralNetwork that will inherit from nn.Module class
class MNISTNeuralNetwork(nn.Module):
    #Define constructor. #self == this
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 10*10),
            nn.ReLU(),
            nn.Linear(10*10, 5*5),
            nn.ReLU(),
            nn.Linear(5*5, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = MNISTNeuralNetwork().to(device)

learning_rate = 1e-3
epochs = 100
batch_size = 50

crossEntropyLossFunction = nn.CrossEntropyLoss()

sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)

def trainHandwriting(dataloader, model, crossEntropyLossFunction, sgd):
    #Tells the network is being trained. Best practice to add as some layers behave differently when being trained
    #Need this for batch norm and dropout layers
    model.train()
    size = len(dataloader.dataset)

    for batch, (input, expectedOutput) in enumerate(dataloader):
        input = input.to(device)
        expectedOutput = expectedOutput.to(device)
        
        predictedOutput = model(input)
        loss = crossEntropyLossFunction(predictedOutput, expectedOutput)
        loss.backward()
        sgd.step()
        sgd.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(input)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


trainHandwriting(train_loader, model, crossEntropyLossFunction, sgd)