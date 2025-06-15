import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

learning_rate = 1e-2
epochs = 10
batch_size = 50

transform = transforms.Compose([transforms.ToTensor()])
handwriting_train_dataset = datasets.MNIST(root="data", download=True, train=True, transform=transform)
handwriting_train_loader = DataLoader(handwriting_train_dataset, batch_size=batch_size, shuffle=True)

handwriting_test_dataset = datasets.MNIST(root="data", download=True, train=False, transform=transform)
handwriting_test_loader = DataLoader(handwriting_test_dataset, batch_size=batch_size)

fashion_train_dataset = datasets.FashionMNIST(root="data", download=True, train=True, transform=   transform)
fashion_train_loader = DataLoader(fashion_train_dataset, batch_size=batch_size, shuffle=True)

fashion_test_dataset = datasets.FashionMNIST(root="data", download=True, transform=transform)
fashion_test_loader = DataLoader(fashion_test_dataset, batch_size=batch_size)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#Create a class NeuralNetwork that will inherit from nn.Module class
class MNISTNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 1 x 28 x 28
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: 32 x 28 x 28
        self.pool = nn.MaxPool2d(2, 2)                           # After pool: 32 x 14 x 14
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Output: 64 x 14 x 14
        # After pool: 64 x 7 x 7
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for digits 0-9

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + MaxPool
        
        x = x.view(-1, 64 * 7 * 7)  # flatten
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MNISTNeuralNetwork().to(device)

crossEntropyLossFunction = nn.CrossEntropyLoss()

sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)

def trainHandwriting(dataloader, model, crossEntropyLossFunction, sgd):
    #Tells the network is being trained. Best practice to add as some layers behave differently when being trained
    #Need this for batch norm and dropout layers
    model.train()
    size = len(dataloader.dataset)
    print(f"Size of dataset: {size}")

    for batch, (input, expectedOutput) in enumerate(dataloader):
        input = input.to(device)
        expectedOutput = expectedOutput.to(device)
        
        predictedOutput = model(input)
        loss = crossEntropyLossFunction(predictedOutput, expectedOutput)
        loss.backward()
        sgd.step()
        sgd.zero_grad()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * batch_size + len(input)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def testHandwriting(dataloader, model, crossEntropyLossFunction):
    #Same as model.train() but indicates we are evaluating not training
    model.eval()
    size = len(dataloader.dataset)

    test_loss, correct = 0, 0
    with torch.no_grad():
        for input, expectedOutput in dataloader:
            input = input.to(device)
            expectedOutput = expectedOutput.to(device)

            prediction = model(input)
            test_loss += crossEntropyLossFunction(prediction, expectedOutput).item()
            correct += (prediction.argmax(1) == expectedOutput).type(torch.float).sum().item()
    
    test_loss /= len(handwriting_test_loader)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for epoch in range(epochs):
    trainHandwriting(fashion_train_loader, model, crossEntropyLossFunction, sgd)
    testHandwriting(fashion_test_loader, model, crossEntropyLossFunction)

torch.save(model.state_dict(), "./model_weights.pth")

model.load_state_dict(torch.load("./model_weights.pth", weights_only=True))

testHandwriting(model, crossEntropyLossFunction)