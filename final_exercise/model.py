import torch
import torch.nn.functional as F
from torch import nn, optim


class MyShittyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        ## dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        x = F.log_softmax(self.fc3(x), dim=1)

        return x

def validation(model, testloader, criterion):
    total_accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        i += 1
        ## flatten
        img = images.view(images.shape[0], -1)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ## probabilities
        ps = torch.exp(output)

        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)

        ## accuracy
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        total_accuracy += accuracy.item()

    return test_loss, total_accuracy

def train(model, trainloader, testloader, criterion, optimizer, epochs=10):
    steps = 0
    running_loss = 0
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1
            ## zero grads
            optimizer.zero_grad()

            ## model output
            output = model(images)

            ## loss and step
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            model.eval()
            with torch.no_grad():
                test_loss, accuracy = validation(model, testloader, criterion)
                
            print(f'Epoch: {e+1}/{epochs}, Accuracy: {(accuracy/len(testloader))*100}%, Train loss: {running_loss}')
            running_loss = 0

            model.train()

