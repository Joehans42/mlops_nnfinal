import argparse
import sys

import click
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from data import corruptData
from model import MyShittyModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyShittyModel()
    train_set = corruptData(train=True)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 15

    for e in range(epochs):
        losses = []
        for images, labels in trainloader:
            optimizer.zero_grad()
            ps = model(images)
            loss = criterion(ps, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        print(f'Epoch: {e+1}/{epochs}, Loss: {loss}')
    torch.save(model.state_dict(), 'model_chkpt.pt')

    plt.plot(losses, 'r--')
    plt.xlabel('Training steps')
    plt.ylabel('Training loss')
    plt.savefig('training_losscurve.png')

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyShittyModel()
    model.load_state_dict(torch.load(model_checkpoint))
    
    test_set = corruptData(train=False)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64)

    total_correct = 0
    total = 0

    for image, label in testloader:
        ps = model(image)
        ps = ps.argmax(dim=-1)

        num_correct = torch.sum(ps == label)
        total_correct += num_correct
        total += len(label)
    
    print(f'Test accuracy: {total_correct/total*100}%')

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    