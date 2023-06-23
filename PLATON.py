import argparse
from Pruner import Pruner
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from functools import partial
import wandb
import os

# Function to calculate accuracy
def calculate_accuracy(outputs, targets, topk=(1, 5)):
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    accuracies = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        accuracies.append(acc.item())
    
    return accuracies

# Function to train the model
def train_model(model, criterion, optimizer, dataloader,scheduler,platon,epoch):
    model.train()
    running_loss = 0.0

    for i, batch_data in enumerate(dataloader):
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        global_step = epoch * len(dataloader) + i
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        threshold,mask_threhold = platon.update_and_pruning(model,global_step)

        running_loss += loss.item()
        wandb.log({"loss": loss.item()})

        if i%100==0:
            print(f"Batch: {i+1}/{len(dataloader)}, Loss: {loss.item()}")
    scheduler.step()
    wandb.log({"lr": scheduler.get_last_lr()[0]})

    return running_loss / len(dataloader)

# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    total_samples = 0
    top1_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            total_samples += labels.size(0)

            # Forward pass
            outputs = model(inputs)

            # Calculate accuracy
            top1, top5 = calculate_accuracy(outputs, labels)
            top1_correct += top1 * labels.size(0)
            top5_correct += top5 * labels.size(0)

    top1_accuracy = top1_correct / total_samples
    top5_accuracy = top5_correct / total_samples

    wandb.log({"top1_accuracy": top1_accuracy})
    wandb.log({"top5_accuracy": top5_accuracy})

    return top1_accuracy, top5_accuracy


if __name__=='__main__':
    # create args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--sparsity_level', type=float, default=0.5, help='sparsity level')
    parser.add_argument('--beta1', type=float, default=0.85, help='beta1 for Pruning')
    parser.add_argument('--beta2', type=float, default=0.85, help='beta2 for Pruning')
    parser.add_argument('--deltaT',type=int,default=200,help='deltaT for Pruning')
    args=parser.parse_args()


    if os.path.isdir('./ckpt')==False:
        os.mkdir('./ckpt')
    # Load CIFAR-10 dataset
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),                        
        transforms.RandomCrop(224,padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Define the dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=14, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=14, pin_memory=True)

    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # sparsity_levels=[0.99]

    # Train and evaluate the model for different sparsity levels
    for sparsity_level in sparsity_levels:
        wandb.init(
                project='nndl',
                entity='gongcaho',
                name=f'PLATON_sparsity_{sparsity_level}',
                job_type="training",
                reinit=True)
        deit = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        deit.head = nn.Linear(deit.head.in_features, 10)  
        # Create the ViT model with L1 sparsity
        model = deit.to(device)
        best_1_acc=0

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        platon=Pruner(model,args,threshold=1-sparsity_level)
        # Train the model
        for epoch in range(args.epochs):
            train_loss = train_model(model, criterion, optimizer, train_loader, scheduler,platon,epoch)
            print(f"Sparsity Level: {sparsity_level}, Epoch: {epoch+1}, Train Loss: {train_loss}")
            # evaluate the model
            if (epoch+1)%5==0:
                top1_acc, top5_acc = evaluate_model(model, test_loader)
                print(f"Sparsity Level: {sparsity_level}, Top-1 Accuracy: {top1_acc}, Top-5 Accuracy: {top5_acc}")
                if top1_acc>best_1_acc:
                    best_1_acc=top1_acc
                    torch.save(model.state_dict(), f"ckpt/model_{sparsity_level}.pth")

        print("---------------------------------------------------------------------------")