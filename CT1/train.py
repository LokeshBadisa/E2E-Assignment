from model import resnet15v2
import torch
import torch.nn as nn
import h5py
import numpy as np
import argparse
import mlflow
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from einops import rearrange
from tqdm import tqdm
from pathlib import Path
from typing import Tuple

# Optimizers map
optimizers_map = {
    'adamw': torch.optim.AdamW,
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}


#Argparse section
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runname', type=str,required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    return args


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    # Load data
    electrons = h5py.File('SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5', 'r')
    photons = h5py.File('SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5', 'r')

    # Get data and labels
    e_data = electrons['X'][...]
    p_data = photons['X'][...]
    e_labels = electrons['y'][...]
    p_labels = photons['y'][...]

    # Concatenate electrons data and photons data
    data = np.concatenate((e_data, p_data))
    labels = np.concatenate((e_labels, p_labels))

    # One hot encode labels
    flabels = np.zeros((labels.shape[0], 2))
    flabels[range(labels.shape[0]), labels.astype(int)] = 1
    labels = flabels

    # Rearrange data to be in the format (batch, channels, height, width)
    data = rearrange(data, 'b h w c -> b c h w')

    # Close files
    electrons.close()
    photons.close()

    return data, labels


def get_data_loader(data : np.ndarray, 
                    labels : np.ndarray, 
                    batch_size : int, 
                    device : torch.device) -> \
                    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    # Data statistics for normalization
    mean = np.array([ 0.00114013, -0.00022465])
    std = np.array([0.02360118, 0.06654396])

    # Normalization transform
    transform = transforms.Normalize(mean,std)

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Transform data to tensors and move to device
    X_train = transform(torch.tensor(X_train)).float().to(device)
    X_test = transform(torch.tensor(X_test)).float().to(device)

    # Create datasets and dataloaders
    train_data = torch.utils.data.TensorDataset(X_train, torch.tensor(y_train).to(device).float())
    test_data = torch.utils.data.TensorDataset(X_test, torch.tensor(y_test).to(device).float())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def run_one_epoch(model : nn.Module, 
                  optimizer : torch.optim.Optimizer, 
                  criterion : nn.Module, 
                  train_loader : torch.utils.data.DataLoader) -> float:
    model.train()

    # Initialize variables
    running_loss = 0.0
    total = 0

    # Training routine
    for data in train_loader:
        inputs, labels = data
        total += labels.size(0)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss/total


def test(model : nn.Module, 
         criterion : nn.Module, 
         test_loader : torch.utils.data.DataLoader) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss = 0

    #Testing routine
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss += criterion(outputs, labels)
            labels = labels.argmax(dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total, loss / total


if __name__ == '__main__':

    # Load data
    data, labels = load_data()

    # Get arguments
    args = get_args_parser()
    device = torch.device(args.device)

    # Get data loaders
    trainloader, testloader = get_data_loader(data, labels, args.batch_size, device)

    # Initialize model, optimizer and criterion
    model = resnet15v2().to(device)
    optimizer = optimizers_map[args.optim.lower()](model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    

    # Start mlflow run for logging
    mlflow.set_experiment('Common Task 1')
    mlflow.start_run(run_name=args.runname)
    mlflow.log_params({
        'runname': args.runname,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'optimizer': args.optim,
        'device': args.device
    })

    
    global_acc = 0
    Path(f'{args.runname}').mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in tqdm(range(args.epochs)):
        running_loss = run_one_epoch(model, optimizer, criterion, trainloader)
        test_acc, test_loss = test(model, criterion, testloader)

        # Log metrics
        mlflow.log_metric('train_loss', running_loss, step=epoch)
        mlflow.log_metric('test_acc', test_acc, step=epoch)
        mlflow.log_metric('test_loss', test_loss, step=epoch)
        if test_acc > global_acc:
            global_acc = test_acc
            torch.save(model.state_dict(), f'{args.runname}/best.pth')

    # Save model
    mlflow.end_run()
    torch.save(model.state_dict(), f'{args.runname}/last.pth')