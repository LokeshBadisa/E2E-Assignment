import torch
import torch.nn as nn
from model import *
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm 
import argparse
from sklearn.model_selection import train_test_split
from model import VGG
import mlflow
from einops import rearrange
from pathlib import Path
from typing import Tuple

# Optimizers map
optimizers_map = {
    'adamw': torch.optim.AdamW,
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

# Argparse Section
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runname', type=str,required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--gpu', type=str, default='cuda:6')

    args = parser.parse_args()
    return args


# Load data
def load_data() -> Tuple[np.ndarray, np.ndarray]:

    # Data files
    f1 = "QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272.test.snappy.parquet"
    f2 = "QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540.test.snappy.parquet"
    f3 = "QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494.test.snappy.parquet"

    # Load data into numpy arrays
    df = pq.read_table([f1,f2,f3]).to_pandas()
    X_jets = df['X_jets'].to_numpy()


    # Rearrange data
    data = []
    for i in tqdm(range(X_jets.shape[0])):
        another_data = []
        for j in range(3):
            new_data = np.stack(X_jets[i][j][:125], axis=-1)
            another_data.append(new_data)
        data.append(np.stack(another_data, axis=-1))

    data = np.array(data,dtype=np.float32)
    print("Data Loaded!!")

    # Rearrange data to nchw format
    data = rearrange(data, 'b h w c -> b c h w')

    # Load labels
    yt = df['y'].to_numpy()
    y = np.zeros((yt.shape[0],2),dtype=np.float32)
    y[:,0] = 1 - yt
    y[:,1] = yt

    return data, y


def get_dataloaders(args) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    # Load data
    X,y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create dataloaders
    train_data = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_data = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    return train_loader, test_loader


# Runs 1 epoch of training
def run_one_epoch(dataloader : torch.utils.data.DataLoader,
                  lossfn : nn.Module,
                  model : nn.Module,
                  optimizer : torch.optim.Optimizer,
                  device : torch.device) -> float:
    model.train()
    running_loss = 0.0

    # Training loop
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lossfn(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()        
    return running_loss


# Tests the model
def test_model(dataloader : torch.utils.data.DataLoader, 
               model : nn.Module, 
               lossfn : nn.Module, 
               device : torch.device) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    #Testing loop
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            outputs = model(inputs)
            loss = lossfn(outputs, labels)
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
            total_loss += loss.item()
    return correct / total , total_loss



if __name__ == "__main__":

    # Get arguments
    args = get_args_parser()

    # Get dataloaders
    train_loader, test_loader = get_dataloaders(args)
    device = torch.device(args.gpu)

    # Initialize model, loss function and optimizer
    model = VGG().to(device)
    lossfn = nn.CrossEntropyLoss()
    optimizer = optimizers_map[args.optim.lower()](model.parameters(), lr=args.lr)


    # Start mlflow run for logging experiments
    mlflow.set_experiment("CT2")
    mlflow.start_run(run_name=args.runname)
    mlflow.log_params(vars(args))
    Path(f'./{args.runname}').mkdir(exist_ok=True, parents=True )

    gloval_acc = 0


    # Training loop
    for epoch in tqdm(range(args.epochs)):
        running_loss = run_one_epoch(train_loader,lossfn,model,optimizer,device)
        mlflow.log_metric("train_loss", running_loss, step=epoch)

        acc, test_loss = test_model(test_loader, model, lossfn, device)


        # Log metrics
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        mlflow.log_metric("test_acc", acc, step=epoch)
        if acc > gloval_acc:
            gloval_acc = acc
            torch.save(model.cpu().state_dict(), f'{args.runname}/best.pth')
            model.to(device)


    # Save the model
    torch.save(model.cpu().state_dict(), f'{args.runname}/last.pth')
    mlflow.end_run()