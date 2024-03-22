import torch
import torch.nn as nn
import mlflow
from pathlib import Path
import argparse
from tqdm import tqdm
import h5py
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model import ViTMAE
import numpy as np
from einops import rearrange


optimizers_map = {
    'adamw': torch.optim.AdamW,
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

def load_weights(wpath):
    print(wpath)
    weights = torch.load(wpath)
    for key in list(weights.keys()):
        weights[key.replace('module.','')] = weights.pop(key)
    return weights

def load_data(args):
    data = h5py.File('Dataset_Specific_labelled.h5', 'r')  
    X = data['jet'][...]
    y = data['Y'][...]
    y = np.squeeze(y)
    X = rearrange(X, 'b h w c -> b c h w')
    yt = np.zeros((y.shape[0],2))
    yt[:,0] = 1 - y
    yt[:,1] = y
    y = yt
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    trainset = TensorDataset(torch.tensor(X_train,dtype=torch.float32), torch.tensor(y_train,dtype=torch.float32))
    testset = TensorDataset(torch.tensor(X_test,dtype=torch.float32), torch.tensor(y_test,dtype=torch.float32))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    return trainloader, testloader

def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--runname', type=str,default='first')
    parser.add_argument('--wpath', type=str,default='./maskrat15/last.pt')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--optim', type=str, default='AdamW')
    args = parser.parse_args()
    return args

def run_one_epoch(dataloader,lossfn,model,optimizer,device):
    model.train()
    running_loss = 0.0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lossfn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def test_model(dataloader, model, lossfn, device):  
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
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


def train(args, model):
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizers_map[args.optim.lower()](model.parameters(), lr=args.lr)

    trainloader, testloader = load_data(args)
    model.to(device)
    Path(f'./{args.runname}').mkdir(exist_ok=True, parents=True)
    
    mlflow.set_experiment('FineTuning Experiments')
    mlflow.start_run(run_name=args.runname)
    mlflow.log_params(vars(args))
    
    for epoch in tqdm(range(args.epochs)):  # loop over the dataset multiple times
        running_loss = run_one_epoch(trainloader,criterion,model,optimizer,device)
        acc, test_loss = test_model(testloader, model, criterion, device)
        mlflow.log_metric("train_loss", running_loss, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        mlflow.log_metric("test_acc", acc, step=epoch)
            
    torch.save(model.state_dict(), f'{args.runname}/model_finetuned.pth')
    mlflow.end_run()

    
class FineTunedModel(nn.Module):
    def __init__(self,weights_path=None):
        super(FineTunedModel, self).__init__()
        self.model = ViTMAE().cpu()
        if weights_path is not None:
            self.model.load_state_dict(load_weights(weights_path))
        self.model.decoder_embed = nn.Identity()
        self.mask_token = nn.Identity()
        self.decoder_pos_embed = nn.Identity()
        self.decoder_blocks = nn.Identity()
        self.decoder_norm = nn.Identity()
        self.decoder_pred = nn.Identity()
        self.fc = nn.Linear(626*256, 2)
        
    def forward(self, x):
        x = self.model.forward_encoder(x,0.5,True)
        x = x.view(x.size(0),-1)
        return self.fc(x)


if __name__ == "__main__":
    args = get_args_parser()
    model = FineTunedModel(args.wpath)
    train(args, model)