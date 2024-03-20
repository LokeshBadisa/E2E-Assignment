import torch
from tqdm import tqdm
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset
from einops import rearrange
import mlflow


def pre_train(expname,
              model,              
              dataset_train,
              epochs=1,
              batch_size=512,
              lr=0.001,
              num_workers=4,
              ):
    #np.mean(data, axis=(0,1,2)) : [0.07848097, 0.08429243, 0.05751758, 0.12098689, 1.2899013 , 1.1099757 , 1.15771   , 1.1159292 ]
    #np.std(data, axis=(0,1,2)) : [ 3.0687237,  3.2782698,  2.9819856,  3.2468746, 13.511705 , 12.441227 , 12.12112  , 11.721005 ]
    transform_train = transforms.Compose([
                transforms.RandomResizedCrop(125, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=[0.07848097, 0.08429243, 0.05751758, 0.12098689, 1.2899013 ,
       1.1099757 , 1.15771   , 1.1159292 ], std=[ 3.0687237,  3.2782698,  2.9819856,  3.2468746, 13.511705 ,
       12.441227 , 12.12112  , 11.721005 ])])
    
    dataset_train = torch.Tensor(dataset_train)
    dataset_train = rearrange(dataset_train, 'b h w c-> b c h w')
    dataset_train = transform_train(dataset_train)
    dataset_train = TensorDataset(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    model = nn.DataParallel(model)
    model = model.cuda()


    mlflow.set_experiment(f'{expname}')
    mlflow.start_run()
    mlflow.log_params({
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'num_workers': num_workers,
        'optimizer': optimizer.__class__.__name__,
        'transform_train': transform_train 
    })

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch, (data, ) in tqdm(enumerate(data_loader_train)):
            data = data.cuda()
            optimizer.zero_grad()
            loss,_,_ = model(data)
            loss = loss.mean()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss /= (batch+1)
        mlflow.log_metric('loss', epoch_loss, step=epoch)

    mlflow.end_run()