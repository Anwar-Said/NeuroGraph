from NeuroGraph.datasets import NeuroGraphDataset
import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import os,random
import os.path as osp
import sys
import time
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HCPGender')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--model', type=str, default="GCNConv")
parser.add_argument('--hidden', type=int, default=32)
parser.add_argument('--hidden_mlp', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--echo_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--early_stopping', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.5)
args = parser.parse_args()
path = "base_params/"
res_path = "results/"
root = "data/"
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(res_path):
    os.mkdir(res_path)
def logger(info):
    f = open(os.path.join(res_path, 'results_new.csv'), 'a')
    print(info, file=f)

fix_seed(args.seed)
dataset = NeuroGraphDataset(root=root, name= args.dataset)
print(dataset.num_classes)
print(len(dataset))

print("dataset loaded successfully!",args.dataset)
labels = [d.y.item() for d in dataset]

train_tmp, test_indices = train_test_split(list(range(len(labels))),
                        test_size=0.2, stratify=labels,random_state=args.seed,shuffle= True)
tmp = dataset[train_tmp]
train_labels = [d.y.item() for d in tmp]
train_indices, val_indices = train_test_split(list(range(len(train_labels))),
 test_size=0.125, stratify=train_labels,random_state=args.seed,shuffle = True)
train_dataset = tmp[train_indices]
val_dataset = tmp[val_indices]
test_dataset = dataset[test_indices]
print("dataset {} loaded with train {} val {} test {} splits".format(args.dataset,len(train_dataset), len(val_dataset), len(test_dataset)))
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
args.num_features,args.num_classes = dataset.num_features,dataset.num_classes

criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.L1Loss()
def train(train_loader):
    model.train()
    total_loss = 0
    for data in train_loader:  
        data = data.to(args.device)
        out = model(data)  
        loss = criterion(out, data.y) 
        total_loss +=loss
        loss.backward()
        optimizer.step() 
        optimizer.zero_grad()
    return total_loss/len(train_loader.dataset)
    # return total_loss/len(train_loader) # For L1 loss. This may retun higher loss on the regression tasks since the paper used (total_loss/len(train_loader.dataset))

@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for data in loader:  
        data = data.to(args.device)
        out = model(data)  
        pred = out.argmax(dim=1)  
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)  

val_acc_history, test_acc_history, test_loss_history = [],[],[]
seeds = [123,124]
for index in range(args.runs):
    start = time.time()
    fix_seed(seeds[index])
    gnn = eval(args.model)
    model = ResidualGNNs(args,train_dataset,args.hidden,args.hidden_mlp,args.num_layers,gnn).to(args.device) ## apply GNN*
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters is: {total_params}")
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss, test_acc = [],[]
    best_val_acc,best_val_loss = 0.0,0.0
    for epoch in range(args.epochs):
        loss = train(train_loader)
        val_acc = test(val_loader)
        test_acc = test(test_loader)
        # if epoch%10==0:
        print("epoch: {}, loss: {}, val_acc:{}, test_acc:{}".format(epoch, np.round(loss.item(),6), np.round(val_acc,2),np.round(test_acc,2)))
        val_acc_history.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if epoch> int(args.epochs/2):## save the best model
                torch.save(model.state_dict(), path + args.dataset+args.model+'task-checkpoint-best-acc.pkl')
       

    #test the model   
    model.load_state_dict(torch.load(path + args.dataset+args.model+'task-checkpoint-best-acc.pkl'))
    model.eval()
    test_acc = test(test_loader)
    test_loss = train(test_loader).item()
    test_acc_history.append(test_acc)
    test_loss_history.append(test_loss)
