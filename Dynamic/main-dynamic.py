import argparse
import torch
import torch.nn.functional as F
from networks import GNNs
from dynamicGnn import *
from torch import tensor
from torch.optim import Adam
import numpy as np
import os,random
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GCNConv, GINConv,SAGEConv,GraphConv,TransformerConv,ResGatedGraphConv,ChebConv,GATConv,SGConv,GeneralConv
from torch_geometric.loader import DataLoader
# from gin import *
import os,random
import os.path as osp
from utils import *
import sys
import time
from torch_geometric.data import Batch
from sklearn.decomposition import PCA
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='DynHCPGender') #HCPTask, HCPRest
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--x', type=str, default="corr")
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--model', type=str, default="TransformerConv")
parser.add_argument('--hidden1', type=int, default=128)
parser.add_argument('--hidden2', type=int, default=32)
parser.add_argument('--num_heads', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--echo_epoch', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--early_stopping', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.5)
args = parser.parse_args()
path = "base_params/"
res_path = "base_results/"
path_data = "../../data/"
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(res_path):
    os.mkdir(res_path)
def logger(info):
    f = open(os.path.join(res_path, 'dynamic_results.csv'), 'a')
    print(info, file=f)
log = "dataset,model,hidden, num_layers,epochs,batch size, loss, acc, std"
logger(log)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

start = time.time()
# Please see NeuroGraph's documentation for downloading the procesed datasets. This block of code (data loading) may need to be revised for incorporating the latest datasets.
if args.dataset=='DynHCPGender':
    dataset_raw = torch.load(os.path.join(path_data,args.dataset,"processed", args.dataset+".pt"))
    dataset,labels = [],[]
    for v in dataset_raw:
        batches = v.get('batches')
        if len(batches)>0:
            for b in batches:
                y = b.y[0].item()
                dataset.append(b)
                labels.append(y)
else:
    dataset = torch.load(os.path.join(path_data,args.dataset,"processed", args.dataset+".pt"))
    labels = dataset['labels']
    dataset = dataset['batches']


print("dataset loaded successfully!",args.dataset)


train_tmp, test_indices = train_test_split(list(range(len(labels))),
                        test_size=0.2, stratify=labels,random_state=123,shuffle= True)

tmp = [dataset[i] for i in train_tmp]
labels_tmp = [labels[i] for i in train_tmp]
train_indices, val_indices = train_test_split(list(range(len(labels_tmp))),
test_size=0.125, stratify=labels_tmp,random_state=123,shuffle = True)
train_dataset = [tmp[i] for i in train_indices]
val_dataset = [tmp[i] for i in val_indices]
train_labels= [labels_tmp[i] for i in train_indices]
val_labels = [labels_tmp[i] for i in val_indices]
test_dataset = [dataset[i] for i in test_indices]
test_labels =[labels[i] for i in test_indices]


print("dataset {} loaded with train {} val {} test {} splits".format(args.dataset,len(train_dataset), len(val_dataset), len(test_dataset)))

args.num_features,args.num_classes = 100,len(np.unique(labels))
print("number of features and classes",args.num_features,args.num_classes)
criterion = torch.nn.CrossEntropyLoss()
def train(train_loader):
    model.train()
    total_loss = 0
    for data in train_loader:  
        data = data.to(args.device)
        out = model(data).reshape(1,-1)

        loss = criterion(out, data[0].y) 
        total_loss +=loss
        loss.backward()
        optimizer.step() 
        optimizer.zero_grad()
    return total_loss/len(train_loader)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:  
        data = data.to(args.device)
        out = model(data)
        out = torch.argmax(out)
        if out == data[0].y:
            correct +=1  
    return correct / len(loader)  


seeds = [123,124]
for index in range(args.runs):
    # this block of code needs to be updated for the recent pytoch versions
    # torch.manual_seed(seeds[index])
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seeds[index])
    # random.seed(seeds[index])
    # np.random.seed(seeds[index])
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    gnn = eval(args.model)
    model = DynamicGNN(args.num_features,args.hidden1,args.hidden2,args.num_heads,args.num_layers,gnn,args.dropout, args.num_classes).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss, test_acc = [],[]
    best_val_acc,best_val_loss,pat = 0.0,0.0,0
    for epoch in range(args.epochs):
        ep_start = time.time()
        loss = train(train_dataset)
        val_acc = test(val_dataset)
        test_acc = test(test_dataset)
        if epoch%10==0:
            print("epoch: {}, loss: {}, val_acc:{}, test_acc:{}".format(epoch, np.round(loss.item(),6), np.round(val_acc,2),np.round(test_acc,2)))
        # val_acc_history.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            pat = 0 
            torch.save(model.state_dict(), path + args.dataset+args.model+'-checkpoint-best-acc.pkl')
        else:
            pat += 1
        if pat >=args.early_stopping and epoch > args.epochs // 2:
            print("early stopped!")
            break
        ep_end = time.time()
        print("epoch time:", ep_end-ep_start)
    model.load_state_dict(torch.load(path + args.dataset+args.model+'-checkpoint-best-acc.pkl'))
    model.eval()
    test_acc = test(test_dataset)
    test_loss = train(test_dataset).item()
    
