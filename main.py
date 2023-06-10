import torch
from torch_geometric.loader import DataLoader
from neurograph import *
from torch_geometric.datasets import TUDataset

name = "HCPGender"
root = "data/"
dataset = NeuroGraphStatic(root, name)
data2 = TUDataset(root,"MUTAG")
# print(dataset.num_classes)
print(dataset.num_node_attributes)
for d in data2:
    print(d)
    break

