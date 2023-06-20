import torch
import sys
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from neurograph import *
from torch_geometric.datasets import TUDataset


def main():
    """this is the main function for obtaining dataset"""
    name = "HCPGender"
    root = "data/"
    dataset = NeuroGraphStatic(root, name)
    print(dataset.num_classes)
    print(dataset.num_features)
    print(dataset.num_node_features)



if __name__=="__main__":
    main()

