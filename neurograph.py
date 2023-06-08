from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import InMemoryDataset, download_url
import os,glob
import torch


class BrainConnectomeBenchmarks(InMemoryDataset):

    def __init__(self, root,name, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        if self.name=="HCPRest":
            self.url = "https://www.dropbox.com/s/mjx6yybnw00n521/HCPRest.pt?raw=1"
        if self.name=="HCPTask":
            self.url = "https://www.dropbox.com/s/ojegv3xrjvmeo2b/HCPTask.pt?raw=1" 
            
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_dir(self):
        return os.path.join(self.root,self.name, self.name+'raw')
    
    @property
    def raw_file_names(self):
        return [self.name]
    
    @property
    def processed_dir(self) -> str:
        name = 'processed'
        return os.path.join(self.root, self.name, name)
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    
    def download(self):
        # Download to `self.raw_dir`.
        print("downloading the data")
        download_url(self.url, self.raw_dir)
    
    @property
    def num_node_attributes(self) -> int:
        return self.data.x.shape[1]
    
    @property
    def num_node_labels(self):
        return None
    
    @property
    def num_edge_labels(self):
        return None
    
    def process(self):
        print("processing the data")
        data, slices = torch.load(os.path.join(self.raw_dir, self.name+'.pt'))
        num_samples = slices['x'].size(0)-1
        data_list = []
        for i in range(num_samples):
            start_x = slices['x'][i]
            end_x = slices['x'][i + 1]
            x= data.x[start_x:end_x,:]
            start_ei = slices['edge_index'][i]
            end_ei = slices['edge_index'][i + 1]
            edge_index = data.edge_index[:,start_ei:end_ei]
            y = data.y[i]
            data_sample = Data(x =x, edge_index = edge_index, y=y)
            data_list.append(data_sample)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
