from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import InMemoryDataset, download_url
import os,glob
import torch
import zipfile


class NeuroGraphStatic(InMemoryDataset):

    def __init__(self, root,name, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.root = root
        self.urls = {"HCPGender":'https://vanderbilt.box.com/shared/static/r6hlz2arm7yiy6v6981cv2nzq3b0meax.zip',
                    "HCPActivity":'https://vanderbilt.box.com/shared/static/b4g59ibn8itegr0rpcd16m9ajb2qyddf.zip',
                    "HCPAge":'https://vanderbilt.box.com/shared/static/lzzks4472czy9f9vc8aikp7pdbknmtfe.zip',
                    "HCPWM":'https://vanderbilt.box.com/shared/static/xtmpa6712fidi94x6kevpsddf9skuoxy.zip',
                    "HCPFI":'https://vanderbilt.box.com/shared/static/g2md9h9snh7jh6eeay02k1kr9m4ido9f.zip'
                    }
        super(NeuroGraphStatic,self).__init__(root, name,transform, pre_transform, pre_filter)
        
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
        return [self.name+'.pt']
    
    
    def download(self):
        # Download to `self.raw_dir`.
        print("downloading the data. The files are large and may take a few minutes")
        if self.urls.get(self.name):
            download_url(self.urls.get(self.name), self.raw_dir)
            basename = os.path.basename(self.urls.get(self.name))
            with zipfile.ZipFile(os.path.join(self.raw_dir,basename), 'r') as file:
                file.extractall(os.path.join(self.raw_dir,os.path.dirname(basename)))
            # self.remove(os.path.join(self.raw_dir,basename))
        else:
            print('dataset not found! The name of the datasets are: "HCPGender","HCPActivity","HCPAge","HCPWM","HCPFI"')
    
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
        data, slices = torch.load(os.path.join(self.raw_dir,self.name,"processed", self.name+'.pt'))
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
