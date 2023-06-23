import torch
from torch_geometric.data import InMemoryDataset, download_url
import os
import zipfile

class NeuroGraphDynamic():
    r"""
    Graph-based neuroimaging benchmark datasets, *.e.g.*,
    :obj:`"DynHCPGender"`, :obj:`"DynHCPAge"`, :obj:`"DynHCPActivity"`,
    :obj:`"DynHCP-WM"`, or :obj:`"DynHCP-FI"`


    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The `name
            <https://anwar-said.github.io/anwarsaid/neurograph.html/>`_ of the
            dataset.
        
        return: a list of graphs in pyg format. Each graph has dynamic graphs batched in pyg batch 
    """
    def __init__(self,root, name):
        self.root = root
        self.name = name
        self.urls = {"DynHCPGender":'https://vanderbilt.box.com/shared/static/mj0z6unea34lfz1hkdwsinj7g22yohxn.zip',
                    "DynHCPActivity":'https://vanderbilt.box.com/shared/static/2so3fnfqakeu6hktz322o3nm2c8ocus7.zip',
                    "DynHCPAge":'https://vanderbilt.box.com/shared/static/195f9teg4t4apn6kl6hbc4ib4g9addtq.zip',
                    "DynHCPWM":'https://vanderbilt.box.com/shared/static/mxy8fq3ghm60q6h7uhnu80pgvfxs6xo2.zip',
                    "DynHCPFI":'https://vanderbilt.box.com/shared/static/un7w3ohb2mmyjqt1ou2wm3g87y1lfuuo.zip'
                    }
        if self.urls.get(name):
            self.download(self.urls.get(name))
        else:
            print('dataset not found! The name of the datasets are: "DynHCPGender","DynHCPActivity","DynHCPAge","DynHCPWM","DynHCPFI"')
        self.dataset, self.labels = self.load_data()
    
    def download(self,url):
        
        download_url(url, os.path.join(self.root, self.name))
        basename = os.path.basename(url)
        with zipfile.ZipFile(os.path.join(self.root,self.name,basename), 'r') as file:
            file.extractall(os.path.join(self.root,self.name,os.path.dirname(basename)))
            # self.remove(os.path.join(self.raw_dir,basename))
    def load_data(self):
        if self.name=='DynHCPActivity':
            dataset_raw = torch.load(os.path.join(self.root,self.name,self.name,"processed", self.name+".pt"))
            dataset,labels = [],[]
            for v in dataset_raw:
                batches = v.get('batches')
                if len(batches)>0:
                    for b in batches:
                        y = b.y[0].item()
                        dataset.append(b)
                        labels.append(y)
        else:
            dataset = torch.load(os.path.join(self.root,self.name,self.name,"processed", self.name+".pt"))
            labels = dataset['labels']
            dataset = dataset['batches']
        return dataset,labels
    

