import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import torch
import zipfile
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)

class NeuroGraphStatic(InMemoryDataset):
    r"""The NeuroGraph benchmark datasets from the
    `"NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics"
    <https://arxiv.org/abs/2306.06202>`_ paper.
    :class:`NeuroGraphStatic` holds a collection of five neuroimaging graph
    learning datasets that span multiple categories of demographics, mental
    states, and cognitive traits.
    See the `documentation
    <https://neurograph.readthedocs.io/en/latest/NeuroGraph.html>`_ and the
    `Github <https://github.com/Anwar-Said/NeuroGraph>`_ for more details.

    +--------------------+---------+----------------------+
    | Dataset            | #Graphs | Task                 |
    +====================+=========+======================+
    | :obj:`HCPActivity` | 7,443   | Graph Classification |
    +--------------------+---------+----------------------+
    | :obj:`HCPGender`   | 1,078   | Graph Classification |
    +--------------------+---------+----------------------+
    | :obj:`HCPAge`      | 1,065   | Graph Classification |
    +--------------------+---------+----------------------+
    | :obj:`HCPFI`       | 1,071   | Graph Regression     |
    +--------------------+---------+----------------------+
    | :obj:`HCPWM`       | 1,078   | Graph Regression     |
    +--------------------+---------+----------------------+

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (one of :obj:`"HCPGender"`,
            :obj:`"HCPActivity"`, :obj:`"HCPAge"`, :obj:`"HCPFI"`,
            :obj:`"HCPWM"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    url = 'https://vanderbilt.box.com/shared/static'
    filenames = {
        'HCPGender': 'r6hlz2arm7yiy6v6981cv2nzq3b0meax.zip',
        'HCPActivity': 'b4g59ibn8itegr0rpcd16m9ajb2qyddf.zip',
        'HCPAge': 'static/lzzks4472czy9f9vc8aikp7pdbknmtfe.zip',
        'HCPWM': 'xtmpa6712fidi94x6kevpsddf9skuoxy.zip',
        'HCPFI': 'g2md9h9snh7jh6eeay02k1kr9m4ido9f.zip',
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        assert name in self.filenames.keys()
        self.name = name

        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self) -> str:
        return 'data.pt'

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = f'{self.url}/{self.filenames[self.name]}'
        path = download_url(url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        os.rename(
            osp.join(self.raw_dir, self.name, 'processed', f'{self.name}.pt'),
            osp.join(self.raw_dir, 'data.pt'))
        shutil.rmtree(osp.join(self.raw_dir, self.name))

    def process(self):
        data, slices = torch.load(self.raw_paths[0])

        num_samples = slices['x'].size(0) - 1
        data_list: List[Data] = []
        for i in range(num_samples):
            x = data.x[slices['x'][i]:slices['x'][i + 1]]
            edge_index = data.edge_index[
                :,
                slices['edge_index'][i]:slices['edge_index'][i + 1],
            ]
            sample = Data(x=x, edge_index=edge_index, y=data.y[i])

            if self.pre_filter is not None and not self.pre_filter(sample):
                continue

            if self.pre_transform is not None:
                sample = self.pre_transform(sample)

            data_list.append(sample)

        self.save(data_list, self.processed_paths[0])

class NeuroGraphDynamic():
    r"""Graph-based neuroimaging benchmark datasets, e.g.,
        :obj:`"DynHCPGender"`, :obj:`"DynHCPAge"`, :obj:`"DynHCPActivity"`,
        :obj:`"DynHCPWM"`, or :obj:`"DynHCPFI"`

        Args:
            root (str): Root directory where the dataset should be saved.
            name (str): The name of the dataset.

        Returns:
            list: A list of graphs in PyTorch Geometric (pyg) format. Each graph contains a list of dynamic graphs batched in pyg batch.
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
    

