import nibabel as nib
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_atlas_schaefer_2018
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import InMemoryDataset, download_url
import os,glob
import math
import csv
from torch_geometric.data import Batch
from collections import Counter
# from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import load_img
import torch
import networkx as nx
import pandas as pd
import zipfile
from torch_geometric.utils import degree,to_networkx
import io, shutil
from torch_geometric.data import Data
import os
import boto3
import networkx as nx
import pickle
from scipy.stats import zscore
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools
import random
import shutil
from random import randrange





class Brain_connectome_Rest(InMemoryDataset):
    r"""
    Graph-based neuroimaging benchmark datasets,


    Args:
        root (str): Root directory where the dataset should be saved.

        name (str): The `name
            <https://anwar-said.github.io/anwarsaid/neurograph.html/>`_ of the
            dataset.

        rois (int): the number of ROIs to be used for parcellation. 
        Options are: {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}

        path_to_data: Path to the fMRI data.

        n_jobs: the number of jobs to process the data in parallel.

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
        
        return: Pytroch geometric dataset with four classes: [0] - gender, [1]-age, [2]- working memory and [3] -fluid intelligence 
    """
    def __init__(self, root,name,n_rois, threshold,path_to_data,n_jobs, transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name,self.n_rois,self.threshold,self.path_to_data,self.n_jobs = root, name,n_rois,threshold,path_to_data,n_jobs
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']
    
    def extract_from_3d_no(self,volume, fmri):
        """
        Extract time-series data from a 3d atlas with non-overlapping ROIs.
        
        Inputs:
            path_to_atlas = '/path/to/atlas.nii.gz'
            path_to_fMRI = '/path/to/fmri.nii.gz'
            
        Output:
            returns extracted time series # volumes x # ROIs
        """

        subcor_ts = []
        for i in np.unique(volume):
            if i != 0: 
    #             print(i)
                bool_roi = np.zeros(volume.shape, dtype=int)
                bool_roi[volume == i] = 1
                bool_roi = bool_roi.astype(np.bool)
    #             print(bool_roi.shape)
                # extract time-series data for each roi
                roi_ts_mean = []
                for t in range(fmri.shape[-1]):
                    roi_ts_mean.append(np.mean(fmri[:, :, :, t][bool_roi]))
                subcor_ts.append(np.array(roi_ts_mean))
        Y = np.array(subcor_ts).T
        return Y


    def construct_adj_postive_perc(self,corr):
        """construct adjacency matrix from the given correlation matrix and threshold"""
        corr_matrix_copy = corr.detach().clone()
        threshold = np.percentile(corr_matrix_copy[corr_matrix_copy > 0], 100 - self.threshold)
        corr_matrix_copy[corr_matrix_copy < threshold] = 0
        corr_matrix_copy[corr_matrix_copy >= threshold] = 1
        return corr_matrix_copy

    def get_data_obj(self,iid,behavioral_data,path_to_data,volume):
        try:
            image_path_LR = os.path.join(path_to_data, iid+"_rfMRI_REST1_LR.nii.gz")
            reg_path = os.path.join(path_to_data, iid+"_Movement_Regressors.txt")
            img = nib.load(image_path_LR)
            if img.shape[3]<1200:
                return None
            regs = np.loadtxt(reg_path)
            fmri = img.get_fdata()
            Y = self.extract_from_3d_no(volume,fmri)
            start = 1
            stop = Y.shape[0]
            step = 1
            # detrending
            t = np.arange(start, stop+step, step)
            tzd = zscore(np.vstack((t, t**2)), axis=1)
            XX = np.vstack((np.ones(Y.shape[0]), tzd))
            B = np.matmul(np.linalg.pinv(XX).T,Y)
            Yt = Y - np.matmul(XX.T,B) 
            # regress out head motion regressors
            B2 = np.matmul(np.linalg.pinv(regs),Yt)
            Ytm = Yt - np.matmul(regs,B2) 
            # zscore over axis=0 (time)
            zd_Ytm = (Ytm - np.nanmean(Ytm, axis=0)) / np.nanstd(Ytm, axis=0, ddof=1)
            conn = ConnectivityMeasure(kind='correlation')
            fc = conn.fit_transform([Ytm])[0]
            zd_fc = conn.fit_transform([zd_Ytm])[0]
            fc *= np.tri(*fc.shape)
            np.fill_diagonal(fc, 0)

            # zscored upper triangle
            zd_fc *= 1 - np.tri(*zd_fc.shape, k=-1)
            np.fill_diagonal(zd_fc, 0)
            corr = torch.tensor(fc + zd_fc).to(torch.float)
            iid = int(iid)
            gender = behavioral_data.loc[iid,'Gender']
            g = 1 if gender=="M" else 0
            labels = torch.tensor([g,behavioral_data.loc[iid,'AgeClass'],behavioral_data.loc[iid,'ListSort_AgeAdj'],behavioral_data.loc[iid,'PMAT24_A_CR']])
            A = self.construct_adj_postive_perc(corr)
            edge_index = A.nonzero().t().to(torch.long)
            
            data = Data(x = corr, edge_index=edge_index, y = labels)
        except:
            return None
        return data


#         ...
    def process(self):
        behavioral_df = pd.read_csv(os.path.join(self.root, 'HCP_behavioral.csv')).set_index('Subject')[['Gender','Age','ListSort_AgeAdj','PMAT24_A_CR']]
        mapping = {'22-25':0, '26-30':1,'31-35':2,'36+':3}
        behavioral_df['AgeClass'] = behavioral_df['Age'].replace(mapping)

        dataset = []
        with open(os.path.join(self.root,"ids.pkl"),'rb') as f:
            ids = pickle.load(f)
        
        roi = fetch_atlas_schaefer_2018(n_rois=self.n_rois,yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()
        data_list = []
        data_list = Parallel(n_jobs=self.n_jobs)(delayed(self.get_data_obj)(iid,behavioral_df,self.path_to_data,volume) for iid in tqdm(ids))
        # Remove None
        dataset = [x for x in data_list if x is not None]
        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

class Gender_Dataset(InMemoryDataset):
    r"""
    Graph-based neuroimaging dataset,


    Args:
        root (str): Root directory where the dataset should be saved.

        name (str): The `name
            <https://anwar-said.github.io/anwarsaid/neurograph.html/>`_ of the
            dataset.

        

        dataset: Pyg dataset obtained from Brain_connectome_Rest class 

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
        
        return: Pytroch geometric dataset
    """
    def __init__(self, root,dataset_name, dataset,transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name, self.dataset = root, dataset_name,dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']

    def process(self):
        gender_dataset = []
        for d in self.dataset:
            labels = d.y
            gender = labels[0].item()
            data = Data(x= d.x, edge_index=d.edge_index,y = int(gender))
            gender_dataset.append(data)
        if self.pre_filter is not None:
            gender_dataset = [data for data in gender_dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            gender_dataset = [self.pre_transform(data) for data in gender_dataset]

        data, slices = self.collate(gender_dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])


class Age_Dataset(InMemoryDataset):
    r"""
    Graph-based neuroimaging dataset,


    Args:
        root (str): Root directory where the dataset should be saved.

        name (str): The `name
            <https://anwar-said.github.io/anwarsaid/neurograph.html/>`_ of the
            dataset.

        

        dataset: Pyg dataset obtained from Brain_connectome_Rest class 

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
        
        return: Pytroch geometric dataset
    """
    def __init__(self, root,dataset_name, dataset,transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name, self.dataset = root, dataset_name,dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']

    def process(self):
        age_dataset = []
        for d in self.dataset:
            labels = d.y
            age = labels[1].item()
            if int(age)<=2:
                data = Data(x= d.x, edge_index=d.edge_index,y = int(age))
                age_dataset.append(data)
        if self.pre_filter is not None:
            age_dataset = [data for data in age_dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            age_dataset = [self.pre_transform(data) for data in age_dataset]

        data, slices = self.collate(age_dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

class WM_Dataset(InMemoryDataset):
    r"""
    Graph-based neuroimaging dataset,


    Args:
        root (str): Root directory where the dataset should be saved.

        name (str): The `name
            <https://anwar-said.github.io/anwarsaid/neurograph.html/>`_ of the
            dataset.

        

        dataset: Pyg dataset obtained from Brain_connectome_Rest class 

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
        
        return: Pytroch geometric dataset with working memory values  as labels
        """
    def __init__(self, root,dataset_name, dataset,transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name, self.dataset = root, dataset_name,dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']


    def process(self):
        wm_dataset = []
        for d in self.dataset:
            labels = d.y
            wm = labels[2].item()
            if wm is not None:
                data = Data(x= d.x, edge_index=d.edge_index,y = wm)
                wm_dataset.append(data)
        print(len(wm_dataset))
       
        if self.pre_filter is not None:
            wm_dataset = [data for data in wm_dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            wm_dataset = [self.pre_transform(data) for data in wm_dataset]

        data, slices = self.collate(wm_dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

class FI_Dataset(InMemoryDataset):
    r"""
    Graph-based neuroimaging dataset,


    Args:
        root (str): Root directory where the dataset should be saved.

        name (str): The `name
            <https://anwar-said.github.io/anwarsaid/neurograph.html/>`_ of the
            dataset.

        

        dataset: Pyg dataset obtained from Brain_connectome_Rest class 

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
        
        return: Pytroch geometric dataset with Fluid Intelligence values as labels
        """
    def __init__(self, root,dataset_name, dataset,transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name, self.dataset = root, dataset_name,dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']

    def process(self):
        fl_dataset = []
        for d in self.dataset:
            labels = d.y
            fl = labels[3].item()
            if not math.isnan(fl):
                data = Data(x= d.x, edge_index=d.edge_index,y = fl)
                fl_dataset.append(data)
                
            # else:
            #     print("none found")
        # print(len(fl_dataset))
        if self.pre_filter is not None:
        
            fl_dataset = [data for data in fl_dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            fl_dataset = [self.pre_transform(data) for data in fl_dataset]

        data, slices = self.collate(fl_dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

class Brain_connectome_Activity(InMemoryDataset):
    r"""
    Graph-based neuroimaging benchmark datasets,


    Args:
        root (str): Root directory where the dataset should be saved.

        name (str): The `name
            <https://anwar-said.github.io/anwarsaid/neurograph.html/>`_ of the
            dataset.

        rois (int): the number of ROIs to be used for parcellation. 
        Options are: {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}

        path_to_data: Path to the fMRI data.

        n_jobs: the number of jobs to process the data in parallel.

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
        
        return: Pytroch geometric dataset with activity as class
    """

    def __init__(self, root, dataset_name,n_rois, threshold,path_to_data,n_jobs,transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name,self.n_rois,self.threshold,self.path_to_data,self.n_jobs = root, dataset_name,n_rois,threshold,path_to_data,n_jobs
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']
    
    def extract_from_3d_no(self,volume, fmri):
        """
        Extract time-series data from a 3d atlas with non-overlapping ROIs.
        
        Inputs:
            path_to_atlas = '/path/to/atlas.nii.gz'
            path_to_fMRI = '/path/to/fmri.nii.gz'
            
        Output:
            returns extracted time series # volumes x # ROIs
        """

        subcor_ts = []
        for i in np.unique(volume):
            if i != 0: 
    #             print(i)
                bool_roi = np.zeros(volume.shape, dtype=int)
                bool_roi[volume == i] = 1
                bool_roi = bool_roi.astype(np.bool)
    #             print(bool_roi.shape)
                # extract time-series data for each roi
                roi_ts_mean = []
                for t in range(fmri.shape[-1]):
                    roi_ts_mean.append(np.mean(fmri[:, :, :, t][bool_roi]))
                subcor_ts.append(np.array(roi_ts_mean))
        Y = np.array(subcor_ts).T
        return Y


    def construct_adj_postive_perc(self,corr):
        """construct adjacency matrix from the given correlation matrix and threshold"""
        corr_matrix_copy = corr.detach().clone()
        threshold = np.percentile(corr_matrix_copy[corr_matrix_copy > 0], 100 - self.threshold)
        corr_matrix_copy[corr_matrix_copy < threshold] = 0
        corr_matrix_copy[corr_matrix_copy >= threshold] = 1
        return corr_matrix_copy

    def get_data_obj_task(self,iid,target_path,volume):
        emotion_path = "tfMRI_EMOTION_LR.nii.gz"
        reg_path = "Movement_Regressors.txt"

        gambling_path = "tfMRI_GAMBLING_LR.nii.gz"
        
        language_path = "tfMRI_LANGUAGE_LR.nii.gz"

        motor_path = "tfMRI_MOTOR_LR.nii.gz"
        relational_path = "tfMRI_RELATIONAL_LR.nii.gz"

        social_path = "tfMRI_SOCIAL_LR.nii.gz"

        wm_path = "tfMRI_WM_LR.nii.gz"
        all_paths = [emotion_path,gambling_path,language_path,motor_path,relational_path,social_path,wm_path]
        data_list = []
        for y, path in enumerate(all_paths):
            try:
                image_path_LR = os.path.join(target_path, iid+"_"+path)
                reg_path = os.path.join(target_path, reg_path+"_"+reg_path)
                img = nib.load(image_path_LR)
                regs = np.loadtxt(reg_path)
                # regs_dt = np.loadtxt(regdt_path)
                fmri = img.get_fdata()
                Y = self.extract_from_3d_no(volume,fmri)
                start = 1
                stop = Y.shape[0]
                step = 1
                # detrending
                t = np.arange(start, stop+step, step)
                tzd = zscore(np.vstack((t, t**2)), axis=1)
                XX = np.vstack((np.ones(Y.shape[0]), tzd))
                B = np.matmul(np.linalg.pinv(XX).T,Y)
                Yt = Y - np.matmul(XX.T,B) 
                # regress out head motion regressors
                B2 = np.matmul(np.linalg.pinv(regs),Yt)
                Ytm = Yt - np.matmul(regs,B2) 

                # zscore over axis=0 (time)
                zd_Ytm = (Ytm - np.nanmean(Ytm, axis=0)) / np.nanstd(Ytm, axis=0, ddof=1)
                conn = ConnectivityMeasure(kind='correlation')
                fc = conn.fit_transform([Ytm])[0]
                zd_fc = conn.fit_transform([zd_Ytm])[0]
                fc *= np.tri(*fc.shape)
                np.fill_diagonal(fc, 0)

                # zscored upper triangle
                zd_fc *= 1 - np.tri(*zd_fc.shape, k=-1)
                np.fill_diagonal(zd_fc, 0)
                corr = torch.tensor(fc + zd_fc).to(torch.float)
                A = self.construct_Adj_postive_perc(corr)
                edge_index = A.nonzero().t().to(torch.long)
            
                data = Data(x = corr, edge_index=edge_index, y = y)
                data_list.append(data)

                # os.remove(image_path_LR)
                # os.remove(reg_path)
            except:
                print("file skipped!") 
            
        return data_list

    def process(self):
        dataset = []
        target_path = self.target_dir
        
        with open(os.path.join(self.root,"ids.pkl"),'rb') as f:
            ids = pickle.load(f)
        roi = fetch_atlas_schaefer_2018(n_rois=self.n_rois,yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()
        data_list = Parallel(n_jobs=self.n_jobs)(delayed(self.get_data_obj_task)(iid,self.path_to_data,volume) for iid in tqdm(ids))
        print("length of data list:", len(data_list))       
        dataset = list(itertools.chain(*data_list))
        
        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])


class Brain_connectome_Rest_Download(InMemoryDataset):
    r"""
        Graph-based neuroimaging benchmark datasets crawling from HCP S3 bucket,


        Args:
            root (str): Root directory where the dataset should be saved.

            name (str): The `name
                <https://anwar-said.github.io/anwarsaid/neurograph.html/>`_ of the
                dataset.

            rois (int): the number of ROIs to be used for parcellation. 
            Options are: {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}

            path_to_data: Path where the data to be stored and processed.

            n_jobs: the number of jobs to process the data in parallel.

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
            
            return: Pytroch geometric dataset with four classes: [0] - gender, [1]-age, [2]- working memory and [3] -fluid intelligence 
        """
    def __init__(self, root,name,n_rois, threshold,path_to_data,n_jobs,s3, transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name,self.n_rois,self.threshold,self.target_path,self.n_jobs,self.s3 = root, name,n_rois,threshold,path_to_data,n_jobs,s3
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']
    

    def extract_from_3d_no(self,volume, fmri):
        ''' 
        Extract time-series data from a 3d atlas with non-overlapping ROIs.
        
        Inputs:
            path_to_atlas = '/path/to/atlas.nii.gz'
            path_to_fMRI = '/path/to/fmri.nii.gz'
            
        Output:
            returns extracted time series # volumes x # ROIs
        '''

        subcor_ts = []
        for i in np.unique(volume):
            if i != 0: 
    #             print(i)
                bool_roi = np.zeros(volume.shape, dtype=int)
                bool_roi[volume == i] = 1
                bool_roi = bool_roi.astype(np.bool)
    #             print(bool_roi.shape)
                # extract time-series data for each roi
                roi_ts_mean = []
                for t in range(fmri.shape[-1]):
                    roi_ts_mean.append(np.mean(fmri[:, :, :, t][bool_roi]))
                subcor_ts.append(np.array(roi_ts_mean))
        Y = np.array(subcor_ts).T
        return Y


    def construct_Adj_postive_perc(self,corr):
        corr_matrix_copy = corr.detach().clone()
        threshold = np.percentile(corr_matrix_copy[corr_matrix_copy > 0], 100 - self.threshold)
        corr_matrix_copy[corr_matrix_copy < threshold] = 0
        corr_matrix_copy[corr_matrix_copy >= threshold] = 1
        return corr_matrix_copy

    def get_data_obj(self,iid,behavioral_data,BUCKET_NAME,volume):
        try:
            mri_file_path = "HCP_1200/"+iid+"/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz"
            reg_path = "HCP_1200/" + iid + '/MNINonLinear/Results/rfMRI_REST1_LR/Movement_Regressors.txt'
            if not os.path.exists(os.path.join(self.target_path, iid+"_"+os.path.basename(mri_file_path))):
                self.s3.download_file(BUCKET_NAME, mri_file_path,os.path.join(self.target_path, iid+"_"+os.path.basename(mri_file_path)))
            if not os.path.exists(os.path.join(self.target_path, iid+"_"+os.path.basename(reg_path))):
                self.s3.download_file(BUCKET_NAME, reg_path,os.path.join(self.target_path, iid+"_"+os.path.basename(reg_path)))
            image_path_LR = os.path.join(self.target_path, iid+"_"+os.path.basename(mri_file_path))
            reg_path = os.path.join(self.target_path, iid+"_"+os.path.basename(reg_path))
            img = nib.load(image_path_LR)
            if img.shape[3]<1200:
                return None
            regs = np.loadtxt(reg_path)
            fmri = img.get_fdata()
            Y = self.extract_from_3d_no(volume,fmri)
            start = 1
            stop = Y.shape[0]
            step = 1
            # detrending
            t = np.arange(start, stop+step, step)
            tzd = zscore(np.vstack((t, t**2)), axis=1)
            XX = np.vstack((np.ones(Y.shape[0]), tzd))
            B = np.matmul(np.linalg.pinv(XX).T,Y)
            Yt = Y - np.matmul(XX.T,B) 
            # regress out head motion regressors
            B2 = np.matmul(np.linalg.pinv(regs),Yt)
            Ytm = Yt - np.matmul(regs,B2) 
            # zscore over axis=0 (time)
            zd_Ytm = (Ytm - np.nanmean(Ytm, axis=0)) / np.nanstd(Ytm, axis=0, ddof=1)
            conn = ConnectivityMeasure(kind='correlation')
            fc = conn.fit_transform([Ytm])[0]
            zd_fc = conn.fit_transform([zd_Ytm])[0]
            fc *= np.tri(*fc.shape)
            np.fill_diagonal(fc, 0)

            # zscored upper triangle
            zd_fc *= 1 - np.tri(*zd_fc.shape, k=-1)
            np.fill_diagonal(zd_fc, 0)
            corr = torch.tensor(fc + zd_fc).to(torch.float)
            iid = int(iid)
            gender = behavioral_data.loc[iid,'Gender']
            g = 1 if gender=="M" else 0
            labels = torch.tensor([g,behavioral_data.loc[iid,'AgeClass'],behavioral_data.loc[iid,'ListSort_AgeAdj'],behavioral_data.loc[iid,'PMAT24_A_CR']])
            A = self.construct_Adj_postive_perc(corr)
            edge_index = A.nonzero().t().to(torch.long)
            data = Data(x = corr, edge_index=edge_index, y = labels)
        except:
            return None
        return data


#         ...
    def process(self):
        behavioral_df = pd.read_csv(os.path.join(self.root,'HCP_behavioral.csv')).set_index('Subject')[['Gender','Age','ListSort_AgeAdj','PMAT24_A_CR']]
        mapping = {'22-25':0, '26-30':1,'31-35':2,'36+':3}
        behavioral_df['AgeClass'] = behavioral_df['Age'].replace(mapping)

        dataset = []
        BUCKET_NAME = 'hcp-openaccess'
        with open(os.path.join(self.root,"ids.pkl"),'rb') as f:
            ids = pickle.load(f)
        
        print(len(ids))
        roi = fetch_atlas_schaefer_2018(n_rois=self.n_rois,yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()
        data_list = Parallel(n_jobs=self.n_jobs)(delayed(self.get_data_obj)(iid,behavioral_df,BUCKET_NAME,volume) for iid in tqdm(ids))
        dataset = [x for x in data_list if x is not None]
        # print(len(dataset))
        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])


class Brain_connectome_Activity_Download(InMemoryDataset):
    r"""
    Graph-based neuroimaging benchmark datasets,


    Args:
        root (str): Root directory where the dataset should be saved.

        name (str): The `name
            <https://anwar-said.github.io/anwarsaid/neurograph.html/>`_ of the
            dataset.

        rois (int): the number of ROIs to be used for parcellation. 
        Options are: {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}

        path_to_data: Path to the fMRI data.

        n_jobs: the number of jobs to process the data in parallel.

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
        
        return: Pytroch geometric dataset with activity as class
    """

    def __init__(self, root, dataset_name,n_rois, threshold,path_to_data,n_jobs,s3,transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name,self.n_rois,self.threshold,self.target_path,self.n_jobs,self.s3 = root, dataset_name,n_rois,threshold,path_to_data,n_jobs,s3
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']

    def get_data_obj_task(self,iid,BUCKET_NAME,volume):
        emotion_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_EMOTION_LR/tfMRI_EMOTION_LR.nii.gz"
        reg_emo_path = "HCP_1200/" + iid + '/MNINonLinear/Results/tfMRI_EMOTION_LR/Movement_Regressors.txt'

        gambling_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_GAMBLING_LR/tfMRI_GAMBLING_LR.nii.gz"
        reg_gamb_path = "HCP_1200/" + iid + '/MNINonLinear/Results/tfMRI_GAMBLING_LR/Movement_Regressors.txt'

        language_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_LANGUAGE_LR/tfMRI_LANGUAGE_LR.nii.gz"
        reg_lang_path = "HCP_1200/" + iid + '/MNINonLinear/Results/tfMRI_LANGUAGE_LR/Movement_Regressors.txt'

        motor_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_MOTOR_LR/tfMRI_MOTOR_LR.nii.gz"
        reg_motor_path = "HCP_1200/" + iid + '/MNINonLinear/Results/tfMRI_MOTOR_LR/Movement_Regressors.txt'

        relational_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_RELATIONAL_LR/tfMRI_RELATIONAL_LR.nii.gz"
        reg_rel_path = "HCP_1200/" + iid + '/MNINonLinear/Results/tfMRI_RELATIONAL_LR/Movement_Regressors.txt'

        social_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_SOCIAL_LR/tfMRI_SOCIAL_LR.nii.gz"
        reg_soc_path = "HCP_1200/" + iid + '/MNINonLinear/Results/tfMRI_SOCIAL_LR/Movement_Regressors.txt'

        wm_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_WM_LR/tfMRI_WM_LR.nii.gz"
        reg_wm_path = "HCP_1200/" + iid + '/MNINonLinear/Results/tfMRI_WM_LR/Movement_Regressors.txt'
        all_paths = [emotion_path,gambling_path,language_path,motor_path,relational_path,social_path,wm_path]
        reg_paths = [reg_emo_path,reg_gamb_path,reg_lang_path,reg_motor_path,reg_rel_path,reg_soc_path,reg_wm_path]
        data_list = []
        
        for y, path in enumerate(all_paths):
            try:
                # print("processing path:",path)
                if not os.path.exists(os.path.join(self.target_path, iid+"_"+os.path.basename(path))):
                    self.s3.download_file(BUCKET_NAME, path,os.path.join(self.target_path, iid+"_"+os.path.basename(path)))
            # if not os.path.exists(os.path.join(target_path, iid+"_"+os.path.basename(reg_paths[y]))):
                rnd = random.randint(0,1000)
                reg_prefix = iid+str(rnd)
                self.s3.download_file(BUCKET_NAME, reg_paths[y],os.path.join(self.target_path, reg_prefix+"_"+os.path.basename(reg_paths[y])))
                
                image_path_LR = os.path.join(self.target_path, iid+"_"+os.path.basename(path))
                reg_path = os.path.join(self.target_path, reg_prefix+"_"+os.path.basename(reg_paths[y]))
                img = nib.load(image_path_LR)
                
                regs = np.loadtxt(reg_path)
                # regs_dt = np.loadtxt(regdt_path)
                fmri = img.get_fdata()
                Y = self.extract_from_3d_no(volume,fmri)
                start = 1
                stop = Y.shape[0]
                step = 1
                # detrending
                t = np.arange(start, stop+step, step)
                tzd = zscore(np.vstack((t, t**2)), axis=1)
                XX = np.vstack((np.ones(Y.shape[0]), tzd))
                B = np.matmul(np.linalg.pinv(XX).T,Y)
                Yt = Y - np.matmul(XX.T,B) 
                # regress out head motion regressors
                B2 = np.matmul(np.linalg.pinv(regs),Yt)
                Ytm = Yt - np.matmul(regs,B2) 

                # zscore over axis=0 (time)
                zd_Ytm = (Ytm - np.nanmean(Ytm, axis=0)) / np.nanstd(Ytm, axis=0, ddof=1)
                conn = ConnectivityMeasure(kind='correlation')
                fc = conn.fit_transform([Ytm])[0]
                zd_fc = conn.fit_transform([zd_Ytm])[0]
                fc *= np.tri(*fc.shape)
                np.fill_diagonal(fc, 0)

                # zscored upper triangle
                zd_fc *= 1 - np.tri(*zd_fc.shape, k=-1)
                np.fill_diagonal(zd_fc, 0)
                corr = torch.tensor(fc + zd_fc).to(torch.float)
                A = self.construct_Adj_postive_perc(corr)
                edge_index = A.nonzero().t().to(torch.long)
            
                data = Data(x = corr, edge_index=edge_index, y = y)
                data_list.append(data)

                # os.remove(image_path_LR)
                # os.remove(reg_path)
            except:
                print("file skipped!") 
            
        return data_list
    def extract_from_3d_no(self,volume, fmri):
        ''' 
        Extract time-series data from a 3d atlas with non-overlapping ROIs.
        
        Inputs:
            path_to_atlas = '/path/to/atlas.nii.gz'
            path_to_fMRI = '/path/to/fmri.nii.gz'
            
        Output:
            returns extracted time series # volumes x # ROIs
        '''

        subcor_ts = []
        for i in np.unique(volume):
            if i != 0: 
    #             print(i)
                bool_roi = np.zeros(volume.shape, dtype=int)
                bool_roi[volume == i] = 1
                bool_roi = bool_roi.astype(np.bool)
    #             print(bool_roi.shape)
                # extract time-series data for each roi
                roi_ts_mean = []
                for t in range(fmri.shape[-1]):
                    roi_ts_mean.append(np.mean(fmri[:, :, :, t][bool_roi]))
                subcor_ts.append(np.array(roi_ts_mean))
        Y = np.array(subcor_ts).T
        return Y


    def construct_Adj_postive_perc(self,corr):
        corr_matrix_copy = corr.detach().clone()
        threshold = np.percentile(corr_matrix_copy[corr_matrix_copy > 0], 100 - self.threshold)
        corr_matrix_copy[corr_matrix_copy < threshold] = 0
        corr_matrix_copy[corr_matrix_copy >= threshold] = 1
        return corr_matrix_copy
    def process(self):
        dataset = []
        BUCKET_NAME = 'hcp-openaccess'
        with open(os.path.join(self.root,"ids.pkl"),'rb') as f:
            ids = pickle.load(f)
        ids = ids[:2]
        roi = fetch_atlas_schaefer_2018(n_rois=self.n_rois,yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()
        data_list = Parallel(n_jobs=self.n_jobs)(delayed(self.get_data_obj_task)(iid,BUCKET_NAME,volume) for iid in tqdm(ids))
        print("length of data list:", len(data_list))       
        dataset = list(itertools.chain(*data_list))
        
        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])


