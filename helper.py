import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker,NiftiLabelsMasker
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
from nilearn import plotting
import networkx as nx
from scipy.sparse import coo_matrix
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
from sklearn.decomposition import PCA
from random import randrange


ACCESS_KEY = '' #USE YOUR ACCESS AND SECRET KEY
SECRET_KEY = ''

# Set the HCP bucket name and file paths


# Create an S3 client
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

def extract_from_3d_no(volume, fmri):
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


def construct_Adj_postive_perc(corr, perc):
    corr_matrix_copy = corr.detach().clone()
    threshold = np.percentile(corr_matrix_copy[corr_matrix_copy > 0], 100 - perc)
    corr_matrix_copy[corr_matrix_copy < threshold] = 0
    corr_matrix_copy[corr_matrix_copy >= threshold] = 1
    return corr_matrix_copy

def get_data_obj(iid,behavioral_data,target_path,BUCKET_NAME,volume, threshold,feat):
    try:
        mri_file_path = "HCP_1200/"+iid+"/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz"
        reg_path = "HCP_1200/" + iid + '/MNINonLinear/Results/rfMRI_REST1_LR/Movement_Regressors.txt'
        if not os.path.exists(os.path.join(target_path, iid+"_"+os.path.basename(mri_file_path))):
            s3.download_file(BUCKET_NAME, mri_file_path,os.path.join(target_path, iid+"_"+os.path.basename(mri_file_path)))
        if not os.path.exists(os.path.join(target_path, iid+"_"+os.path.basename(reg_path))):
            s3.download_file(BUCKET_NAME, reg_path,os.path.join(target_path, iid+"_"+os.path.basename(reg_path)))
        image_path_LR = os.path.join(target_path, iid+"_"+os.path.basename(mri_file_path))
        reg_path = os.path.join(target_path, iid+"_"+os.path.basename(reg_path))
        img = nib.load(image_path_LR)
        if img.shape[3]<1200:
            return None
        regs = np.loadtxt(reg_path)
        fmri = img.get_fdata()
        Y = extract_from_3d_no(volume,fmri)
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
        A = construct_Adj_postive_perc(corr, threshold)
        edge_index = A.nonzero().t().to(torch.long)
        if feat=="signal":
            attr = torch.tensor(Ytm.T).to(torch.float)
        if feat=="corr":
            attr = corr
        if feat=="signal_corr":
            ytm = torch.tensor(Ytm.T).to(torch.float)
            attr = torch.cat((corr,ytm),1)
        data = Data(x = attr, edge_index=edge_index, y = labels)
        # print(data.num_nodes,data.num_edges, data.num_node_features,data.has_isolated_nodes(), data.has_self_loops(),data.is_directed(),data.y,label)
    except:
        return None
    return data


class Brain_connectome_Rest(InMemoryDataset):
    def __init__(self, root,dataset_name,feat, threshold,target_path, transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name,self.feat,self.threshold,self.target_path = root, dataset_name,feat,threshold,target_path
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']

#         ...
    def process(self):
        behavioral_df = pd.read_csv(os.path.join('data/HCP_behavioral.csv')).set_index('Subject')[['Gender','Age','ListSort_AgeAdj','PMAT24_A_CR']]
        mapping = {'22-25':0, '26-30':1,'31-35':2,'36+':3}
        behavioral_df['AgeClass'] = behavioral_df['Age'].replace(mapping)

        dataset = []
        target_path = self.target_path
        BUCKET_NAME = 'hcp-openaccess'
        with open("data/ids.pkl",'rb') as f:
            ids = pickle.load(f)
        roi = fetch_atlas_schaefer_2018(n_rois=1000,yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()
        data_list = []
        data_list = Parallel(n_jobs=2)(delayed(get_data_obj)(iid,behavioral_df,target_path,BUCKET_NAME,volume, self.threshold,self.feat) for iid in tqdm(ids))
        dataset = [x for x in data_list if x is not None]
        print(len(dataset))
        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

def get_data_obj_task(iid,target_path,BUCKET_NAME,volume, threshold,feat):
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
    max_ = 0
    for y, path in enumerate(all_paths):
        try:
            # print("processing path:",path)
            if not os.path.exists(os.path.join(target_path, iid+"_"+os.path.basename(path))):
                s3.download_file(BUCKET_NAME, path,os.path.join(target_path, iid+"_"+os.path.basename(path)))
        # if not os.path.exists(os.path.join(target_path, iid+"_"+os.path.basename(reg_paths[y]))):
            rnd = random.randint(0,1000)
            reg_prefix = iid+str(rnd)
            s3.download_file(BUCKET_NAME, reg_paths[y],os.path.join(target_path, reg_prefix+"_"+os.path.basename(reg_paths[y])))
            
            image_path_LR = os.path.join(target_path, iid+"_"+os.path.basename(path))
            reg_path = os.path.join(target_path, reg_prefix+"_"+os.path.basename(reg_paths[y]))
            img = nib.load(image_path_LR)
            
            regs = np.loadtxt(reg_path)
            # regs_dt = np.loadtxt(regdt_path)
            fmri = img.get_fdata()
            Y = extract_from_3d_no(volume,fmri)
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
            A = construct_Adj_postive_perc(corr, threshold)
            edge_index = A.nonzero().t().to(torch.long)
           
            data = Data(x = corr, edge_index=edge_index, y = y)
            data_list.append(data)

            # os.remove(image_path_LR)
            # os.remove(reg_path)
        except:
            print("file skipped!") 
        
    return data_list

class Brain_connectome_Task(InMemoryDataset):
    def __init__(self, root,dataset_name,feat,threshold,target_dir,transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name,self.feat,self.threshold,self.target_dir = root, dataset_name,feat,threshold,target_dir
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']


    def process(self):
        dataset = []
        target_path = self.target_dir
        BUCKET_NAME = 'hcp-openaccess'
        with open("data/ids.pkl",'rb') as f:
            ids = pickle.load(f)
        roi = fetch_atlas_schaefer_2018(n_rois=100,yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()
        data_list = Parallel(n_jobs=50)(delayed(get_data_obj_task)(iid,target_path,BUCKET_NAME,volume, self.threshold,self.feat) for iid in tqdm(ids))
        print("length of data list:", len(data_list))       
        dataset = list(itertools.chain(*data_list))
        
        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])
class Age_Datasets(InMemoryDataset):
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
        print(len(age_dataset))
        if self.pre_filter is not None:
            age_dataset = [data for data in age_dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            age_dataset = [self.pre_transform(data) for data in age_dataset]

        data, slices = self.collate(age_dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])
class WM_Datasets(InMemoryDataset):
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
class FI_Datasets(InMemoryDataset):
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
        print(len(fl_dataset))
        if self.pre_filter is not None:
        
            fl_dataset = [data for data in fl_dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            fl_dataset = [self.pre_transform(data) for data in fl_dataset]

        data, slices = self.collate(fl_dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])


def process_dynamic_fc(timeseries, window_size, window_stride, threshold,y, dynamic_length=None, sampling_init=None, self_loop=True):
    
    if dynamic_length is None:
        dynamic_length = timeseries.shape[0]
        sampling_init = 0
    else:
        if isinstance(sampling_init, int):
            assert timeseries.shape[0] > sampling_init + dynamic_length
    assert sampling_init is None or isinstance(sampling_init, int)
    # assert timeseries.ndim==3
    assert dynamic_length > window_size

    if sampling_init is None:
        sampling_init = randrange(timeseries.shape[0]-dynamic_length+1)
    sampling_points = list(range(sampling_init, sampling_init+dynamic_length-window_size, window_stride))

    dynamic_fc_list = []
    for i in sampling_points:
        slice = timeseries[i:i+window_size]
        zd_Ytm = (slice - np.nanmean(slice, axis=0)) / np.nanstd(slice, axis=0, ddof=1)
        conn = ConnectivityMeasure(kind='correlation')
        fc = conn.fit_transform([slice])[0]
        zd_fc = conn.fit_transform([zd_Ytm])[0]
        fc *= np.tri(*fc.shape)
        np.fill_diagonal(fc, 0)
        zd_fc *= 1 - np.tri(*zd_fc.shape, k=-1)
        np.fill_diagonal(zd_fc, 0)
        corr = torch.tensor(fc + zd_fc).to(torch.float)
        A = construct_Adj_postive_perc(corr, threshold)
        edge_index = A.nonzero().t().to(torch.long)
        data = Data(x = corr, edge_index=edge_index, y = y)
        dynamic_fc_list.append(data)
    return dynamic_fc_list

def get_dynamic_data_object(iid,behavioral_data,target_path,BUCKET_NAME,volume, threshold,feat, window_size, stride, dynamic_length):
    try:
        mri_file_path = "HCP_1200/"+iid+"/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz"
        reg_path = "HCP_1200/" + iid + '/MNINonLinear/Results/rfMRI_REST1_LR/Movement_Regressors.txt'
       
        if not os.path.exists(os.path.join(target_path, iid+"_"+os.path.basename(mri_file_path))):
            s3.download_file(BUCKET_NAME, mri_file_path,os.path.join(target_path, iid+"_"+os.path.basename(mri_file_path)))
        if not os.path.exists(os.path.join(target_path, iid+"_"+os.path.basename(reg_path))):
            s3.download_file(BUCKET_NAME, reg_path,os.path.join(target_path, iid+"_"+os.path.basename(reg_path)))
       
        image_path_LR = os.path.join(target_path, iid+"_"+os.path.basename(mri_file_path))
        reg_path = os.path.join(target_path, iid+"_"+os.path.basename(reg_path))
       
        img = nib.load(image_path_LR)
       
        regs = np.loadtxt(reg_path)
       
        fmri = img.get_fdata()
        Y = extract_from_3d_no(volume,fmri)
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
        iid = int(iid)
        gender = behavioral_data.loc[iid,'Gender']
        g = 1 if gender=="M" else 0
        labels = torch.tensor([g,behavioral_data.loc[iid,'AgeClass'],behavioral_data.loc[iid,'ListSort_AgeAdj'],behavioral_data.loc[iid,'PMAT24_A_CR']])

        
        dynamic_fc_list = process_dynamic_fc(Ytm, window_size, stride, threshold,labels,dynamic_length)
    except:
        return None
    return {"id":iid, "data":dynamic_fc_list}



class Dynamic_Brain_connectome_Rest(InMemoryDataset):
    def __init__(self, root,dataset_name,feat, threshold,window_size, stride, dynamic_length,target_dir,transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name,self.feat,self.threshold,self.window_size, self.stride, self.dynamic_length,self.target_path = root, dataset_name,feat,threshold,window_size, stride, dynamic_length,target_dir
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.dataset = torch.load(self.processed_paths[0])
    @property
    def processed_file_names(self):
        return [self.dataset_name+'.pt']


    def process(self):
        behavioral_df = pd.read_csv(os.path.join('data/HCP_behavioral.csv')).set_index('Subject')[['Gender','Age','ListSort_AgeAdj','PMAT24_A_CR']]
        
        mapping = {'22-25':0, '26-30':1,'31-35':2,'36+':3}
        behavioral_df['AgeClass'] = behavioral_df['Age'].replace(mapping)     # behavioral_dict = behavioral_df.to_dict()
        dataset = []
        
        target_path = self.target
        BUCKET_NAME = 'hcp-openaccess'
        with open("data/ids.pkl",'rb') as f:
            ids = pickle.load(f)
        roi = fetch_atlas_schaefer_2018(n_rois=100,yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        volume = atlas.get_fdata()
        
        data_list = Parallel(n_jobs=5)(delayed(get_dynamic_data_object)(iid,behavioral_df,target_path,BUCKET_NAME,volume, self.threshold,self.feat,self.window_size, self.stride, self.dynamic_length) for iid in tqdm(ids))
        
        print("saving path:",self.processed_paths[0])
        torch.save(data_list, self.processed_paths[0])

def dynamic_get_data_obj_task(iid,target_path,BUCKET_NAME,volume, threshold,window_size, stride, dynamic_length):
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
        
            iid = str(iid)
            if not os.path.exists(os.path.join(target_path, iid+"_"+os.path.basename(path))):
                s3.download_file(BUCKET_NAME, path,os.path.join(target_path, iid+"_"+os.path.basename(path)))
        
            rnd = random.randint(0,1000)
            reg_prefix = iid+str(rnd)
            s3.download_file(BUCKET_NAME, reg_paths[y],os.path.join(target_path, reg_prefix+"_"+os.path.basename(reg_paths[y])))
            
            image_path_LR = os.path.join(target_path, iid+"_"+os.path.basename(path))
            reg_path = os.path.join(target_path, reg_prefix+"_"+os.path.basename(reg_paths[y]))
            
            img = nib.load(image_path_LR)
            
            regs = np.loadtxt(reg_path)
            # regs_dt = np.loadtxt(regdt_path)
            fmri = img.get_fdata()
            Y = extract_from_3d_no(volume,fmri)
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
            Ytm = Yt - np.matmul(regs,B2) 
            iid = int(iid)
            dynamic_fc_list = process_dynamic_fc(Ytm, window_size, stride, threshold,y,dynamic_length)

            batch  = Batch.from_data_list(dynamic_fc_list)
            data_list.append(batch)
            # os.remove(image_path_LR)
            # os.remove(reg_path)
        except Exception as e:
            print("file skipped!",iid, type(e).__name__) 
        
    return {"id":iid, "batches":data_list}




class Dynamic_Brain_connectome_Task(InMemoryDataset):
    def __init__(self, root,dataset_name,threshold,window_size, stride, dynamic_length,transform=None, pre_transform=None, pre_filter=None):
        self.root, self.dataset_name,self.threshold,self.window_size, self.stride, self.dynamic_length = root, dataset_name,threshold,window_size, stride, dynamic_length
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.dataset = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        # return ['rest1_processed_gender.pt']
        return [self.dataset_name+'.pt']

    def process(self):
       
        target_path = "data/"
        BUCKET_NAME = 'hcp-openaccess'
        with open("data/ids.pkl",'rb') as f:
            ids = pickle.load(f)
        roi = fetch_atlas_schaefer_2018(n_rois=100,yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        
        volume = atlas.get_fdata()
        
        data_list = Parallel(n_jobs=10)(delayed(dynamic_get_data_obj_task)(iid,target_path,BUCKET_NAME,volume, self.threshold,self.window_size, self.stride, self.dynamic_length) for iid in tqdm(ids))
        
        print("length of data list:", len(data_list))
       
        print("saving path:",self.processed_paths[0])
        torch.save(data_list, self.processed_paths[0])