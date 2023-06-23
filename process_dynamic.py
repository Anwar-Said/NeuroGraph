import pandas as pd
import os
import nibabel as nib
import pickle
import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import load_img
from nilearn.connectome import ConnectivityMeasure
from scipy.stats import zscore
import torch
from torch_geometric.data import Data
from random import randrange

class Preprocess_dynamic():
    def __init__(self,root, name,s3,n_rois = 100, threshold = 10, window_size = 50,stride = 3,dynamic_length=150):
        self.root, self.name, self.s3,self.n_rois, self.threshold, self.window_size,self.stride,self.dynamic_length = root, name,s3, n_rois,threshold, window_size,stride,dynamic_length

        self.behavioral_df = pd.read_csv(os.path.join(self.root, 'HCP_behavioral.csv')).set_index('Subject')[['Gender','Age','ListSort_AgeAdj','PMAT24_A_CR']]
            
        mapping = {'22-25':0, '26-30':1,'31-35':2,'36+':3}
        self.behavioral_df['AgeClass'] = self.behavioral_df['Age'].replace(mapping)

        self.target_path = os.path.join(self.root, self.name)
        if not os.path.exists(self.target_path):
            os.makedirs(self.target_path)
        self.BUCKET_NAME = 'hcp-openaccess'
        with open(os.path.join(self.root,"ids.pkl"),'rb') as f:
            self.ids = pickle.load(f)
        self.ids = self.ids[:2]
        roi = fetch_atlas_schaefer_2018(n_rois=self.n_rois,yeo_networks=17, resolution_mm=2)
        atlas = load_img(roi['maps'])
        
        self.volume = atlas.get_fdata()


        self.data_dict = self.construct_dataset()

    def extract_from_3d_no(self, fmri):
        ''' 
        Extract time-series data from a 3d atlas with non-overlapping ROIs.
        
        Inputs:
            path_to_atlas = '/path/to/atlas.nii.gz'
            path_to_fMRI = '/path/to/fmri.nii.gz'
            
        Output:
            returns extracted time series # volumes x # ROIs
        '''
    #     atlas = nib.load(path_to_atlas)
    #     volume = atlas.get_fdata()
    #     subj_scan = nib.load(path_to_fMRI)
    #     fmri = subj_scan.get_fdata()
    #     print(fmri.shape,volume.shape)
        subcor_ts = []
        for i in np.unique(self.volume):
            if i != 0: 
    #             print(i)
                bool_roi = np.zeros(self.volume.shape, dtype=int)
                bool_roi[self.volume == i] = 1
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

    def process_dynamic_fc(self,timeseries,y, sampling_init=None, self_loop=True):
    
        # assumes input shape [minibatch x time x node]
        # output shape [minibatch x time x node x node]
        if self.dynamic_length is None:
            self.dynamic_length = timeseries.shape[0]
            sampling_init = 0
        else:
            if isinstance(sampling_init, int):
                assert timeseries.shape[0] > sampling_init + self.dynamic_length
        assert sampling_init is None or isinstance(sampling_init, int)
        # assert timeseries.ndim==3
        assert self.dynamic_length > self.window_size

        if sampling_init is None:
            sampling_init = randrange(timeseries.shape[0]-self.dynamic_length+1)
        sampling_points = list(range(sampling_init, sampling_init+self.dynamic_length-self.window_size, self.stride))

        dynamic_fc_list = []
        for i in sampling_points:
            slice = timeseries[i:i+self.window_size]
            zd_Ytm = (slice - np.nanmean(slice, axis=0)) / np.nanstd(slice, axis=0, ddof=1)
            conn = ConnectivityMeasure(kind='correlation')
            fc = conn.fit_transform([slice])[0]
            zd_fc = conn.fit_transform([zd_Ytm])[0]
            fc *= np.tri(*fc.shape)
            np.fill_diagonal(fc, 0)
            zd_fc *= 1 - np.tri(*zd_fc.shape, k=-1)
            np.fill_diagonal(zd_fc, 0)
            corr = torch.tensor(fc + zd_fc).to(torch.float)
            A = self.construct_Adj_postive_perc(corr)
            edge_index = A.nonzero().t().to(torch.long)
            # y = 1 if label=="M" else 0
            data = Data(x = corr, edge_index=edge_index, y = y)
            dynamic_fc_list.append(data)
        return dynamic_fc_list

    def get_dynamic_data_object(self,iid):
        try:
            mri_file_path = "HCP_1200/"+iid+"/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz"
            reg_path = "HCP_1200/" + iid + '/MNINonLinear/Results/rfMRI_REST1_LR/Movement_Regressors.txt'
            if not os.path.exists(os.path.join(self.target_path, iid+"_"+os.path.basename(mri_file_path))):
                self.s3.download_file(self.BUCKET_NAME, mri_file_path,os.path.join(self.target_path, iid+"_"+os.path.basename(mri_file_path)))
            if not os.path.exists(os.path.join(self.target_path, iid+"_"+os.path.basename(reg_path))):
                self.s3.download_file(self.BUCKET_NAME, reg_path,os.path.join(self.target_path, iid+"_"+os.path.basename(reg_path)))
            
            image_path_LR = os.path.join(self.target_path, iid+"_"+os.path.basename(mri_file_path))
            reg_path = os.path.join(self.target_path, iid+"_"+os.path.basename(reg_path))
            # regdt_path = os.path.join(target_path, iid+"_"+os.path.basename(regdt_path))
            img = nib.load(image_path_LR)
            # if img.shape[3]<1200:
            #     return None
            regs = np.loadtxt(reg_path)
            # regs_dt = np.loadtxt(regdt_path)
            fmri = img.get_fdata()
            Y = self.extract_from_3d_no(fmri)
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
            gender = self.behavioral_df.loc[iid,'Gender']
            g = 1 if gender=="M" else 0
            labels = torch.tensor([g,self.behavioral_df.loc[iid,'AgeClass'],self.behavioral_df.loc[iid,'ListSort_AgeAdj'],self.behavioral_df.loc[iid,'PMAT24_A_CR']])

            
            dynamic_fc_list = self.process_dynamic_fc(Ytm, labels)
        except:
            return None
        return dynamic_fc_list
    
    
    
    def construct_dataset(self):
        data_dict = {}
        # ids = ids[:2]
        
        for iid in self.ids:
            try:
                dynamic_list  = self.get_dynamic_data_object(iid)
                data_dict[iid] = dynamic_list
            except:
                print("file skipped!", iid)
        torch.save(data_dict, os.path.join(self.target_path,self.name+".pt"))
        print("dataset has been saved successfully!")
        return data_dict
    
def preprocess_dynamic(fmri,regs,n_rois = 100, window_size = 50,stride = 3,dynamic_length=None):
    """
    Preprocess fMRI data using NeuroGraph preprocessing pipeline and construct dynamic functional connectome matrices

    Args:

    fmri (numpy array): fmri image
    regs (numpy array): regressor array
    rois (int): {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}, optional,
    Number of regions of interest. Default=100.
    window_size (int) : the length of the window, default = 50
    stride (int): default: 3
    dynamic_length (int) : length of the timeseries to be considered for dynamic graphs. 
    For memory and compution efficiency, we set dynamic length = 50, default = None,
    if None, consider the whole timeseries object
    
    """
    roi = fetch_atlas_schaefer_2018(n_rois=n_rois,yeo_networks=17, resolution_mm=2)
    atlas = load_img(roi['maps'])
    volume = atlas.get_fdata()
    subcor_ts = []
    for i in np.unique(volume):
        if i != 0: 
            bool_roi = np.zeros(volume.shape, dtype=int)
            bool_roi[volume == i] = 1
            bool_roi = bool_roi.astype(np.bool)
            roi_ts_mean = []
            for t in range(fmri.shape[-1]):
                roi_ts_mean.append(np.mean(fmri[:, :, :, t][bool_roi]))
            subcor_ts.append(np.array(roi_ts_mean))
    Y = np.array(subcor_ts).T
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
    timeseries = Yt - np.matmul(regs,B2) 
    # zscore over axis=0 (time)
    
    sampling_init = None
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
    sampling_points = list(range(sampling_init, sampling_init+dynamic_length-window_size, stride))

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
        dynamic_fc_list.append(corr)
    return dynamic_fc_list