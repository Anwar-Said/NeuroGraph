import numpy as np
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import load_img
from nilearn.connectome import ConnectivityMeasure
from scipy.stats import zscore
import torch
from torch_geometric.data import Data

def preprocess(fmri,regs, n_rois =1000):
    """
    Preprocess fMRI data using NeuroGraph preprocessing pipeline

    Args:

    fmri (numpy array): fmri image
    regs (numpy array): regressor array
    rois (int): {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}, optional,
    Number of regions of interest. Default=1000.
    
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
    corr = fc + zd_fc
    return corr
def construct_Adj(corr, threshold=5):
    """
    create adjacency matrix from functional connectome matrix
    
    Args:

    corr (n x n numpy matrix): functional connectome matrix

    Threshold (int (1- 100)): threshold for controling graph density. 

    the more higher the threshold, the more denser the graph. default: 5 
    
    """
    corr_matrix_copy = corr.copy()
    threshold = np.percentile(corr_matrix_copy[corr_matrix_copy > 0], 100 - threshold)
    corr_matrix_copy[corr_matrix_copy < threshold] = 0
    corr_matrix_copy[corr_matrix_copy >= threshold] = 1
    return corr_matrix_copy

def construct_data(corr, label, threshold = 5):
    """
    create pyg data object from functional connectome matrix. We use correlation as node features
    Args:

    corr (n x n numpy matrix): functional connectome matrix

    Threshold (int (1- 100)): threshold for controling graph density. 

    the more higher the threshold, the more denser the graph. default: 5 

    
    """
    A = corr.copy()
    threshold = np.percentile(A[A > 0], 100 - threshold)
    A[A < threshold] = 0
    A[A >= threshold] = 1
    edge_index = A.nonzero().t().to(torch.long)
    data = Data(x = corr, edge_index=edge_index, y = label)
    return data

    