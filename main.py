import torch
import sys
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from neurograph import *
from preprocess_fmri import *
from preprocess import *
from torch_geometric.datasets import TUDataset
import boto3


def main():
    """Load preprocessed datasets"""
    # name = "HCPGender"
    # root = "data/"
    # dataset = NeuroGraphStatic(root, name)
    # print(dataset.num_classes)
    # print(dataset.num_features)
    # print(dataset.num_node_features)


    """Load and preprocess datasets from Human Connectome DB
    Args:

    
    # """
    
    
    
    # root ="data/"
    # dataset_name = "HCPRest"
    # rois = 100
    # threshold = 10
    # n_jobs = 4
    # path_to_data = "/data/rest/"    
    # rest_dataset = Brain_connectome_Rest(root, dataset_name,rois, threshold,path_to_data,n_jobs)
    
    
    # age_dataset = Age_Dataset(root, "HCPAge",rest_dataset)
    # for d in age_dataset:
    #     print(d, d.y.item())
    # print(age_dataset.num_classes)
    # gender_dataset = Gender_Dataset(root, "HCPGender",rest_dataset)
    # for d in gender_dataset:
    #     print(d, d.y.item())
    # print(gender_dataset.num_classes) 
    # WM_dataset = WM_Dataset(root, "HCPWM",rest_dataset)
    # for d in WM_dataset:
    #     print(d, d.y.item())
        
    # FI_dataset = FI_Dataset(root, "HCPFI",rest_dataset)
    # # print(FI_dataset.num_classes)
    # for d in FI_dataset:
    #     print(d, d.y.item())

    # root ="data/"
    # dataset_name = "HCPActivity"
    # rois = 100
    # threshold = 10
    # n_jobs = 4
    # path_to_data = "/data/activity/"    
    # activity_dataset = Brain_connectome_Activity(root, dataset_name,rois, threshold,path_to_data,n_jobs)
    # for d in activity_dataset:
    #     print(d, d.y.item())

    # ACCESS_KEY = 'AKIAXO65CT57HFCOMHVK'
    # SECRET_KEY = '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'
    # s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    # root ="data/"
    # dataset_name = "HCPRestDownload"
    # rois = 100
    # threshold = 10
    # n_jobs = 1
    # path_to_data = "data/rest/"    
    # rest_dataset = Brain_connectome_Rest_Download(root, dataset_name,rois, threshold,path_to_data,n_jobs,s3)
    # print(len(rest_dataset))
    # root ="data/"
    # dataset_name = "HCPActivityDownload"
    # rois = 100
    # threshold = 10
    # n_jobs = 1
    # path_to_data = "data/task/"    
    # activity_dataset = Brain_connectome_Activity_Download(root, dataset_name,rois, threshold,path_to_data,n_jobs,s3)
    # print(len(activity_dataset))


    ### Use NeuroGraph preprocessing pipleine to construct functional connectome matrices
    
    # path = "data/raw/"
    # img = nib.load(os.path.join(path,"3.nii.gz"))
            
    # regs = np.loadtxt(os.path.join(path,"3.txt"))
    # fmri = img.get_fdata()
    # print(regs.shape, fmri.shape)
    functional_connectome = preprocess(fmri,regs,n_rois = 100)
    # print(type(functional_connectome), functional_connectome.shape)

    # np.save("data/fc.npy",functional_connectome)
    functional_connectome = np.load("data/fc.npy")
    ### CONSTRUCT GRAPH IN PYG FORMAT FROM FUNCTIONAL CONNECTOME MATRIX
    threshold = 5
    adj = construct_Adj(functional_connectome, threshold)
    print(adj.shape)

    data = construct_data(functional_connectome,1,threshold)
    print(data)

if __name__=="__main__":
    main()

