from process_dynamic import *
import boto3
from torch_geometric.data import Batch
import math



# name = 'DynHCPRest'
# root = "data/"
# ngd = NeuroGraphDynamic(root, name)
# dataset = ngd.dataset
# labels = ngd.labels
# print(len(dataset),len(labels))

# ACCESS_KEY = ''
# SECRET_KEY = ''
# s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
# pd_obj = Preprocess_dynamic(root, name,s3)
# dataset = pd_obj.data_dict

# dataset = torch.load(os.path.join(root,name, name+".pt"))

# ### construct gender dataset
# gender_dataset, labels = [],[]
# for k,v in dataset.items():
#     if v is None:
#         continue
#     l = v[0].y
#     gender = int(l[0].item())
#     sub = []
#     for d in v:
# #         print(d)
#         new_data = Data(x = d.x, edge_index = d.edge_index, y = gender)
#         sub.append(new_data)
#     batch = Batch.from_data_list(sub)
#     gender_dataset.append(batch)
#     labels.append(gender)
# print("gender dataset created with {} {} number of instances".format(len(gender_dataset), len(labels)))
# new_dataset = {'labels':labels, "batches":gender_dataset}

# age_dataset, labels = [],[]
# for k,v in dataset.items():
#     if v is None:
#         continue
#     l = v[0].y
#     age = int(l[1].item())
#     if age<=2:  ### Ignoring subjects with age >=36
#         sub = []
#         for d in v:
#     #         print(d)
#             new_data = Data(x = d.x, edge_index = d.edge_index, y = age)
#             sub.append(new_data)
#         batch = Batch.from_data_list(sub)
#         age_dataset.append(batch)
#         labels.append(gender)
# print("Age dataset created with {} {} number of instances".format(len(age_dataset), len(labels)))
# new_dataset = {'labels':labels, "batches":age_dataset}

# wm_dataset, labels = [],[]
# for k,v in dataset.items():
#     if v is None:
#         continue
#     l = v[0].y
#     wm = int(l[2].item())
#     if wm is not None: ## there are some None which should be removed 
#         sub = []
#         for d in v:
#     #         print(d)
#             new_data = Data(x = d.x, edge_index = d.edge_index, y = wm)
#             sub.append(new_data)
#         batch = Batch.from_data_list(sub)
#         wm_dataset.append(batch)
#         labels.append(gender)
# print("Working memory dataset created with {} {} number of instances".format(len(wm_dataset), len(labels)))
# new_dataset = {'labels':labels, "batches":wm_dataset}

# fi_dataset, labels = [],[]
# for k,v in dataset.items():
#     if v is None:
#         continue
#     l = v[0].y
#     fi = int(l[3].item())
#     if not math.isnan(fi): ## there are some None which should be removed 
#         sub = []
#         for d in v:
#     #         print(d)
#             new_data = Data(x = d.x, edge_index = d.edge_index, y = fi)
#             sub.append(new_data)
#         batch = Batch.from_data_list(sub)
#         fi_dataset.append(batch)
#         labels.append(gender)
# print("Fluid intelligence dataset created with {} {} number of instances".format(len(fi_dataset), len(labels)))
# new_dataset = {'labels':labels, "batches":fi_dataset}


# path = "data/raw/"
# img = nib.load(os.path.join(path,"3.nii.gz"))
    
# regs = np.loadtxt(os.path.join(path,"3.txt"))
# fmri = img.get_fdata()
# print(regs.shape, fmri.shape)
# functional_connectome = preprocess_dynamic(fmri,regs,n_rois=100)






