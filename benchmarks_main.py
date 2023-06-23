# from helper import *
# import os

# # construct Rest dataset with all labels: Gender, Age, Working Memory and Fluid Intelligence
# dataset_name = 'HCPRest1000'
# threshold = 10
# root = "data/"
# target_dir = "data/rest/"
# dataset =  Brain_connectome_Rest(os.path.join(root+dataset_name),dataset_name,"corr", threshold,target_dir)
#  # construct Age dataset from HCPRest1000
# age_data = "HCPRestSparseAge1000"
# age_dataset = Age_Datasets(root+age_data,age_data,dataset)

# wm_data = "HCPRestWM1000"
# wm_dataset = WM_Datasets(os.path.join(root+wm_data),wm_data,dataset)

# FL_data = "HCPRestFI1000"
# fl_dataset = FI_Datasets(os.path.join(root,+FL_data),FL_data,dataset)



# # construct Task datase
# dataset_name = 'HCPTask100'
# threshold = 10
# root = "data/"
# target_dir = "data/task/"
# dataset =  Brain_connectome_Task(os.path.join(root+dataset_name),dataset_name,"corr", threshold,target_dir)



# seed = 123
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# dataset_name = 'DynamicHCPRestMedium100'
# target_dir = "data/rest/"
# threshold = 10
# window_stride = 3
# window_size = 50
# dynamic_length = 150
# dataset = Dynamic_Brain_connectome_Rest(os.path.join(root,+dataset_name),dataset_name,"corr", threshold,window_size, window_stride, dynamic_length,target_dir)

# ###loading dynamic dataset
# # dyn_dataset = torch.load(os.path.join(os.path.join(root,+dataset_name),dataset_name,"processed", dataset_name+".pt"))



# #Dynamic HCPTask construction
# dataset_name = 'DynamicHCPTaskMedium100'
# threshold = 10
# window_stride = 3
# window_size = 50
# dynamic_length = 150
# target_dir = "data/task/"
# dataset = Dynamic_Brain_connectome_Task(os.path.join(root,+dataset_name),dataset_name, threshold,window_size, window_stride, dynamic_length,target_dir)


# ## DYNAMIC DATASETS CAN'T BE LOADED DIRECTOLY WITH PYG BECAUSE THEY DON'T FOLLOW PYG INMEMORYDATASET CLASS PROTOCOL. THEY COULD BE LOADED AS FOLLOW
# #

# dataset = torch.load(os.path.join(os.path.join(root,+dataset_name),dataset_name,"processed", dataset_name+".pt"))  
# gender_dataset, labels = [],[]
# for obj in dataset:
#     if obj is None:
#         continue
#     data = obj['data']
#     l = data[0].y
#     gender = int(l[0].item())
#     sub = []
#     for d in data:
# #         print(d)
#         new_data = Data(x = d.x, edge_index = d.edge_index, y = gender)
#         sub.append(new_data)
#     batch = Batch.from_data_list(sub) ## CONSTRUCT A BATCH FROM THE SNAPSHOTS
#     gender_dataset.append(batch)
#     labels.append(gender)
# print("gender dataset created with {} {} number of instances".format(len(gender_dataset), len(labels)))
# new_dataset = {'labels':labels, "batches":gender_dataset}

# ## SAVE THE ABOVE DATASET
# data_name = "DynamicHCPRestMediumGender100"
# path = "data/"+data_name+"/processed/"
# os.makedirs(path, exist_ok=True)
# torch.save(new_dataset,os.path.join(path,data_name+".pt"))

# wm_dataset, labels = [],[]
# for obj in dataset:
#     if obj is None:
#         continue
#     data = obj['data']
#     l = data[0].y
#     wm = l[2].item()
#     if wm is not None:
#         sub = []
#         for d in data:
#     #         print(d)
#             new_data = Data(x = d.x, edge_index = d.edge_index, y = wm)
#             sub.append(new_data)
#         batch = Batch.from_data_list(sub)
#         wm_dataset.append(batch)
#         labels.append(wm)
# print("gender dataset created with {} {} number of instances".format(len(wm_dataset), len(labels)))
# new_dataset = {'labels':labels, "batches":wm_dataset}

# data_name = "DynamicHCPRestMediumWM100"
# path = "data/"+data_name+"/processed/"
# os.makedirs(path, exist_ok=True)
# torch.save(new_dataset,os.path.join(path,data_name+".pt"))

# fl_dataset, labels = [],[]
# for obj in dataset:
#     if obj is None:
#         continue
#     data = obj['data']
#     l = data[0].y
#     fl = l[3].item()
# #     print(fl, l)
#     if not math.isnan(fl):
#         sub = []
#         for d in data:
#     #         print(d)
#             new_data = Data(x = d.x, edge_index = d.edge_index, y = fl)
#             sub.append(new_data)
#         batch = Batch.from_data_list(sub)
#         fl_dataset.append(batch)
#         labels.append(fl)
# print("FL dataset created with {} {} number of instances".format(len(fl_dataset), len(labels)))
# new_dataset = {'labels':labels, "batches":fl_dataset}

# data_name = "DynamicHCPRestMediumFI100"
# path = "data/"+data_name+"/processed/"
# os.makedirs(path, exist_ok=True)
# torch.save(new_dataset,os.path.join(path,data_name+".pt"))
