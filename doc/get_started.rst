Introduction by Example
================================

We will briefly introduce the fundamental concepts of NeuroGraph through self-contained examples. We closely follow the data representation format of `PyG <https://pytorch-geometric.readthedocs.io/en/latest/>`_. Therefore, interested readers are referred to the `PyG <https://pytorch-geometric.readthedocs.io/en/latest/>`_ documentation for an introduction to the graph machine learning and PyG's data representation formats.   


Loading Benchmark datasets
----------------------------------

NeuroGraph provides two classes for loading static and dynamic benchmark datastes. 

Loading Static Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NeuroGraph utilizes the `PyG` `InMemoryDataset` class to facilitate the loading of datasets. this allows an easy-to-use interface for applying graph machine learning pipelines. For example, the `HCPGender` benchmark can be loaded as follows:


.. code-block:: python
    :linenos:

    from NeuroGraph.datasets import NeuroGraphDataset
    dataset = NeuroGraphDataset(root="data/", name= "HCPGender")
    print(dataset.num_classes)
    print(dataset.num_features)


Loading Dynamic Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To efficiently store and utilize the dynamic datasets in `PyG`` Batch format, we provide the corresponding functionality. Here is an example of loading the `DynHCPGender` dataset:


.. code-block:: python
    :linenos:
    from NeuroGraph.datasets import NeuroGraphDynamic
    data_obj = NeuroGraphDynamic(root="data/", name= "DynHCPGender")
    dataset = data_obj.dataset
    labels = data_obj.labels
    print(len(dataset), len(labels))

The dataset is a list of dynamic graphs represented in the `PyG` batch format, making it compatible with graph machine learning pipelines.


Preprocessing Examples
====================================

To bridge the gap betwee NeuroGraph and graph machine learning domains, NeuroGraph offers tools to easily preprocess and construct graph-based neuroimaging datasets. Here, we demonstrate how to preprocess your own data to construct functional connectomes and generate corresponding graphs-based representations.


.. code-block:: python
    :linenos:
    from NeuroGraph import utils
    fc = utils.preprocess(fmri, regs, n_rois= 1000) # fmri and regs could be numpy arrays

The corresponding `Adjacency matrix` and `PyG` data objects can be created from the functional_connectome as follows. 

.. code-block:: python
    :linenos:

    from NeuroGraph import utils
    adj = utils.construct_adj(fc, threshold= 5) # construct the adjacency matrix
    data = utils.construct_data(fc, label= 1,threshold = 5) # construct PyG data object

We use correlation as node features while constructing data object from functional connectome. 

The following is the source code for processing one fMRI scan with corresponding regressor using our preprocessing pipeline.

.. code-block:: python
    :linenos:

    from NeuroGraph import utils
    import numpy as np
    from nilearn.image import load_img
    img = load_img("data/raw/1.nii.gz") # 1.nii.gz is fMRI scan
    regs = np.loadtxt("data/raw/1.txt") # 1.txt is the movement regressor
    fmri = img.get_fdata()
    fc = utils.preprocess(fmri, regs, n_rois= 100)
    adj = utils.construct_adj(fc, threshold= 5) # construct the adjacency matrix
    data = utils.construct_data(fc, label = 1,threshold = 5) # construct torch Data object
    

Our preprocessing pipeline consists of five steps and can also be applied seperately in steps.

.. code-block:: python
    :linenos:

    from NeuroGraph import utils
    import numpy as np
    from nilearn.image import load_img

    img = load_img("data/raw/1.nii.gz")
    regs = np.loadtxt("data/raw/1.txt")
    fmri = img.get_fdata()
    parcells = utils.parcellation(fmri,n_rois = 100) ## this uses schaefer atlas by default
    Y = utils.remove_drifts(parcells)
    Y = utils.regress_head_motions(Y,regs)
    fc = utils.construct_corr(Y)
    adj = utils.construct_adj(fc, threshold= 5) # construct the adjacency matrix
    data = utils.construct_data(fc, label = 1,threshold = 5)
    


Preprocessing Human Connectome Project (HCP1200) Dataset
==============================================================================

NeuroGraph utilizes the HCP1200 dataset as a primary data source for exploring the dataset generation search space and constructing benchmarks. The HCP1200 dataset can be accessed from the `HCP website <https://www.humanconnectome.org/study/hcp-young-adult>`_ by accepting the data usage terms. Additionally, the dataset is also available on an AWS S3 bucket, which can be accessed once authorization has been obtained from HCP. In this section, we provide various functions that allow you to crawl and preprocess the HCP datasets, enabling the construction of graph-based neuroimaging datasets. These functions streamline the process of obtaining and preparing the data for further analysis and modeling.


Download and preprocess static datasets
---------------------------------------------------

.. code-block:: python
    :linenos:

    from NeuroGraph.preprocess import Brain_Connectome_Rest_Download
    import boto3

    root = "data/"
    name = "HCPGender"
    threshold = 5
    path_to_data = "data/raw/HCPGender"  # store the raw downloaded scans
    n_rois = 100
    n_jobs = 5 # this script runs in parallel and requires the number of jobs is an input

    ACCESS_KEY = ''  # your connectomeDB credentials
    SECRET_KEY = ''
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    # this function requires both HCP_behavioral.csv and ids.pkl files under the root directory. Both files have been provided and can be found under the data directory
    rest_dataset = Brain_Connectome_Rest_Download(root,name,n_rois, threshold,path_to_data,n_jobs,s3)


The provided function facilitates the download of data from the AWS S3 bucket, performs preprocessing steps, and generates a graph-based dataset. It is important to note that the `rest_dataset` used in this function consists of four labels: gender, age, working memory, and fluid intelligence. To create separate datasets based on these labels, the following functionalities can be used. 

.. code-block:: python
    :linenos:

    from NeuroGraph import preprocess

    rest_dataset = preprocess.Brain_Connectome_Rest_Download(root,name,n_rois, threshold,path_to_data,n_jobs,s3)
    gender_dataset = preprocess.Gender_Dataset(root, "HCPGender",rest_dataset) 
    age_dataset = preprocess.Age_Dataset(root, "HCPAge",rest_dataset)
    wm_datast = preprocess.WM_Dataset(root, "HCPWM",rest_dataset)
    fi_datast = preprocess.FI_Dataset(root, "HCPFI",rest_dataset)

To construct the State dataset, the following functionalities can be used. 

.. code-block:: python
    :linenos:

    from NeuroGraph import preprocess

    state_dataset = preprocess.Brain_Connectome_State_Download(root, dataset_name,rois, threshold,path_to_data,n_jobs,s3)

If you have the data locally, then the following functionalities can be used to preprocess the data. 


.. code-block:: python
    :linenos:

    from NeuroGraph import preprocess

    rest_dataset = preprocess.Brain_Connectome_Rest(root, name, n_rois, threshold, path_to_data, n_jobs)

Similarly, for constructing the State dataset, the following function can be used. 

.. code-block:: python
    :linenos:

    from NeuroGraph import preprocess

    state_dataset = preprocess.Brain_Connectome_State(root, name, n_rois, threshold, path_to_data, n_jobs)


Download and preprocess dynamic datasets
---------------------------------------------------

We also offer similar functionalities for constructing dynamic datasets. You can create a dynamic REST dataset from the data stored locally as follows. 



.. code-block:: python
    :linenos:

    from NeuroGraph import preprocess

    ngd = Dyn_Prep(fmri, regs, n_rois=100, window_size=50, stride=3, dynamic_length=None)
    dataset = ngd.dataset
    labels = ngd.labels
    print(len(dataset),len(labels))

Here the dataset is a list containing dynamic graphs in the form of PyG Batch, which can be easily fed into graph machine learning pipelines. The following examples demonstrate how a dynamic REST dataset can be downloaded and preprocessed on the fly. 

.. code-block:: python
    :linenos:

    from NeuroGraph import preprocess

    dyn_obj = preporcess.Dyn_Down_Prep(root, name,s3,n_rois = 100, threshold = 10, window_size = 50,stride == 3, dynamic_length=150)
    dataset = dyn_obj.data_dict

Dyn_Down_Prep class downloads and preprocess the rest dataset and provides a dictionary that contains a list of dynamic graphs against each id. The dataset can be further prprocessed as follows to construct each benchmark. 

.. code-block:: python
    :linenos:

    from NeuroGraph import preprocess

    dyn_obj = preporcess.Dyn_Down_Prep(root, name,s3,n_rois = 100, threshold = 10, window_size = 50,stride == 3, dynamic_length=150)
    dataset = dyn_obj.data_dict
    gender_dataset, labels = [],[]
    for k,v in dataset.items():
        if v is None:
            continue
        l = v[0].y
        gender = int(l[0].item())
        sub = []
        for d in v:
            new_data = Data(x = d.x, edge_index = d.edge_index, y = gender)
            sub.append(new_data)
        batch = Batch.from_data_list(sub)
        gender_dataset.append(batch)
        labels.append(gender)
    print("gender dataset created with {} {} number of instances".format(len(gender_dataset), len(labels)))
    new_dataset = {'labels':labels, "batches":gender_dataset}

    age_dataset, labels = [],[]
    for k,v in dataset.items():
        if v is None:
            continue
        l = v[0].y
        age = int(l[1].item())
        if age <=2:  ### Ignoring subjects with age >=36
            sub = []
            for d in v:
                new_data = Data(x = d.x, edge_index = d.edge_index, y = age)
                sub.append(new_data)
            batch = Batch.from_data_list(sub)
            age_dataset.append(batch)
            labels.append(gender)
    print("Age dataset created with {} {} number of instances".format(len(age_dataset), len(labels)))
    new_dataset = {'labels':labels, "batches":age_dataset}

    wm_dataset, labels = [],[]
    for k,v in dataset.items():
        if v is None:
            continue
        l = v[0].y
        wm = int(l[2].item())
        if wm is not None: ## there are some None which should be removed 
            sub = []
            for d in v:
        #         print(d)
                new_data = Data(x = d.x, edge_index = d.edge_index, y = wm)
                sub.append(new_data)
            batch = Batch.from_data_list(sub)
            wm_dataset.append(batch)
            labels.append(gender)
    print("Working memory dataset created with {} {} number of instances".format(len(wm_dataset), len(labels)))
    new_dataset = {'labels':labels, "batches":wm_dataset}

    fi_dataset, labels = [],[]
    for k,v in dataset.items():
        if v is None:
            continue
        l = v[0].y
        fi = int(l[3].item())
        if not math.isnan(fi): ## there are some None which should be removed 
            sub = []
            for d in v:
        #         print(d)
                new_data = Data(x = d.x, edge_index = d.edge_index, y = fi)
                sub.append(new_data)
            batch = Batch.from_data_list(sub)
            fi_dataset.append(batch)
            labels.append(gender)
    print("Fluid intelligence dataset created with {} {} number of instances".format(len(fi_dataset), len(labels)))
    new_dataset = {'labels':labels, "batches":fi_dataset}
