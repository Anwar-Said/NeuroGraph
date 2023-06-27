Intorduction by Example
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

    from NeuroGraph.datasets import NeuroGraphStatic


    dataset = NeuroGraphStatic(root="data/", name= "HCPGender")

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



