Datasets
========

This section provides detailed documentation for all datasets available in stable-datasets.

Overview
--------

stable-datasets provides easy access to a wide variety of datasets for machine learning research, with a focus on stability and reproducibility. Each dataset page includes:

- **Example Samples**: Visual examples or data snippets from the dataset
- **Dataset Details**: Number of classes, target types, and data specifications
- **Data Structure**: Keys and data types returned when accessing the dataset
- **Usage Examples**: Code snippets showing how to load and use the dataset
- **Related Datasets**: Links to similar or derived datasets
- **Citation**: The original paper to cite when using the dataset

Getting Started
---------------

All datasets can be loaded using the same consistent API:

.. code-block:: python

    from stable_datasets.images.<dataset_module> import <DatasetClass>

    # First run will download + prepare cache, then return the split as a HF Dataset
    ds = <DatasetClass>(split="train")

    # If you omit the split (split=None), you get a DatasetDict with all available splits
    ds_all = <DatasetClass>(split=None)

    # Access individual examples
    sample = ds[0]
    print(sample.keys())  # e.g., {"image", "label"}

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

Available Datasets
------------------

.. toctree::
   :maxdepth: 1
   :caption: Image Classification Datasets

   images/arabic_characters
   images/arabic_digits
   images/cifar10
   images/cifar100
   images/cifar10_c
   images/cifar100_c
   images/cars196
   images/dtd
   images/fashion_mnist
   images/k_mnist
   images/medmnist
   images/awa2
   images/beans
   images/stl10
   images/tiny_imagenet
   images/tiny_imagenet_c
   images/e_mnist
   images/fgvc_aircraft
   images/flowers102
   images/food101
   images/cub200
   images/country211
   images/galaxy10
   images/hasy_v2
   images/face_pointing
   images/rock_paper_scissor
   images/linnaeus5

.. toctree::
   :maxdepth: 1
   :caption: Time Series Datasets

   timeseries/japanese_vowels
   timeseries/phoneme

.. note::
   Documentation is being added progressively, as datasets are ready for usage. Please only use datasets found in the documentation.
