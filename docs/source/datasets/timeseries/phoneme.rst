Phoneme
=======

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Time%20Series%20Classification-blue" alt="Task: Time Series Classification">
   <img src="https://img.shields.io/badge/Classes-39-green" alt="Classes: 39">
   <img src="https://img.shields.io/badge/Shape-1024×1-orange" alt="Shape: 1024x1">
   </p>

Overview
--------

The Phoneme dataset is a univariate time-series classification benchmark derived from
spoken phoneme recordings. Each time series represents a single phoneme utterance with
1024 time steps, and is assigned one of 39 phoneme class labels.

- **Train**: 214 samples
- **Test**: 1896 samples
- **Classes**: 39
- **Time steps**: 1024
- **Dimensions**: 1 (univariate)

Data Structure
--------------

When accessing an example using ``ds[i]``, you will receive a dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Type
     - Description
   * - ``series``
     - ``np.ndarray``
     - Float32 array of shape (1024, 1) — time steps × features
   * - ``label``
     - int
     - Phoneme class label (0–38)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.timeseries import Phoneme

    # First run will download + prepare cache, then return the split as a StableDataset
    ds = Phoneme(split="train")

    # If you omit the split (split=None), you get a StableDatasetDict with all available splits
    ds_all = Phoneme(split=None)

    sample = ds[0]
    print(sample.keys())   # {"series", "label"}
    print(sample["series"].shape)  # (1024, 1)

References
----------

- Official website: https://www.timeseriesclassification.com/description.php?Dataset=Phoneme

Citation
--------

.. code-block:: bibtex

    @inproceedings{hamooni2016dualdomain,
        title={Dual-domain Hierarchical Classification of Phonetic Time Series},
        author={Hamooni, Hossein and Mueen, Abdullah},
        booktitle={2016 IEEE 16th International Conference on Data Mining (ICDM)},
        year={2016},
        organization={IEEE}
    }
