MosquitoSound
=============

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Time%20Series%20Classification-blue" alt="Task: Time Series Classification">
   <img src="https://img.shields.io/badge/Type-Audio-purple" alt="Type: Audio">
   <img src="https://img.shields.io/badge/Classes-6-green" alt="Classes: 6">
   <img src="https://img.shields.io/badge/Shape-3750×1-orange" alt="Shape: 3750x1">
   </p>

Overview
--------

The MosquitoSound dataset contains recordings of mosquito wingbeats captured using large-aperture
optoelectronic sensors. Each time series represents the change in amplitude of an infrared light
as it is occluded by the wings of a flying mosquito, sampled at 6,000 Hz. The database contains
wav recordings from six insectary boxes, each containing only one mosquito species of both sexes.
The train/test split was created through a random partition.

- **Train**: 139,883 samples
- **Test**: 139,883 samples
- **Classes**: 6
- **Time steps**: 3,750
- **Dimensions**: 1 (univariate)

.. list-table:: Class labels
   :header-rows: 1
   :widths: 10 30 20

   * - Label
     - Species
     - Count
   * - 0
     - *Ae. aegypti*
     - 85,553
   * - 1
     - *Ae. albopictus*
     - 20,231
   * - 2
     - *An. arabiensis*
     - 19,297
   * - 3
     - *An. gambiae*
     - 49,471
   * - 4
     - *Cu. pipiens*
     - 30,415
   * - 5
     - *Cu. quinquefasciatus*
     - 74,599

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
     - Float32 array of shape (3750, 1) — time steps × features
   * - ``label``
     - int
     - Species label (0–5)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.timeseries import MosquitoSound

    # First run will download + prepare cache, then return the split as a StableDataset
    ds = MosquitoSound(split="train")

    # If you omit the split (split=None), you get a StableDatasetDict with all available splits
    ds_all = MosquitoSound(split=None)

    sample = ds[0]
    print(sample.keys())   # {"series", "label"}
    print(sample["series"].shape)  # (3750, 1)

References
----------

- Official website: http://www.timeseriesclassification.com/description.php?Dataset=MosquitoSound

Citation
--------

.. code-block:: bibtex

    @article{potamitis2016large,
        title={Large aperture optoelectronic devices to record and time-stamp insects wingbeats},
        author={Potamitis, Ilyas and Rigakis, Iraklis},
        journal={IEEE Sensors Journal},
        volume={16},
        number={15},
        pages={6053--6061},
        year={2016},
        publisher={IEEE}
    }
