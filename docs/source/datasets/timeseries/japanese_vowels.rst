Japanese Vowels
===============

.. raw:: html

   <p style="display: flex; gap: 10px;">
   <img src="https://img.shields.io/badge/Task-Time%20Series%20Classification-blue" alt="Task: Time Series Classification">
   <img src="https://img.shields.io/badge/Classes-9-green" alt="Classes: 9">
   <img src="https://img.shields.io/badge/Shape-29×12-orange" alt="Shape: 29x12">
   </p>

Overview
--------

The Japanese Vowels dataset is a multivariate time-series classification benchmark collected
from 9 male speakers each uttering two Japanese vowels /ae/ successively. Each utterance is
encoded as a sequence of 12-dimensional LPC-derived feature vectors.

- **Train**: 270 utterances (30 per speaker)
- **Test**: 370 utterances
- **Classes**: 9 (one per speaker)
- **Time steps**: up to 29 (padded/truncated to a fixed length of 29)
- **Dimensions**: 12 (LPC cepstral coefficients)

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
     - Float32 array of shape (29, 12) — time steps × features
   * - ``label``
     - int
     - Speaker identity label (0–8)

Usage Example
-------------

**Basic Usage**

.. code-block:: python

    from stable_datasets.timeseries import JapaneseVowels

    # First run will download + prepare cache, then return the split as a StableDataset
    ds = JapaneseVowels(split="train")

    # If you omit the split (split=None), you get a StableDatasetDict with all available splits
    ds_all = JapaneseVowels(split=None)

    sample = ds[0]
    print(sample.keys())   # {"series", "label"}
    print(sample["series"].shape)  # (29, 12)

    # Optional: make it PyTorch-friendly
    ds_torch = ds.with_format("torch")

References
----------

- Official website: https://www.timeseriesclassification.com/description.php?Dataset=JapaneseVowels

Citation
--------

.. code-block:: bibtex

    @inproceedings{kudo1999multidimensional,
        title={Multidimensional curve classification using passing-through regions},
        author={Kudo, Mineichi and Toyama, Jun and Shimbo, Masaru},
        booktitle={Pattern Recognition Letters},
        volume={20},
        number={11-13},
        pages={1103--1111},
        year={1999}
    }
