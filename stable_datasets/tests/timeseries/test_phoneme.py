import numpy as np

from stable_datasets.timeseries import Phoneme


def test_phoneme_dataset():
    ds = Phoneme(split="train")

    # Test 1: expected number of training samples
    expected_num_train_samples = 214
    assert len(ds) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, got {len(ds)}."
    )

    # Test 2: expected keys
    sample = ds[0]
    expected_keys = {"series", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: series shape and dtype
    series = sample["series"]
    assert isinstance(series, np.ndarray), f"series should be a numpy array, got {type(series)}."
    assert series.shape == (1024, 1), f"series should have shape (1024, 1), got {series.shape}"
    assert series.dtype == np.float32, f"series should have dtype float32, got {series.dtype}"

    # Test 4: label type and range
    label = sample["label"]
    assert isinstance(label, int), f"label should be an integer, got {type(label)}."
    assert 0 <= label < 39, f"label should be between 0 and 38, got {label}."

    # Test 5: test split size
    ds_test = Phoneme(split="test")
    expected_num_test_samples = 1896
    assert len(ds_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, got {len(ds_test)}."
    )

    print("All Phoneme dataset tests passed successfully!")
