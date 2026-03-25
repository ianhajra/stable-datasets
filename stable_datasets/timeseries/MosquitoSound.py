import zipfile
from pathlib import Path

import numpy as np
from scipy.io import arff

from stable_datasets.schema import Array3D, ClassLabel, DatasetInfo, Features, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder, bulk_download


_LABELS = ["0", "1", "2", "3", "4", "5"]
_NUM_TIMESTEPS = 3750
_NUM_DIMS = 1


class MosquitoSound(BaseDatasetBuilder):
    """MosquitoSound dataset — univariate time-series classification.

    279,566 mosquito wingbeat recordings across 6 species, each represented
    as a univariate time series of 3750 time steps.
    """

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "http://www.timeseriesclassification.com/description.php?Dataset=MosquitoSound",
        "assets": {
            "train": "https://www.timeseriesclassification.com/aeon-toolkit/MosquitoSound.zip",
            "test": "https://www.timeseriesclassification.com/aeon-toolkit/MosquitoSound.zip",
        },
        "citation": """@article{potamitis2016large,
            title={Large aperture optoelectronic devices to record and time-stamp insects wingbeats},
            author={Potamitis, Ilyas and Rigakis, Iraklis},
            journal={IEEE Sensors Journal},
            volume={16},
            number={15},
            pages={6053--6061},
            year={2016},
            publisher={IEEE}}""",
    }

    def _info(self):
        return DatasetInfo(
            description=(
                "The MosquitoSound dataset contains 279,566 mosquito wingbeat recordings across 6 species, "
                "split equally into 139,883 train and 139,883 test samples. Each time series represents "
                "the change in amplitude of an infrared light occluded by a flying mosquito's wings, "
                "sampled at 6,000 Hz, with 3,750 time steps per recording."
            ),
            features=Features(
                {
                    "series": Array3D(shape=(_NUM_TIMESTEPS, _NUM_DIMS), dtype="float32"),
                    "label": ClassLabel(names=_LABELS),
                }
            ),
            supervised_keys=("series", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        source = self._source()
        self._validate_source(source)

        download_dir = getattr(self, "_raw_download_dir", None) or _default_dest_folder()
        [zip_path] = bulk_download(
            [source["assets"]["train"]],
            dest_folder=download_dir,
        )

        extract_dir = Path(zip_path).parent / "MosquitoSound_extracted"
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        ts_dir = extract_dir / "MosquitoSound"
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"arff_path": ts_dir / "MosquitoSound_TRAIN.arff"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"arff_path": ts_dir / "MosquitoSound_TEST.arff"},
            ),
        ]

    def _generate_examples(self, arff_path):
        data, meta = arff.loadarff(arff_path)
        names = meta.names()
        feature_names = names[:-1]
        label_name = names[-1]

        for idx, row in enumerate(data):
            series = np.array([row[n] for n in feature_names], dtype="float32").reshape(_NUM_TIMESTEPS, _NUM_DIMS)
            label = str(row[label_name].decode() if isinstance(row[label_name], bytes) else row[label_name])
            yield idx, {"series": series, "label": label}
