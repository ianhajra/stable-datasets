import zipfile
from pathlib import Path

import numpy as np
from scipy.io import arff

from stable_datasets.schema import Array3D, ClassLabel, DatasetInfo, Features, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, _default_dest_folder, bulk_download


_LABELS = [str(i) for i in range(1, 40)]
_NUM_TIMESTEPS = 1024
_NUM_DIMS = 1


class Phoneme(BaseDatasetBuilder):
    """Phoneme dataset — univariate time-series classification.

    214 training and 1896 test utterances across 39 phoneme classes,
    each represented as a 1024-length univariate time series.
    """

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "https://www.timeseriesclassification.com/description.php?Dataset=Phoneme",
        "assets": {
            "train": "https://www.timeseriesclassification.com/aeon-toolkit/Phoneme.zip",
            "test": "https://www.timeseriesclassification.com/aeon-toolkit/Phoneme.zip",
        },
        "citation": """@inproceedings{hamooni2016dualdomain,
            title={Dual-domain Hierarchical Classification of Phonetic Time Series},
            author={Hamooni, Hossein and Mueen, Abdullah},
            booktitle={2016 IEEE 16th International Conference on Data Mining (ICDM)},
            year={2016},
            organization={IEEE}}""",
    }

    def _info(self):
        return DatasetInfo(
            description=(
                "The Phoneme dataset contains 214 training and 1896 test univariate time series "
                "across 39 phoneme classes. Each series has 1024 time steps."
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

        extract_dir = Path(zip_path).parent / "Phoneme_extracted"
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"arff_path": extract_dir / "Phoneme_TRAIN.arff"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"arff_path": extract_dir / "Phoneme_TEST.arff"},
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
