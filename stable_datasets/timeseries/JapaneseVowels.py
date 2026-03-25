import zipfile
from pathlib import Path

import numpy as np

from stable_datasets.schema import Array3D, ClassLabel, DatasetInfo, Features, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import (
    BaseDatasetBuilder,
    _default_dest_folder,
    bulk_download,
    load_from_tsfile_to_dataframe,
)


_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
_NUM_TIMESTEPS = 29
_NUM_DIMS = 12


class JapaneseVowels(BaseDatasetBuilder):
    """Japanese Vowels dataset — multivariate time-series classification.

    270 training and 370 test utterances from 9 male speakers, each
    represented as a 29×12 array of LPC-derived features.
    """

    VERSION = Version("1.0.0")

    SOURCE = {
        "homepage": "https://www.timeseriesclassification.com/description.php?Dataset=JapaneseVowels",
        "assets": {
            "train": "https://www.timeseriesclassification.com/aeon-toolkit/JapaneseVowels.zip",
            "test": "https://www.timeseriesclassification.com/aeon-toolkit/JapaneseVowels.zip",
        },
        "citation": """@inproceedings{kudo1999multidimensional,
            title={Multidimensional curve classification using passing-through regions},
            author={Kudo, Mineichi and Toyama, Jun and Shimbo, Masaru},
            booktitle={Pattern Recognition Letters},
            volume={20},
            number={11-13},
            pages={1103--1111},
            year={1999}}""",
    }

    def _info(self):
        return DatasetInfo(
            description=(
                "The Japanese Vowels dataset contains 270 training and 370 test utterances "
                "from 9 male speakers. Each utterance is represented as a multivariate "
                "time series of shape (29, 12) — 29 time steps × 12 LPC-derived features."
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
        # Both splits share one zip — bulk_download deduplicates.
        [zip_path] = bulk_download(
            [source["assets"]["train"]],
            dest_folder=download_dir,
        )

        extract_dir = Path(zip_path).parent / "JapaneseVowels_extracted"
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"ts_path": extract_dir / "JapaneseVowels_TRAIN.ts"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"ts_path": extract_dir / "JapaneseVowels_TEST.ts"},
            ),
        ]

    def _generate_examples(self, ts_path):
        X, y = load_from_tsfile_to_dataframe(ts_path)
        dims = []
        for col in X.columns:
            dims.append(np.stack(list(X[col].map(lambda x: x.reindex(range(_NUM_TIMESTEPS))))))
        X_arr = np.nan_to_num(np.stack(dims, -1).astype("float32"), nan=0.0)

        for idx, (series, label) in enumerate(zip(X_arr, y)):
            yield idx, {"series": series, "label": str(label)}
