from dataclasses import dataclass, asdict, fields
from pathlib import Path
import pandas as pd

CURRENT_FILE_PATH = Path(__file__)
PROJECT_DIR = CURRENT_FILE_PATH.parents[2]
DATASET_FOLDER = PROJECT_DIR / "PlantDoc-Dataset"
METADATA = DATASET_FOLDER / "metadata.csv"

SPLITS = ("train", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class ClassInfo:
    label_index: int
    class_name: str
    train_count: int = 0
    test_count: int = 0


class DatasetMetadata:
    """The classes in the dataset, their label indices and image counts per split."""

    def __init__(self, classes: list[ClassInfo]):
        self.classes = sorted(classes, key=lambda c: c.label_index)
        self._by_name = {c.class_name: c for c in self.classes}
        self._by_index = {c.label_index: c for c in self.classes}

    def __len__(self):
        return len(self.classes)

    def __iter__(self):
        return iter(self.classes)

    def by_name(self, class_name: str) -> ClassInfo:
        return self._by_name[class_name]

    def by_index(self, label_index: int) -> ClassInfo:
        return self._by_index[label_index]

    @classmethod
    def from_dataset(cls, dataset_folder: Path = DATASET_FOLDER, previous: "DatasetMetadata | None" = None):
        """Scan the train/test split folders and count the images of each class.

        Label indices from `previous` are kept, so a trained model's outputs still
        map to the same classes; new classes get the next free indices.
        """
        counts: dict[str, dict[str, int]] = {}
        for split in SPLITS:
            for class_dir in (dataset_folder / split).iterdir():
                if not class_dir.is_dir():
                    continue
                n_images = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
                counts.setdefault(class_dir.name, dict.fromkeys(SPLITS, 0))[split] = n_images

        indices = {c.class_name: c.label_index for c in previous} if previous else {}
        next_index = max(indices.values(), default=-1) + 1
        for class_name in sorted(counts):
            if class_name not in indices:
                indices[class_name] = next_index
                next_index += 1

        return cls([
            ClassInfo(indices[name], name, split_counts["train"], split_counts["test"])
            for name, split_counts in counts.items()
        ])

    @classmethod
    def load(cls, path: Path = METADATA):
        df = pd.read_csv(path)
        return cls([ClassInfo(**row) for row in df.to_dict("records")])

    def save(self, path: Path = METADATA):
        columns = [f.name for f in fields(ClassInfo)]
        pd.DataFrame([asdict(c) for c in self.classes], columns=columns).to_csv(path, index=False)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(c) for c in self.classes])


def update_metadata() -> DatasetMetadata:
    previous = DatasetMetadata.load(METADATA) if METADATA.exists() else None
    metadata = DatasetMetadata.from_dataset(DATASET_FOLDER, previous)
    metadata.save(METADATA)
    print(metadata.to_dataframe().to_string(index=False))
    return metadata
