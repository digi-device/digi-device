# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import shutil
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter

DATASET = "lacofloris/type-1-diabetes-blood-glucose-prediction"
# Path of a single file inside the dataset to load; leave empty to copy every CSV
FILE_PATH = ""
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def download_dataset(dataset=DATASET, file_path=FILE_PATH, data_dir=DATA_DIR):
    """Download a Kaggle dataset as CSV(s) into data_dir/<dataset name> (created if missing).

    With file_path, that one file is loaded and written as a CSV. Without it, the
    whole dataset is downloaded and every .csv in it is copied, keeping the folder layout.
    """
    out_dir = Path(data_dir) / dataset.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)

    if file_path:
        df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, dataset, file_path)
        print("First 5 records:", df.head())
        out_path = out_dir / f"{Path(file_path).stem}.csv"
        df.to_csv(out_path, index=False)
        print(out_path)
        return out_dir

    src_dir = Path(kagglehub.dataset_download(dataset))
    n_copied = 0
    for src in src_dir.rglob("*.csv"):
        dest = out_dir / src.relative_to(src_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        n_copied += 1
    print(f"Copied {n_copied} CSV files to {out_dir}")
    return out_dir


if __name__ == "__main__":
    download_dataset()
