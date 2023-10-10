import pathlib
import fastbook
import kaggle
import shutil

def get_kaggle_dataset(comp: str) -> pathlib.Path:
    """
    Download and unzip the dataset for the provided competition name (comp).
    Return the local path to the dataset.
    
    If the local path already exists, print a warning and return the local path.
    """
    kaggle_api_credentials = pathlib.Path('~/.kaggle/kaggle.json').expanduser().read_text()
    path = fastbook.URLs.path(comp)
    if path.exists():
        print(path, "already exists.")
        return path
    path.mkdir(parents=True)
    kaggle.api.competition_download_cli(comp, path=path)
    shutil.unpack_archive(str(path/f'{comp}.zip'), str(path))
    return path

