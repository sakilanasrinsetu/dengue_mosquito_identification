"""Download the Aedes dataset into this project's datasets directory."""

import os
from pathlib import Path


if __name__ == '__main__':
    cache = Path(__file__).resolve().parent / 'datasets'
    os.environ.setdefault('KAGGLEHUB_CACHE', str(cache))

    import kagglehub

    path = kagglehub.dataset_download('pradeepisawasan/aedes-mosquitos')
    print('Path to dataset files:', path)
