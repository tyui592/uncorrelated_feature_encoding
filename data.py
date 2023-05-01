# -*- coding: utf-8 -*-
"""Data code.

* Author: Minseong Kim(tyui592@gmail.com)
"""

import torch
from PIL import Image
from pathlib import Path
from random import randint
from utils import get_transformer
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    """Dataset for training."""

    def __init__(self, root_path, max_iter, transforms):
        """Init."""
        super(Dataset, self).__init__()

        path = Path(root_path)
        self.file_paths = sorted(list(path.glob('*.jpg')))
        self.length = len(self.file_paths)

        self.max_iter = max_iter
        self.transforms = transforms

    def __len__(self):
        """Length for training iteration."""
        return self.max_iter

    def __getitem__(self, _):
        """Get item randomly."""
        index = randint(0, self.length - 1)
        image = Image.open(self.file_paths[index]).convert('RGB')
        return self.transforms(image)


def get_dataloader(path, imsize, cropsize, cencrop, max_iter, batch_size):
    """Get dataloder."""
    transforms = get_transformer(imsize, cropsize, cencrop)

    dataset = Dataset(path, max_iter * batch_size, transforms)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader
