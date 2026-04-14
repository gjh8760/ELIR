import glob
import os
import random
from PIL import Image, ImageOps
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset

from ELIR.datasets.dataset import BasicLoader



class LOLv1Dataset(Dataset):
    """Paired LOL-v1 dataset for Low-Light Image Enhancement.

    Expected directory layout::

        root/
          our485/{low,high}/*.png   # training pairs
          eval15/{low,high}/*.png   # test pairs

    In train mode a paired random crop of `patch_size` and a synchronized
    random horizontal flip are applied; in eval mode the full-resolution image
    (native 600x400) is returned and full-resolution inference is handled by
    the `chop` mechanism in `IRSetup.validation_step`.
    """

    def __init__(self, root, split, patch_size=256, is_train=True):
        super().__init__()
        assert split in ("our485", "eval15"), \
            f"Unknown split '{split}' for LOL-v1"
        self.root = root
        self.split = split
        self.patch_size = patch_size
        self.is_train = is_train

        low_dir = os.path.join(root, split, "low")
        high_dir = os.path.join(root, split, "high")
        low_paths = sorted(glob.glob(os.path.join(low_dir, "*.png")))
        high_paths = sorted(glob.glob(os.path.join(high_dir, "*.png")))
        assert len(low_paths) > 0, f"No low images under {low_dir}"
        assert len(low_paths) == len(high_paths), (
            f"low/high count mismatch: {len(low_paths)} vs {len(high_paths)}")
        for lp, hp in zip(low_paths, high_paths):
            assert os.path.basename(lp) == os.path.basename(hp), (
                f"Unpaired filenames: {os.path.basename(lp)} vs "
                f"{os.path.basename(hp)}")
        self.low_paths = low_paths
        self.high_paths = high_paths

        self.to_tensor = v2.ToTensor()

    def __len__(self):
        return len(self.low_paths)

    def _paired_random_crop(self, lq, hq):
        W, H = lq.size
        ph = min(self.patch_size, H)
        pw = min(self.patch_size, W)
        top = random.randint(0, H - ph)
        left = random.randint(0, W - pw)
        lq = lq.crop((left, top, left + pw, top + ph))
        hq = hq.crop((left, top, left + pw, top + ph))
        return lq, hq

    def __getitem__(self, index):
        lq = Image.open(self.low_paths[index]).convert("RGB")
        hq = Image.open(self.high_paths[index]).convert("RGB")

        if self.is_train:
            lq, hq = self._paired_random_crop(lq, hq)
            if random.random() < 0.5:
                lq = ImageOps.mirror(lq)
                hq = ImageOps.mirror(hq)

        lq = self.to_tensor(lq)
        hq = self.to_tensor(hq)
        return lq, hq


class LOLv1(BasicLoader):
    def __init__(self):
        super().__init__()

    def create_loaders(self, dataset_params):
        path = dataset_params.get("path")
        split = dataset_params.get("split", "our485")
        batch_size = dataset_params.get("batch_size", 8)
        patch_size = dataset_params.get("patch_size", 256)
        num_workers = dataset_params.get("num_workers", 4)
        is_train = dataset_params.get("is_train", True)

        dataset = LOLv1Dataset(root=path,
                               split=split,
                               patch_size=patch_size,
                               is_train=is_train)

        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=is_train,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=is_train,
                            persistent_workers=(num_workers > 0 and is_train))
        return loader
