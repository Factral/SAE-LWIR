import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class AtmosphericDataset(Dataset):
    """
    Real dataset backed by .npy files (loaded via lazy mmap).

    Expected shapes:
    - downwelling:    (N, 256)
    - forward:        (N, 7, 7, 256)
    - transmittance:  (N, 7, 256)
    - upwelling:      (N, 7, 256)

    __getitem__ returns:
      x=forward[idx], y1=transmittance[idx], y2=upwelling[idx], y3=downwelling[idx]
    """

    def __init__(self, data_dir=None):
        super().__init__()
        if data_dir is None:
            # default to: <repo_root>/data/
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        self.data_dir = os.path.abspath(data_dir)

        self.down_path = os.path.join(self.data_dir, "downwelling.npy")
        self.fwd_path = os.path.join(self.data_dir, "forward.npy")
        self.trn_path = os.path.join(self.data_dir, "transmittance.npy")
        self.up_path = os.path.join(self.data_dir, "upwelling.npy")

        self.down = torch.from_numpy(np.load(self.down_path, mmap_mode="r"))
        self._fwd = torch.from_numpy(np.load(self.fwd_path, mmap_mode="r")) # N_ATMOSPHERES, N_RANGES, N_TEMPERATURES, BANDS
        self.trn = torch.from_numpy(np.load(self.trn_path, mmap_mode="r")) # N_ATMOSPHERES, N_RANGES, BANDS
        self.up = torch.from_numpy(np.load(self.up_path, mmap_mode="r"))


        self.n_depths = 7
        self.n = self._fwd.shape[0] * self.n_depths # N_ATMOSPHERES * N_TEMPERATURES

        self.fwd = self._fwd.permute(0,2,1,3).contiguous() # N, N_TEMP, N_RANGES, BANDS
        self.temps = [280., 285., 290., 295., 300., 305., 310.] 


        

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        atm_idx = idx // self.n_depths # N_ATMOSPHERES
        temp_idx = idx % len(self.temps) # N_TEMPERATURES
        x = self.fwd[atm_idx, temp_idx, :self.n_depths]

        y1 = self.trn[atm_idx, :self.n_depths]
        y2 = self.up[atm_idx, :self.n_depths]
        y3 = self.down[atm_idx].unsqueeze(0)  # (1, 256)

        return x.float(), y1.float(), y2.float(), y3.float(), self.temps[temp_idx]


def make_split_dataloader(
    *,
    split="train",
    data_dir=None,
    batch_size=128,
    num_workers=0,
    pin_memory=False,
    train_drop_last=True,
    split_seed=0,
):
    """
    Returns a DataLoader for one split of a 70/10/20 train/val/test partition.

    Good practices:
    - Deterministic split controlled by split_seed.
    - Train loader shuffles; val/test do not.
    - drop_last typically only for train.
    """

    split = str(split).lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: 'train', 'val', 'test'")

    dataset = AtmosphericDataset(data_dir=data_dir)
    n = len(dataset)

    train_len = int(n * 0.70)
    val_len = int(n * 0.10)
    test_len = n - train_len - val_len

    split_gen = torch.Generator()
    split_gen.manual_seed(split_seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_len, val_len, test_len], generator=split_gen
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train_drop_last,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


