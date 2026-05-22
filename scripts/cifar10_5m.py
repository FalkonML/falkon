import numpy as np
import torch
import torch.nn as nn
import h5py
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

"""
Pre-processing the CIFAR10-5M dataset (https://github.com/preetum/cifar5m)
1. Load MobileNetv2 using pytorch
2. Parse the images (1 file at a time)
3. Extract features batch-wise within each file
4. Save images and features to a hdf5 file with the following keys
    Xtr, Xts, Ytr, Yts
   where the training set are the first 5M images and the test set are any remaining images in the data
   where X data are the features (flattened) and Y data are the class labels stored as an integer (0 to 9).
"""

# 1. Feature extractor (MobileNetV2)
def get_feature_extractor(device):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Identity()  # remove classification head
    model = model.to(device)
    model.eval()
    return model

# -------------------------
# 2. Image preprocessing
# -------------------------
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------
# 3. Example dataset wrapper (ADAPT THIS to CIFAR10-5M structure)
# -------------------------
class CIFAR5MDataset(Dataset):
    def __init__(self, file, total_length):
        self.npz_file = file
        with np.load(self.npz_file) as fh:
            self.X = fh["X"]
            self.Y = fh["Y"]
        self.total_len = total_length

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return transform(x), int(y)

@torch.no_grad()
def extract_batch(model, imgs, device):
    imgs = imgs.to(device)
    feats = model(imgs)
    return feats.cpu().numpy()


def build_hdf5(tr_files, ts_files, out_path, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_feature_extractor(device)

    # ---- compute total size first (cheap metadata scan)
    tr_total = 0
    tr_shard_sizes = []
    for i, f in enumerate(tr_files):
        with np.load(f, mmap_mode="r") as d:
            size = d["X"].shape[0]
        tr_total += size
        tr_shard_sizes.append(size)
        print(f"Shard {i} has size {size}")
    ts_total = 0
    ts_shard_sizes = []
    for i, f in enumerate(ts_files):
        with np.load(f, mmap_mode="r") as d:
            size = d["X"].shape[0]
        ts_total += size
        ts_shard_sizes.append(size)
        print(f"Shard {i} has size {size}")
    print(f"Total train samples: {tr_total} - test samples: {ts_total}")

    # ---- create HDF5 file (streaming write)
    with h5py.File(out_path, "w") as h5:
        Xtr = h5.create_dataset(
            "Xtr", shape=(tr_total, 1280), dtype=np.float32, compression="gzip", chunks=(batch_size, 1280)
        )
        Ytr = h5.create_dataset(
            "Ytr", shape=(tr_total,), dtype=np.int32, compression="gzip", chunks=(batch_size,)
        )
        Xts = h5.create_dataset(
            "Xts", shape=(ts_total, 1280), dtype=np.float32, compression="gzip", chunks=(batch_size, 1280)
        )
        Yts = h5.create_dataset(
            "Yts", shape=(ts_total,), dtype=np.int32, compression="gzip", chunks=(batch_size,)
        )

        write_ptr = 0
        for shard_idx, file in enumerate(tr_files):
            print(f"Processing shard {shard_idx+1}/{len(tr_files)}")
            dataset = CIFAR5MDataset(file, tr_shard_sizes[shard_idx])
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            for imgs, labels in tqdm(loader, desc=f"Training loader {shard_idx+1}/{len(tr_files)}"):
                feats = extract_batch(model, imgs, device)
                bsz = feats.shape[0]
                Xtr[write_ptr:write_ptr + bsz] = feats
                Ytr[write_ptr:write_ptr + bsz] = labels.numpy()

                write_ptr += bsz

        write_ptr = 0
        for shard_idx, file in enumerate(ts_files):
            print(f"Processing shard {shard_idx+1}/{len(ts_files)}")
            dataset = CIFAR5MDataset(file, ts_shard_sizes[shard_idx])
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )
            for imgs, labels in tqdm(loader, desc=f"Test loader {shard_idx+1}/{len(ts_files)}"):
                feats = extract_batch(model, imgs, device)
                bsz = feats.shape[0]
                Xts[write_ptr:write_ptr + bsz] = feats
                Yts[write_ptr:write_ptr + bsz] = labels.numpy()
                write_ptr += bsz

# 6. Example usage
if __name__ == "__main__":
    build_hdf5(
        [
            "/data/DATASETS/CIFAR10-5M/cifar5m_part0.npz",
            "/data/DATASETS/CIFAR10-5M/cifar5m_part1.npz",
            "/data/DATASETS/CIFAR10-5M/cifar5m_part2.npz",
            "/data/DATASETS/CIFAR10-5M/cifar5m_part3.npz",
            "/data/DATASETS/CIFAR10-5M/cifar5m_part4.npz",
        ],
        [
            "/data/DATASETS/CIFAR10-5M/cifar5m_part5.npz",
        ],
        out_path="/data/DATASETS/CIFAR10-5M/mbv2-features.hdf5",
        batch_size=256,
    )
