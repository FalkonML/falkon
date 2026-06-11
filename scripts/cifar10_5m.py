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
    transforms.Resize((224, 224)),
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
        return x, transform(x), int(y)


@torch.no_grad()
def extract_batch(model, imgs, device):
    imgs = imgs.to(device)
    feats = model(imgs)
    return feats.cpu().numpy()


def get_shard_sizes(files):
    shard_sizes = []
    for i, f in enumerate(files):
        with np.load(f, mmap_mode="r") as d:
            size = d["X"].shape[0]
        shard_sizes.append(size)
        print(f"Shard {i} at '{f}' has size {size}")
    return shard_sizes


def build_hdf5(tr_files, ts_files, out_path, out_path_feat, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_feature_extractor(device)
    feat_size = 1280
    pixel_size = 3072

    # compute total size first
    tr_shard_sizes = get_shard_sizes(tr_files)
    tr_total = sum(tr_shard_sizes)
    ts_shard_sizes = get_shard_sizes(ts_files)
    ts_total = sum(ts_shard_sizes)
    print(f"Total train samples: {tr_total} - test samples: {ts_total}")

    # create HDF5 file (streaming write)
    chunk_size = batch_size * 10
    with (
        h5py.File(out_path, "w") as h5,
        h5py.File(out_path_feat, "w") as h5_feat,
    ):
        Xtr_feat = h5_feat.create_dataset(
            "Xtr", shape=(tr_total, feat_size), dtype=np.float32, compression=1, chunks=(chunk_size, feat_size)
        )
        Ytr_feat = h5_feat.create_dataset(
            "Ytr", shape=(tr_total,), dtype=np.int32, compression=1, chunks=(chunk_size,)
        )
        Xts_feat = h5_feat.create_dataset(
            "Xts", shape=(ts_total, feat_size), dtype=np.float32, compression=1, chunks=(chunk_size, feat_size)
        )
        Yts_feat = h5_feat.create_dataset(
            "Yts", shape=(ts_total,), dtype=np.int32, compression=1, chunks=(chunk_size,)
        )
        Xtr_pix = h5.create_dataset(
            "Xtr", shape=(tr_total, pixel_size), dtype=np.uint8, compression=1, chunks=(chunk_size, pixel_size)
        )
        Ytr_pix = h5.create_dataset(
            "Ytr", shape=(tr_total,), dtype=np.int32, compression=1, chunks=(chunk_size,)
        )
        Xts_pix = h5.create_dataset(
            "Xts", shape=(ts_total, pixel_size), dtype=np.uint8, compression=1, chunks=(chunk_size, pixel_size)
        )
        Yts_pix = h5.create_dataset(
            "Yts", shape=(ts_total,), dtype=np.int32, compression=1, chunks=(chunk_size,)
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
            for img_og, imgs_pt, labels in tqdm(loader, desc=f"Training loader {shard_idx+1}/{len(tr_files)}"):
                bsz = imgs_pt.shape[0]
                feats = extract_batch(model, imgs_pt, device)
                pixels = np.array(img_og, copy=True).reshape(bsz, -1)
                Xtr_feat[write_ptr:write_ptr + bsz] = feats
                Xtr_pix[write_ptr: write_ptr + bsz] = pixels
                Ytr_feat[write_ptr:write_ptr + bsz] = labels.numpy()
                Ytr_pix[write_ptr:write_ptr + bsz] = labels.numpy()

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
            for img_og, imgs_pt, labels in tqdm(loader, desc=f"Test loader {shard_idx+1}/{len(ts_files)}"):
                bsz = imgs_pt.shape[0]
                feats = extract_batch(model, imgs_pt, device)
                pixels = np.array(img_og, copy=True).reshape(bsz, -1)
                Xts_feat[write_ptr:write_ptr + bsz] = feats
                Xts_pix[write_ptr:write_ptr + bsz] = pixels
                Yts_feat[write_ptr:write_ptr + bsz] = labels.numpy()
                Yts_pix[write_ptr:write_ptr + bsz] = labels.numpy()
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
        out_path="/data/DATASETS/CIFAR10-5M/pixelspace.hdf5",
        out_path_feat="/data/DATASETS/CIFAR10-5M/mbv2-features.hdf5",
        batch_size=256,
    )
