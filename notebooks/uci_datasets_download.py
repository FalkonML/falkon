import argparse
import os
import tempfile

import h5py
import numpy as np
import pandas as pd
import requests


def download_protein(out_dir):
    protein_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
    with tempfile.TemporaryDirectory() as tmp_dir:
        protein_file = os.path.join(tmp_dir, 'protein.csv')
        r = requests.get(protein_url, allow_redirects=True)
        with open(protein_file, 'wb') as fh:
            fh.write(r.content)
        df = pd.read_csv(protein_file, index_col=None)
        print(df.head())
        df = df.astype(float)
        Y = df["RMSD"].values.reshape(-1, 1)
        X = df[["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]].values
        assert X.shape == (45730, 9)
        assert Y.shape == (45730, 1)
        protein_hdf_file = os.path.join(out_dir, 'protein.hdf5')
        with h5py.File(protein_hdf_file, 'w') as hf:
            hf.create_dataset("X", data=X, dtype=np.float64, compression='gzip')
            hf.create_dataset("Y", data=Y, dtype=np.float64, compression='gzip')


def download_boston(out_dir):
    boston_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    with tempfile.TemporaryDirectory() as tmp_dir:
        boston_file = os.path.join(tmp_dir, 'boston.tsv')
        r = requests.get(boston_url, allow_redirects=True)
        with open(boston_file, 'wb') as fh:
            fh.write(r.content)
        df = pd.read_csv(boston_file, index_col=None, delim_whitespace=True, header=None)
        print(df.head())
        df = df.astype(float)
        Y = df[0].values.reshape(-1, 1)
        X = df.drop(0, axis=1).values
        assert X.shape == (506, 13)
        assert Y.shape == (506, 1)
        boston_hdf_file = os.path.join(out_dir, 'boston.hdf5')
        with h5py.File(boston_hdf_file, 'w') as hf:
            hf.create_dataset("X", data=X, dtype=np.float64, compression='gzip')
            hf.create_dataset("Y", data=Y, dtype=np.float64, compression='gzip')


def download_energy(out_dir):
    energy_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    with tempfile.TemporaryDirectory() as tmp_dir:
        energy_file = os.path.join(tmp_dir, 'energy.xlsx')
        r = requests.get(energy_url, allow_redirects=True)
        with open(energy_file, 'wb') as fh:
            fh.write(r.content)
        df = pd.read_excel(energy_file, engine='openpyxl', convert_float=False)
        df = df.drop(["Unnamed: 10", "Unnamed: 11"], axis=1)
        df = df.dropna(axis=0, how='all')
        df.head()
        df = df.astype(float)
        Y = df["Y1"].values.reshape(-1, 1)  # heating load
        X = df.drop(["Y1", "Y2"], axis=1).values
        assert X.shape == (768, 8)
        assert Y.shape == (768, 1)
        energy_hdf_file = os.path.join(out_dir, 'energy.hdf5')
        with h5py.File(energy_hdf_file, 'w') as hf:
            hf.create_dataset("X", data=X, dtype=np.float64, compression='gzip')
            hf.create_dataset("Y", data=Y, dtype=np.float64, compression='gzip')


def download_kin40k(out_dir):
    """
    Data is impossible to find from reputable sources. Delve repository does not have 40k points (only 8192).
    Github repository with full data: https://github.com/trungngv/fgp
    """
    url_test_y = "https://github.com/trungngv/fgp/raw/master/data/kin40k/kin40k_test_labels.asc"
    url_train_y = "https://github.com/trungngv/fgp/raw/master/data/kin40k/kin40k_train_labels.asc"
    url_test_x = "https://github.com/trungngv/fgp/raw/master/data/kin40k/kin40k_test_data.asc"
    url_train_x = "https://github.com/trungngv/fgp/raw/master/data/kin40k/kin40k_train_data.asc"
    with tempfile.TemporaryDirectory() as tmp_dir:
        f_test_y = os.path.join(tmp_dir, "kin40k_test_labels.asc")
        f_train_y = os.path.join(tmp_dir, "kin40k_train_labels.asc")
        f_test_x = os.path.join(tmp_dir, "kin40k_test_data.asc")
        f_train_x = os.path.join(tmp_dir, "kin40k_train_data.asc")
        for (url, file) in [(url_test_y, f_test_y), (url_train_y, f_train_y),
                            (url_test_x, f_test_x), (url_train_x, f_train_x)]:
            r = requests.get(url, allow_redirects=True)
            with open(file, 'wb') as fh:
                fh.write(r.content)
        test_y = pd.read_fwf(f_test_y, header=None, index_col=None) \
            .astype(float).values.reshape(-1, 1)
        train_y = pd.read_fwf(f_train_y, header=None, index_col=None) \
            .astype(float).values.reshape(-1, 1)
        test_x = pd.read_fwf(f_test_x, header=None, index_col=None) \
            .astype(float).values
        train_x = pd.read_fwf(f_train_x, header=None, index_col=None) \
            .astype(float).values
        assert test_y.shape == (30_000, 1)
        assert train_y.shape == (10_000, 1)
        assert test_x.shape == (30_000, 8)
        assert train_x.shape == (10_000, 8)
        kin40k_hdf_file = os.path.join(out_dir, 'kin40k.hdf5')
        with h5py.File(kin40k_hdf_file, 'w') as hf:
            hf.create_dataset("Y_test", data=test_y, dtype=np.float64, compression='gzip')
            hf.create_dataset("Y_train", data=train_y, dtype=np.float64, compression='gzip')
            hf.create_dataset("X_test", data=test_x, dtype=np.float64, compression='gzip')
            hf.create_dataset("X_train", data=train_x, dtype=np.float64, compression='gzip')


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dataset downloader")
    p.add_argument('-d', '--out-dir', type=str, required=True,
                   help="Output directory for the downloaded and processed datasets.")
    args = p.parse_args()
    download_fns = [download_energy, download_protein, download_boston, download_kin40k]
    print(f"Will download datasets: {download_fns} to directory {args.out_dir}...")
    for fn in download_fns:
        fn(args.out_dir)
