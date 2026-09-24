import h5py
import numpy as np
from sklearn.datasets import fetch_openml

data, target = fetch_openml(data_id=41150, return_X_y=True)
data_vals = data.values.astype(np.float64)
target_vals = target.map({"True": 1, "False": 0}).values.astype(np.int32)

with h5py.File("MiniBooNE_v2.h5", "w") as h5:
    X = h5.create_dataset(
        "X", data=data_vals, compression=1
    )
    Y = h5.create_dataset(
        "Y", data=target_vals, compression=1
    )
print("Done writing MiniBooNE_v2.h5 dataset")