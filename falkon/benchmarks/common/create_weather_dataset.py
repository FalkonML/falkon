import pickle
import numpy as np
import h5py

horizon = 6
memory = 72
input_name = "smz_CHIET.pkl"
output_name = "CHIET.hdf5"

o_index = horizon - 1

with open(input_name, "rb") as fh:
    data = pickle.load(fh)

otime = np.array(data["otime"])[:, o_index]
zout = np.array(data["O_zonal"])[:, o_index]
mout = np.array(data["O_merid"])[:, o_index]
sout = np.array(data["O_speed"])[:, o_index]

itime = np.array(data["itime"])[:, -memory:]
zinp = np.array(data["I_zonal"])[:, -memory:]
minp = np.array(data["I_merid"])[:, -memory:]
sinp = np.array(data["I_speed"])[:, -memory:]

X = np.concatenate((zinp, minp), axis=1)
Y = sout.reshape(-1, 1)

time_thresh = np.datetime64("2018-01-01")
tr_index = otime < time_thresh

train_x = X[tr_index, :]
test_x = X[~tr_index, :]
train_y = Y[tr_index, :]
test_y = Y[~tr_index, :]

with h5py.File(output_name, "w") as fh:
    fh.create_dataset("X_train", data=train_x)
    fh.create_dataset("X_test", data=test_x)
    fh.create_dataset("Y_train", data=train_y)
    fh.create_dataset("Y_test", data=test_y)
