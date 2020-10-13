from typing import Dict, Callable, Any, Tuple, Union, Optional, List

import numpy as np
from benchmark_utils import Dataset


def rmse(y_true, y_pred, **kwargs):
    if not isinstance(y_true, np.ndarray):
        y_true: np.ndarray = y_true.cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred: np.ndarray = y_pred.cpu().numpy()

    y_true = y_true.reshape((-1, ))
    y_pred = y_pred.reshape((-1, ))

    test_mse = np.sqrt(((y_pred - y_true)**2).mean())
    return test_mse, "RMSE"


def mse(y_true, y_pred, **kwargs):
    if not isinstance(y_true, np.ndarray):
        y_true: np.ndarray = y_true.cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred: np.ndarray = y_pred.cpu().numpy()

    y_true = y_true.reshape((-1, ))
    y_pred = y_pred.reshape((-1, ))

    test_mse = ((y_pred - y_true)**2).mean()
    return test_mse, "MSE"


def rmse_with_std(y_true, y_pred, **kwargs):
    Y_std = kwargs['Y_std']

    if not isinstance(y_true, np.ndarray):
        y_true: np.ndarray = y_true.cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred: np.ndarray = y_pred.cpu().numpy()
    if (not isinstance(Y_std, np.ndarray) and
        not isinstance(Y_std, np.float64) and
        not isinstance(Y_std, np.float32) and
        not isinstance(Y_std, float)):
        Y_std = Y_std.cpu().numpy()

    y_true = y_true.reshape((-1, ))
    y_pred = y_pred.reshape((-1, ))

    test_mse = np.sqrt(((y_pred*Y_std - y_true*Y_std)**2).mean())
    return test_mse, "RMSE"


def ms_calc_mse(y_true, y_pred, **kwargs):
    Y_std = kwargs['Y_std']

    if not isinstance(y_true, np.ndarray):
        y_true: np.ndarray = y_true.cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred: np.ndarray = y_pred.cpu().numpy()
    if (not isinstance(Y_std, np.ndarray) and
        not isinstance(Y_std, np.float64) and
        not isinstance(Y_std, np.float32) and
        not isinstance(Y_std, float)):
        Y_std = Y_std.cpu().numpy()

    y_true = y_true.reshape((-1, ))
    y_pred = y_pred.reshape((-1, ))

    test_mse = ((y_pred*Y_std - y_true*Y_std)**2).mean()
    return test_mse, "MSE"


def ms_calc_relerr(y_true, y_pred, **kwargs):
    Y_std = kwargs['Y_std']
    Y_mean = kwargs['Y_mean']

    if not isinstance(y_true, np.ndarray):
        y_true: np.ndarray = y_true.cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred: np.ndarray = y_pred.cpu().numpy()
    if (not isinstance(Y_std, np.ndarray) and
        not isinstance(Y_std, np.float64) and
        not isinstance(Y_std, np.float32) and
        not isinstance(Y_std, float)):
        Y_std = Y_std.cpu().numpy()
    if (not isinstance(Y_mean, np.ndarray) and
        not isinstance(Y_mean, np.float64) and
        not isinstance(Y_mean, np.float32) and
        not isinstance(Y_mean, float)):
        Y_mean = Y_mean.cpu().numpy()

    y_true = y_true.reshape((-1, ))
    y_pred = y_pred.reshape((-1, ))

    Uypred = y_pred * Y_std + Y_mean
    Uytrue = y_true * Y_std + Y_mean
    rel_err = np.sqrt( np.mean(((Uytrue - Uypred) / Uytrue)**2) )
    return rel_err, "relative error"


def ms_calc_mse_tf(y_true, y_pred, **kwargs):
    Y_std = kwargs['Y_std']

    import tensorflow as tf
    return tf.math.reduce_mean(
            tf.math.square(
            tf.math.subtract(tf.math.multiply(tf.reshape(y_true, (-1,)), Y_std),
                             tf.math.multiply(tf.reshape(y_pred, (-1,)), Y_std))
            ))


def rmse_with_std_tf(y_true, y_pred, **kwargs):
    Y_std = kwargs['Y_std']

    import tensorflow as tf
    return tf.math.sqrt(tf.math.reduce_mean(
            tf.math.square(
            tf.math.subtract(tf.math.multiply(tf.reshape(y_true, (-1,)), Y_std),
                             tf.math.multiply(tf.reshape(y_pred, (-1,)), Y_std))
            )))

def rmse_tf(y_true, y_pred, **kwargs):
    import tensorflow as tf
    return tf.math.sqrt(tf.math.reduce_mean(
            tf.math.square(
            tf.math.subtract(tf.reshape(y_true, (-1,)),
                             tf.reshape(y_pred, (-1,)))
            )))

def mse_tf(y_true, y_pred, **kwargs):
    import tensorflow as tf
    return tf.math.reduce_mean(
            tf.math.square(
            tf.math.subtract(tf.reshape(y_true, (-1,)),
                             tf.reshape(y_pred, (-1,)))
            ))

def calc_auc_tf(y_true, y_pred, **kwargs):
    tf.keras.metrics.AUC(
        num_thresholds=200, curve='ROC', summation_method='interpolation', name=None,
        dtype=None, thresholds=None, multi_label=False, label_weights=None
    )


def higgs_calc_auc(y_true, y_pred, **kwargs):
    from sklearn import metrics
    if not isinstance(y_true, np.ndarray):
        y_true: np.ndarray = y_true.cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred: np.ndarray = y_pred.cpu().numpy()

    if np.min(y_true) == 0:
        y_true = y_true * 2 - 1
        y_pred = y_pred * 2 - 1

    fpr, tpr, thresholds = metrics.roc_curve(
        y_true.reshape((-1, 1)), y_pred.reshape((-1, 1)), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc, "AUC"


def binary_cerr(y_true, y_pred, **kwargs):
    if not isinstance(y_true, np.ndarray):
        y_true: np.ndarray = y_true.cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred: np.ndarray = y_pred.cpu().numpy()

    if np.min(y_true) == 0:
        y_true = y_true * 2 - 1
        y_pred = y_pred * 2 - 1

    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1) * 2 - 1
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1) * 2 - 1

    c_err = np.mean(np.sign(y_pred.ravel()) != np.sign(y_true.ravel()))
    return c_err, "c-error"


def mnist_calc_cerr(y_true, y_pred, **kwargs):
    if not isinstance(y_true, np.ndarray):
        y_true: np.ndarray = y_true.cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred: np.ndarray = y_pred.cpu().numpy()

    if y_true.ndim > 1 and y_true.shape[1] > 2:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 2:
        y_pred = np.argmax(y_pred, axis=1)

    return np.mean(y_true.ravel() != y_pred.ravel()), "c-error"

def mnist_calc_cerr_tf(y_true, y_pred, **kwargs):
    import tensorflow as tf
    y_true = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
    y_pred = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)

    return tf.reduce_mean(tf.cast(tf.math.equal(y_true, y_pred), tf.dtypes.float64))


def binary_cerr_tf(y_true, y_pred, **kwargs):
    import tensorflow as tf
    return tf.reduce_mean(tf.cast(tf.math.not_equal(
        tf.math.sign(tf.reshape(y_true, [-1])),
        tf.math.sign(tf.reshape(y_pred, [-1]))), tf.dtypes.float64))


def timit_calc_error(y_true, y_pred, **kwargs):
    if not isinstance(y_true, np.ndarray):
        y_true: np.ndarray = y_true.cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred: np.ndarray = y_pred.cpu().numpy()

    if y_true.ndim > 1 and y_true.shape[1] > 2:
        y_true = np.argmax(np.sum(y_true.reshape((-1, 48, 3)), axis=2), axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 2:
        y_pred = np.argmax(np.sum(y_pred.reshape((-1, 48, 3)), axis=2), axis=1)

    return np.mean(y_true.ravel() != y_pred.ravel()), "c-error"


def timit_calc_error_tf(y_true, y_pred, **kwargs):
    import tensorflow as tf
    y_true = tf.math.argmax(
                tf.math.reduce_sum(
                    tf.reshape(y_true, (-1, 48, 3)), axis=2),
                    axis=1, output_type=tf.dtypes.int32)

    y_pred = tf.math.argmax(
                tf.math.reduce_sum(
                    tf.reshape(y_pred, (-1, 48, 3)), axis=2),
                    axis=1, output_type=tf.dtypes.int32)

    return tf.reduce_mean(tf.cast(tf.math.not_equal(y_true, y_pred), tf.dtypes.float64))


ARRAY_TYPE = Any
ERROR_FN_TYPE = Callable[[Any, Any, Dict[str, Any]], Tuple[float, str]]

ERROR_METRICS: Dict[Dataset, List[ERROR_FN_TYPE]] = {
    Dataset.TIMIT: [timit_calc_error],
    Dataset.MILLIONSONGS: [ms_calc_relerr, ms_calc_mse],
    Dataset.HIGGS: [higgs_calc_auc, binary_cerr],
    Dataset.TAXI: [rmse_with_std],
    Dataset.YELP: [rmse],
    Dataset.FLIGHTS: [mse],
    Dataset.SUSY: [higgs_calc_auc, binary_cerr],
    Dataset.FLIGHTS_CLS: [binary_cerr, higgs_calc_auc],
    Dataset.MNIST: [mnist_calc_cerr],
    Dataset.MNIST_SMALL: [mnist_calc_cerr],
    Dataset.SVHN: [mnist_calc_cerr],
    Dataset.CIFAR10: [mnist_calc_cerr],
}
TF_ERROR_METRICS: Dict[Dataset, ERROR_FN_TYPE] = {
    Dataset.TIMIT: timit_calc_error_tf,
    Dataset.MILLIONSONGS: ms_calc_mse_tf,
    Dataset.FLIGHTS: mse_tf,
    Dataset.MNIST: mnist_calc_cerr_tf,
    Dataset.SUSY: binary_cerr_tf,
    Dataset.FLIGHTS_CLS: binary_cerr_tf,
}

def get_err_fns(dset: Dataset) -> List[ERROR_FN_TYPE]:
    try:
        return ERROR_METRICS[dset]
    except KeyError:
        raise KeyError(dset, f"No error metrics found for dataset {dset}.")

def get_tf_err_fn(dset: Dataset) -> ERROR_FN_TYPE:
    try:
        return TF_ERROR_METRICS[dset]
    except KeyError:
        raise KeyError(dset, f"No tensorflow error metric found for dataset {dset}.")

