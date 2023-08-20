from typing import Dict, Callable, Any, Tuple, List, Generator, Union

import numpy as np

from .benchmark_utils import Dataset


def _ensure_numpy(*arrays) -> Generator[np.ndarray, None, None]:
    for arr in arrays:
        if not isinstance(arr, np.ndarray):
            yield arr.cpu().numpy()
        else:
            yield arr


def _ensure_numpy_or_float(*vals) -> Generator[Union[float, np.ndarray], None, None]:
    for val in vals:
        if (
            not isinstance(val, np.ndarray)
            and not isinstance(val, np.float64)
            and not isinstance(val, np.float32)
            and not isinstance(val, float)
        ):
            yield val.cpu().numpy()
        else:
            yield val


def mse(y_true, y_pred, **kwargs):
    y_true, y_pred = _ensure_numpy(y_true, y_pred)

    y_true = y_true.reshape((-1,))
    y_pred = y_pred.reshape((-1,))

    test_mse = ((y_pred - y_true) ** 2).mean()
    return test_mse, "MSE"


def rmse(y_true, y_pred, **kwargs):
    pred_mse = mse(y_true, y_pred, **kwargs)[0]
    pred_rmse = np.sqrt(pred_mse)
    return pred_rmse, "RMSE"


def rmse_with_std(y_true, y_pred, **kwargs):
    Y_std = kwargs["Y_std"]

    y_true, y_pred = _ensure_numpy(y_true, y_pred)
    (Y_std,) = _ensure_numpy_or_float(Y_std)

    y_true = y_true.reshape((-1,))
    y_pred = y_pred.reshape((-1,))

    test_mse = np.sqrt(((y_pred * Y_std - y_true * Y_std) ** 2).mean())
    return test_mse, "RMSE"


def nrmse(y_true, y_pred, **kwargs):
    Y_mean = kwargs["Y_mean"]
    (Y_mean,) = _ensure_numpy_or_float(Y_mean)
    Y_std = kwargs.get("Y_std", 1.0)
    (Y_std,) = _ensure_numpy_or_float(Y_std)

    y_true = y_true * Y_std + Y_mean
    y_pred = y_pred * Y_std + Y_mean

    pred_rmse = rmse(y_true, y_pred, **kwargs)[0]
    pred_nrmse = np.abs(pred_rmse / Y_mean)
    return pred_nrmse, "NRMSE"


def ms_calc_mse(y_true, y_pred, **kwargs):
    Y_std = kwargs["Y_std"]

    y_true, y_pred = _ensure_numpy(y_true, y_pred)
    (Y_std,) = _ensure_numpy_or_float(Y_std)

    y_true = y_true.reshape((-1,))
    y_pred = y_pred.reshape((-1,))

    test_mse = ((y_pred * Y_std - y_true * Y_std) ** 2).mean()
    return test_mse, "MSE"


def ms_calc_relerr(y_true, y_pred, **kwargs):
    Y_std = kwargs.get("Y_std", 1.0)
    Y_mean = kwargs["Y_mean"]

    y_true, y_pred = _ensure_numpy(y_true, y_pred)
    Y_std, Y_mean = _ensure_numpy_or_float(Y_std, Y_mean)

    y_true = y_true.reshape((-1,))
    y_pred = y_pred.reshape((-1,))

    Uypred = y_pred * Y_std + Y_mean
    Uytrue = y_true * Y_std + Y_mean
    rel_err = np.sqrt(np.mean(((Uytrue - Uypred) / Uytrue) ** 2))
    return rel_err, "relative error"


def ms_calc_mse_tf(y_true, y_pred, **kwargs):
    Y_std = kwargs["Y_std"]

    import tensorflow as tf

    return tf.math.reduce_mean(
        tf.math.square(
            tf.math.subtract(
                tf.math.multiply(tf.reshape(y_true, (-1,)), Y_std), tf.math.multiply(tf.reshape(y_pred, (-1,)), Y_std)
            )
        )
    )


def rmse_with_std_tf(y_true, y_pred, **kwargs):
    Y_std = kwargs["Y_std"]

    import tensorflow as tf

    return tf.math.sqrt(
        tf.math.reduce_mean(
            tf.math.square(
                tf.math.subtract(
                    tf.math.multiply(tf.reshape(y_true, (-1,)), Y_std),
                    tf.math.multiply(tf.reshape(y_pred, (-1,)), Y_std),
                )
            )
        )
    )


def rmse_tf(y_true, y_pred, **kwargs):
    import tensorflow as tf

    return tf.math.sqrt(
        tf.math.reduce_mean(tf.math.square(tf.math.subtract(tf.reshape(y_true, (-1,)), tf.reshape(y_pred, (-1,)))))
    )


def mse_tf(y_true, y_pred, **kwargs):
    import tensorflow as tf

    return tf.math.reduce_mean(tf.math.square(tf.math.subtract(tf.reshape(y_true, (-1,)), tf.reshape(y_pred, (-1,)))))


def higgs_calc_auc(y_true, y_pred, **kwargs):
    from sklearn import metrics

    y_true, y_pred = _ensure_numpy(y_true, y_pred)
    y_true = y_true.reshape((-1, 1))
    y_pred = y_pred.reshape((-1, 1))

    if np.min(y_true) == 0:
        y_true = y_true * 2 - 1
        y_pred = y_pred * 2 - 1

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return (1.0 - auc), "1-AUC"


def binary_cerr(y_true, y_pred, **kwargs):
    y_true, y_pred = _ensure_numpy(y_true, y_pred)

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
    y_true, y_pred = _ensure_numpy(y_true, y_pred)

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

    return tf.reduce_mean(
        tf.cast(
            tf.math.not_equal(tf.math.sign(tf.reshape(y_true, [-1])), tf.math.sign(tf.reshape(y_pred, [-1]))),
            tf.dtypes.float64,
        )
    )


def timit_calc_error(y_true, y_pred, **kwargs):
    y_true, y_pred = _ensure_numpy(y_true, y_pred)

    if y_true.ndim > 1 and y_true.shape[1] > 2:
        y_true = np.argmax(np.sum(y_true.reshape((-1, 48, 3)), axis=2), axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 2:
        y_pred = np.argmax(np.sum(y_pred.reshape((-1, 48, 3)), axis=2), axis=1)

    return np.mean(y_true.ravel() != y_pred.ravel()), "c-error"


def timit_calc_error_tf(y_true, y_pred, **kwargs):
    import tensorflow as tf

    y_true = tf.math.argmax(
        tf.math.reduce_sum(tf.reshape(y_true, (-1, 48, 3)), axis=2), axis=1, output_type=tf.dtypes.int32
    )

    y_pred = tf.math.argmax(
        tf.math.reduce_sum(tf.reshape(y_pred, (-1, 48, 3)), axis=2), axis=1, output_type=tf.dtypes.int32
    )

    return tf.reduce_mean(tf.cast(tf.math.not_equal(y_true, y_pred), tf.dtypes.float64))


ARRAY_TYPE = Any
ERROR_FN_TYPE = Callable[[Any, Any, Dict[str, Any]], Tuple[float, str]]

ERROR_METRICS: Dict[Dataset, List[ERROR_FN_TYPE]] = {
    Dataset.TIMIT: [timit_calc_error],
    Dataset.MILLIONSONGS: [ms_calc_relerr, ms_calc_mse],
    Dataset.HIGGS: [higgs_calc_auc, binary_cerr],
    Dataset.HOHIGGS: [binary_cerr, higgs_calc_auc],
    Dataset.TAXI: [rmse_with_std],
    Dataset.YELP: [rmse],
    Dataset.FLIGHTS: [mse],
    Dataset.SUSY: [higgs_calc_auc, binary_cerr],
    Dataset.FLIGHTS_CLS: [binary_cerr, higgs_calc_auc],
    Dataset.MNIST: [mnist_calc_cerr],
    Dataset.MNIST_SMALL: [mnist_calc_cerr],
    Dataset.SVHN: [mnist_calc_cerr],
    Dataset.CIFAR10: [mnist_calc_cerr],
    Dataset.CIFAR10RGB: [mnist_calc_cerr],
    Dataset.ICTUS: [binary_cerr],
    Dataset.SYNTH01NOISE: [rmse],
    Dataset.CHIET: [nrmse],
    Dataset.ENERGY: [nrmse],
    Dataset.BOSTON: [nrmse],
    Dataset.PROTEIN: [nrmse],
    Dataset.KIN40K: [nrmse],
    Dataset.CODRNA: [binary_cerr],
    Dataset.SVMGUIDE1: [binary_cerr],
    Dataset.PHISHING: [binary_cerr],
    Dataset.SPACEGA: [nrmse],
    Dataset.CADATA: [nrmse],
    Dataset.MG: [nrmse],
    Dataset.CPUSMALL: [nrmse],
    Dataset.ABALONE: [nrmse],
    Dataset.CASP: [nrmse],
    Dataset.BLOGFEEDBACK: [rmse],
    Dataset.COVTYPE: [binary_cerr],
    Dataset.IJCNN1: [binary_cerr],
    Dataset.FASHION_MNIST: [mnist_calc_cerr],
    Dataset.BUZZ: [nrmse],
    Dataset.ROAD3D: [nrmse],
    Dataset.HOUSEELECTRIC: [nrmse],
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
        raise KeyError(dset, f"No error metrics found for dataset {dset}.") from None


def get_tf_err_fn(dset: Dataset) -> ERROR_FN_TYPE:
    try:
        return TF_ERROR_METRICS[dset]
    except KeyError:
        raise KeyError(dset, f"No tensorflow error metric found for dataset {dset}.") from None
