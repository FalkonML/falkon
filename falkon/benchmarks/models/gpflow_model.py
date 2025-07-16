import time
from functools import partial

import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf
from gpflow import set_trainable
from gpflow.models import SVGP
from gpflow.optimizers import NaturalGradient


@tf.function(autograph=False)
def elbo_opt_step(optimizer, model, batch):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        objective = -model.elbo(batch)
        grads = tape.gradient(objective, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return objective


def data_generator(X, Y, batch_size):
    bstart = 0
    while bstart < X.shape[0]:
        bend = min(X.shape[0], bstart + batch_size)
        yield tf.convert_to_tensor(X[bstart:bend]), tf.convert_to_tensor(Y[bstart:bend])
        bstart = bend


class TrainableGPR:
    def __init__(self, kernel, num_iter, err_fn, lr):
        self.kernel = kernel
        self.num_iter = num_iter
        self.err_fn = err_fn
        self.lr = lr
        self.model = None

    def fit(self, X, Y, Xval, Yval):
        self.model = gpflow.models.GPR((X, Y), kernel=self.kernel, noise_variance=0.1)
        # Create the optimizers
        adam_opt = tf.optimizers.Adam(self.lr)

        gpflow.utilities.print_summary(self.model)
        print("", flush=True)

        @tf.function
        def step_fn():
            adam_opt.minimize(self.model.training_loss, var_list=self.model.trainable_variables)
            return True

        @tf.function
        def pred_fn():
            return self.model.predict_y(Xval)[0]

        t_elapsed = 0
        for step in range(self.num_iter):
            t_s = time.time()
            outcome = step_fn()
            outcome = int(outcome) + 1
            t_elapsed += time.time() - t_s
            if (step + 1) % 1 == 0:
                val_err, err_name = self.err_fn(Yval, pred_fn())
                print(
                    f"Epoch {step + 1} - {t_elapsed:7.2f}s elapsed - " f"validation {err_name} {val_err:7.5f}",
                    flush=True,
                )
                print(f"\tLengthscale: {self.kernel.lengthscales}")

        print("Final model is ")
        gpflow.utilities.print_summary(self.model)
        print("", flush=True)
        return self

    def predict(self, X, pred_fn=None):
        return self.model.predict_y(X)[0]

    def __str__(self):
        return f"TrainableGPR<kernel={self.kernel}, num_iter={self.num_iter}, lr={self.lr}, model={self.model}>"


class TrainableSGPR:
    def __init__(
        self,
        kernel,
        inducing_points,
        num_iter,
        err_fn,
        train_hyperparams: bool = True,
        lr: float = 0.001,
    ):
        self.train_hyperparams = train_hyperparams
        self.lr = lr
        self.kernel = kernel
        self.Z = inducing_points.copy()
        self.num_iter = num_iter
        self.err_fn = err_fn
        self.model = None
        self.optimizer = "adam"

    def fit(self, X, Y, Xval, Yval):
        # Only Gaussian likelihood allowed
        self.model = gpflow.models.SGPR((X, Y), kernel=self.kernel, inducing_variable=self.Z, noise_variance=0.001)
        # self.model.likelihood.variance = gpflow.Parameter(1, transform=tfp.bijectors.Identity())

        # Setup training parameters
        if not self.train_hyperparams:
            set_trainable(self.model.inducing_variable.Z, False)

        gpflow.utilities.print_summary(self.model)
        print("", flush=True)

        @tf.function
        def grad_fn():
            grads = tf.gradients(self.model.training_loss(), self.model.trainable_variables)
            return grads

        if self.optimizer == "scipy":
            opt = gpflow.optimizers.Scipy()

            def scipy_callback(step, variables, value):
                print(f"Step {step} - Variables: {value}")

            opt.minimize(
                self.model.training_loss,
                self.model.trainable_variables,
                method="L-BFGS-B",
                options=dict(maxiter=self.num_iter, ftol=1e-32, maxcor=3, gtol=1e-16, disp=False),
                step_callback=scipy_callback,
                compile=True,
            )
        else:
            if self.optimizer == "adam":
                opt = tf.optimizers.Adam(self.lr)
            elif self.optimizer == "sgd":
                opt = tf.optimizers.SGD(self.lr)
            else:
                raise ValueError(f"Optimizer {self.optimizer} unknown")

            @tf.function
            def step_fn():
                opt.minimize(self.model.training_loss, var_list=self.model.trainable_variables)

            t_elapsed = 0
            for step in range(self.num_iter):
                t_s = time.time()
                step_fn()
                t_elapsed += time.time() - t_s
                val_err, err_name = self.err_fn(Yval, self.predict(Xval))
                gpflow.utilities.print_summary(self.model)
                print(
                    f"Epoch {step + 1} - {t_elapsed:7.2f}s elapsed - " f"validation {err_name} {val_err:7.5f}",
                    flush=True,
                )
            print(self.model.inducing_variable.Z.numpy())

        print("Final model is ")
        gpflow.utilities.print_summary(self.model)
        print("", flush=True)
        return self

    def gradient_map(self, X, Y, Xval, Yval, variance_list, lengthscale_list):
        self.model = gpflow.models.SGPR((X, Y), kernel=self.kernel, inducing_variable=self.Z, noise_variance=0.1)
        # Setup parameters for which to compute gradient. We want only 2 params!
        set_trainable(self.model.inducing_variable.Z, False)
        set_trainable(self.model.kernel.variance, False)
        set_trainable(self.model.kernel.lengthscales, True)
        set_trainable(self.model.likelihood.variance, True)

        @tf.function
        def grad_fn():
            grads = tf.gradients(self.model.training_loss(), self.model.trainable_variables)
            return grads

        df = pd.DataFrame(columns=["sigma", "sigma_g", "variance", "variance_g", "elbo"])
        for lscale in lengthscale_list:
            self.model.kernel.lengthscales.assign([lscale])
            for var in variance_list:
                self.model.likelihood.variance.assign(var)
                # self.model.kernel.variance.assign([var])
                grads = [g.numpy() for g in grad_fn()]
                train_preds = self.model.predict_y(X)[0]
                test_preds = self.model.predict_y(Xval)[0]
                new_row = {
                    "sigma": lscale,
                    "sigma_g": grads[0][0],
                    "variance": var,
                    "variance_g": grads[1],
                    "elbo": self.model.elbo().numpy(),
                }
                print(f"ELBO: {new_row['elbo']:10.3f} - TRAINING LOSS: {self.model.training_loss():10.3f}")
                tr_err, tr_err_name = self.err_fn(Y, train_preds)
                ts_err, ts_err_name = self.err_fn(Yval, test_preds)
                new_row[f"train_{tr_err_name}"] = tr_err
                new_row[f"test_{ts_err_name}"] = ts_err
                df = df.append(new_row, ignore_index=True)
                print(new_row)
        return df

    def predict(self, X):
        return self.model.predict_y(X)[0]

    @property
    def inducing_points(self):
        return self.model.inducing_variable.Z.numpy()

    def __str__(self):
        return (
            f"TrainableSGPR<kernel={self.kernel}, num_inducing_points={self.Z.shape[0]}, "
            f"num_iter={self.num_iter}, lr={self.lr}, train_hyperparams={self.train_hyperparams}, model={self.model}>"
        )


class TrainableSVGP:
    def __init__(
        self,
        kernel,
        inducing_points,
        batch_size,
        num_iter,
        err_fn,
        var_dist,
        classif=None,
        error_every=100,
        train_hyperparams: bool = True,
        optimize_centers: bool = True,
        lr: float = 0.001,
        natgrad_lr: float = 0.01,
    ):
        self.train_hyperparams = train_hyperparams
        self.optimize_centers = optimize_centers
        self.lr = lr
        self.natgrad_lr = natgrad_lr
        self.kernel = kernel
        self.Z = inducing_points.copy()
        self.batch_size = batch_size
        self.num_iter = num_iter
        self.err_fn = err_fn
        self.error_every = error_every
        self.do_classif = classif is not None and classif > 0
        self.num_classes = 1
        if self.do_classif:
            self.num_classes = int(classif)
        self.model = None
        self.whiten = True
        self.var_dist = var_dist

    def fit(self, X, Y, Xval, Yval):
        N = X.shape[0]

        if self.var_dist == "diag":
            q_diag = True
        elif self.var_dist == "full":
            q_diag = False
        else:
            raise NotImplementedError(f"GPFlow cannot implement {self.var_dist} variational distribution")

        if self.natgrad_lr > 0 and q_diag:
            raise ValueError("The variational distribution must be 'full' with natural gradients")

        if self.do_classif:
            if self.num_classes == 2:
                likelihood = gpflow.likelihoods.Bernoulli()
                num_latent = 1
            else:
                # Softmax better than Robustmax (apparently per the gpflow slack)
                # likelihood = gpflow.likelihoods.MultiClass(self.num_classes, invlink=invlink)  # Multiclass likelihood
                likelihood = gpflow.likelihoods.Softmax(self.num_classes)
                num_latent = self.num_classes
                # Y must be 1D for the multiclass model to actually work.
                Y = np.argmax(Y, 1).reshape((-1, 1)).astype(int)
        else:
            num_latent = 1
            likelihood = gpflow.likelihoods.Gaussian(variance=0.1)

        self.model = SVGP(
            kernel=self.kernel,
            likelihood=likelihood,
            inducing_variable=self.Z,
            num_data=N,
            num_latent_gps=num_latent,
            whiten=self.whiten,
            q_diag=q_diag,
        )
        # Setup training
        set_trainable(self.model.inducing_variable.Z, self.optimize_centers)
        if not self.train_hyperparams:
            set_trainable(self.model.inducing_variable.Z, False)
            set_trainable(self.model.likelihood.variance, False)
            set_trainable(self.kernel.lengthscales, False)
            set_trainable(self.kernel.variance, False)
        if self.natgrad_lr > 0:
            set_trainable(self.model.q_mu, False)
            set_trainable(self.model.q_sqrt, False)
            variational_params = [(self.model.q_mu, self.model.q_sqrt)]
        # Create the optimizers
        adam_opt = tf.optimizers.Adam(self.lr)
        if self.natgrad_lr > 0:
            natgrad_opt = NaturalGradient(gamma=self.natgrad_lr)

        # Print
        gpflow.utilities.print_summary(self.model)
        print("", flush=True)

        # Giacomo: If shuffle buffer is too large it will run OOM
        if self.num_classes == 2:
            Y = (Y + 1) / 2
            Yval = (Yval + 1) / 2
        generator = partial(data_generator, X, Y)
        if X.dtype == np.float32:
            tf_dt = tf.float32
        else:
            tf_dt = tf.float64
        train_dataset = (
            tf.data.Dataset.from_generator(generator, args=(self.batch_size,), output_types=(tf_dt, tf_dt))
            .prefetch(self.batch_size * 10)
            .repeat()
            .shuffle(min(N // self.batch_size, 1_000_000 // self.batch_size))
            .batch(1)
        )
        train_iter = iter(train_dataset)

        loss = self.model.training_loss_closure(train_iter)
        t_elapsed = 0

        @tf.function
        def step_fn():
            adam_opt.minimize(loss, var_list=self.model.trainable_variables)
            if self.natgrad_lr > 0:
                natgrad_opt.minimize(loss, var_list=variational_params)
            return True

        for step in range(self.num_iter):
            t_s = time.time()
            outcome = step_fn()
            outcome = int(outcome) + 1
            t_elapsed += time.time() - t_s
            if step % 500 == 0:
                print(f"Step {step} -- Elapsed {t_elapsed:.2f}s", flush=True)
                gpflow.utilities.print_summary(self.model)
                print(self.model.inducing_variable.Z.numpy())
            if (step + 1) % self.error_every == 0:
                preds = self.predict(Xval)
                val_err, err_name = self.err_fn(Yval, preds)
                print(
                    f"Step {step + 1} - {t_elapsed:7.2f}s Elapsed - " f"Validation {err_name} {val_err:7.5f}",
                    flush=True,
                )

        preds = self.predict(Xval)
        val_err, err_name = self.err_fn(Yval, preds)
        print(
            f"Finished optimization - {t_elapsed:7.2f}s Elapsed - " f"Validation {err_name} {val_err:7.5f}", flush=True
        )
        print("Final model is ")
        gpflow.utilities.print_summary(self.model)
        print("", flush=True)
        return self

    def predict(self, X):
        preds = []
        dset = tf.data.Dataset.from_tensor_slices((X,)).batch(self.batch_size)
        for X_batch in iter(dset):
            batch_preds = self.model.predict_y(X_batch[0])[0].numpy()
            if self.do_classif:
                batch_preds = batch_preds.reshape((X_batch[0].shape[0], -1))
            preds.append(batch_preds)
        preds = np.concatenate(preds, axis=0)
        return preds

    def gradient_map(self, X, Y, Xval, Yval, variance_list, lengthscale_list):
        N = X.shape[0]
        likelihood = gpflow.likelihoods.Gaussian(variance=0.1)
        self.model = SVGP(
            kernel=self.kernel,
            likelihood=likelihood,
            inducing_variable=self.Z,
            num_data=N,
            num_latent_gps=1,
            whiten=self.whiten,
            q_diag=False,
        )  # var-dist must be full covar when using natgrad
        # Setup training parameters. We want only 2 params.
        set_trainable(self.model.inducing_variable.Z, False)
        set_trainable(self.kernel.variance, False)
        set_trainable(self.kernel.lengthscales, True)
        set_trainable(self.model.likelihood.variance, True)
        # Variational parameters will be optimized with natgrad.
        set_trainable(self.model.q_mu, False)
        set_trainable(self.model.q_sqrt, False)

        # Set-up for natgrad optimization
        variational_params = [(self.model.q_mu, self.model.q_sqrt)]
        natgrad_opt = NaturalGradient(gamma=1.0)
        generator = partial(data_generator, X, Y)
        if X.dtype == np.float32:
            tf_dt = tf.float32
        else:
            tf_dt = tf.float64
        print(tf_dt)
        train_dataset = (
            tf.data.Dataset.from_generator(generator, args=(self.batch_size,), output_types=(tf_dt, tf_dt))
            .prefetch(self.batch_size * 10)
            .repeat()
            .shuffle(min(N // self.batch_size, 1_000_000 // self.batch_size))
            .batch(1)
        )
        train_iter = iter(train_dataset)
        loss = self.model.training_loss_closure(train_iter)

        @tf.function
        def grad_fn():
            grads = tf.gradients(self.model.training_loss((X, Y)), self.model.trainable_variables)
            return grads

        df = pd.DataFrame(columns=["sigma", "sigma_g", "variance", "variance_g", "elbo"])
        for lscale in lengthscale_list:
            self.model.kernel.lengthscales.assign([lscale])
            for var in variance_list:
                self.model.likelihood.variance.assign(var)

                # Optimize variational parameters (a single iteration is enough with lr=1)
                natgrad_opt.minimize(loss, var_list=variational_params)

                # Get gradients and save output in df row.
                grads = [g.numpy() for g in grad_fn()]
                train_preds = self.model.predict_y(X)[0]
                test_preds = self.model.predict_y(Xval)[0]
                new_row = {
                    "sigma": lscale,
                    "sigma_g": grads[0][0],
                    "variance": var,
                    "variance_g": grads[1],
                    "elbo": self.model.elbo((X, Y)).numpy(),
                }
                tr_err, tr_err_name = self.err_fn(Y, train_preds)
                ts_err, ts_err_name = self.err_fn(Yval, test_preds)
                new_row[f"train_{tr_err_name}"] = tr_err
                new_row[f"test_{ts_err_name}"] = ts_err
                df = df.append(new_row, ignore_index=True)
                print(new_row)
        return df

    @property
    def inducing_points(self):
        return self.model.inducing_variable.Z.numpy()

    def __str__(self):
        return (
            f"TrainableSVGP<kernel={self.kernel}, num_inducing_points={self.Z.shape[0]}, batch_size={self.batch_size}, "
            f"num_iter={self.num_iter}, lr={self.lr}, natgrad_lr={self.natgrad_lr}, error_every={self.error_every}, "
            f"train_hyperparams={self.train_hyperparams}, var_dist={self.var_dist}, do_classif={self.do_classif}, "
            f"model={self.model}, whiten={self.whiten}>"
        )
