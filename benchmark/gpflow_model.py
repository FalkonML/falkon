import time
from functools import partial
import numpy as np
import gpflow
from gpflow.models import SVGP
from gpflow.optimizers import NaturalGradient
from gpflow import set_trainable
import tensorflow as tf


@tf.function(autograph=False)
def elbo_opt_step(optimizer, model, batch):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        objective = - model.elbo(batch)
        grads = tape.gradient(objective, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return objective


def data_generator(X, Y, batch_size):
    bstart = 0
    while bstart < X.shape[0]:
        bend = min(X.shape[0], bstart + batch_size)
        yield tf.convert_to_tensor(X[bstart:bend]), tf.convert_to_tensor(Y[bstart:bend])
        bstart = bend


class TrainableSVGP():
    def __init__(self,
                 kernel,
                 inducing_points,
                 batch_size,
                 num_iter,
                 err_fn,
                 var_dist,
                 classif=None,
                 error_every=100,
                 train_hyperparams: bool = True,
                 lr: float = 0.001,
                 natgrad_lr: float = 0.01):
        self.train_hyperparams = train_hyperparams
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
        self.var_dist = var_dist

    def fit(self, X, Y, Xval, Yval):
        N = X.shape[0]

        if self.var_dist == "diag":
            q_diag = True
        elif self.var_dist == "full":
            q_diag = False
        else:
            raise NotImplementedError("GPFlow cannot implement %s variational distribution" % (self.var_dist))

        if self.natgrad_lr > 0 and q_diag:
            raise ValueError("The variational distribution must be 'full' with natural gradients")

        if self.do_classif:
            if self.num_classes == 2:
                likelihood = gpflow.likelihoods.Bernoulli()
                num_latent = 1
            else:
                # Softmax better than Robustmax (apparently per the gpflow slack)
                #likelihood = gpflow.likelihoods.MultiClass(self.num_classes, invlink=invlink)  # Multiclass likelihood
                likelihood = gpflow.likelihoods.Softmax(self.num_classes)
                num_latent = self.num_classes
                # Y must be 1D for the multiclass model to actually work.
                Y = np.argmax(Y, 1).reshape((-1,1)).astype(int)
        else:
            num_latent = 1
            likelihood = gpflow.likelihoods.Gaussian()

        self.model = SVGP(
            kernel=self.kernel,
            likelihood=likelihood,
            inducing_variable=self.Z,
            num_data=N,
            num_latent_gps=num_latent,
            whiten=False,
            q_diag=q_diag)
        # Setup training
        if not self.train_hyperparams:
            set_trainable(self.model.inducing_variable.Z, False)
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
        #train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)) \
        train_dataset = tf.data.Dataset.from_generator(generator, args=(self.batch_size, ), output_types=(tf.float32, tf.float32)) \
            .prefetch(self.batch_size * 10) \
            .repeat() \
            .shuffle(min(N // self.batch_size, 1_000_000 // self.batch_size)) \
            .batch(1)
        train_iter = iter(train_dataset)

        loss = self.model.training_loss_closure(train_iter)
        t_elapsed = 0
        for step in range(self.num_iter):
            t_s = time.time()
            adam_opt.minimize(loss, var_list=self.model.trainable_variables)
            if self.natgrad_lr > 0:
                natgrad_opt.minimize(loss, var_list=variational_params)
            t_elapsed += time.time() - t_s
            if step % 700 == 0:
                print("Step %d -- Elapsed %.2fs" % (step, t_elapsed), flush=True)
            if (step + 1) % self.error_every == 0:
                preds = self.predict(Xval)
                val_err, err_name = self.err_fn(Yval, preds)
                print(f"Step {step + 1} - {t_elapsed:7.2f}s Elapsed - "
                      f"Validation {err_name} {val_err:7.5f}",
                      flush=True)

        preds = self.predict(Xval)
        val_err, err_name = self.err_fn(Yval, preds)
        print(f"Finished optimization - {t_elapsed:7.2f}s Elapsed - "
              f"Validation {err_name} {val_err:7.5f}", flush=True)
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

    @property
    def inducing_points(self):
        return self.model.inducing_variable.Z.numpy()

    def __str__(self):
        return (("TrainableSVGP<kernel=%s, num_inducing_points=%d, batch_size=%d, "
                 "num_iter=%d, lr=%f, natgrad_lr=%f, error_every=%d, train_hyperparams=%s, "
                 "var_dist=%s, do_classif=%s, model=%s")
                % (self.kernel, self.Z.shape[0], self.batch_size, self.num_iter, self.lr,
                   self.natgrad_lr, self.error_every, self.train_hyperparams,
                   self.var_dist, self.do_classif, self.model))


