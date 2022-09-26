import tensorflow as tf
from tensorflow import keras

from wloss import wasserstein_loss, CL_chooser


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        raise RuntimeError("This should not be getting called; VAE is trained via VAE_trainer class")

    def predict(self, *args):
        raise RuntimeError("Do not call predict directly; call encoder and decoder separately")


class VAE_trainer(keras.Model):
    def __init__(self, VAE, disc, kl_weight, ensemble_size, CLtype, content_loss_weight, **kwargs):
        super(VAE_trainer, self).__init__(**kwargs)
        self.VAE = VAE
        self.disc = disc
        self.kl_weight = kl_weight
        self.ensemble_size = ensemble_size
        self.CLtype = CLtype
        self.content_loss_weight = content_loss_weight
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.vaegen_loss_tracker = keras.metrics.Mean(name="vaegen_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        if self.ensemble_size is not None:
            self.content_loss_tracker = keras.metrics.Mean(name="content_loss")

    @property
    def metrics(self):
        if self.ensemble_size is not None:
            return [
                self.total_loss_tracker,
                self.vaegen_loss_tracker,
                self.kl_loss_tracker,
                self.content_loss_tracker,
            ]
        else:
            return [
                self.total_loss_tracker,
                self.vaegen_loss_tracker,
                self.kl_loss_tracker,
            ]

    @tf.function
    def train_step(self, data):
        gt_inputs, gt_outputs = data
        cond, const, *noise = gt_inputs
        if self.ensemble_size is None:
            gen_target, = gt_outputs
        else:
            gen_target, truthimg = gt_outputs

        batch_size = cond.shape[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.VAE.encoder([cond, const])
            pred = self.VAE.decoder([z_mean, z_log_var, noise[0], const])
            # apply disc to decoder predictions
            y_pred = self.disc([cond, const, pred])

            # target vector of ones used for wasserstein loss
            vaegen_loss = wasserstein_loss(gen_target, y_pred)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            # "flatten" kl_loss to batch_size x n_latent_vars
#             data_shape = kl_loss.get_shape().as_list()
#             temp_dim = tf.reduce_prod(data_shape[1:])
#             kl_loss = tf.reshape(kl_loss, [-1, temp_dim])
            kl_loss = tf.reshape(kl_loss, [batch_size, -1])
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            if self.ensemble_size is None:
                total_loss = vaegen_loss + kl_loss*tf.constant(self.kl_weight)
            else:
                # generate ensemble of predictions for content loss
                preds = [self.VAE.decoder([z_mean, z_log_var, noise[ii+1], const])
                         for ii in range(self.ensemble_size)]
                preds = tf.stack(preds, axis=0)  # ens x batch x W x H x 1
                preds = tf.squeeze(preds, axis=-1)  # ens x batch x W x H

                CLfn = CL_chooser(self.CLtype)
                content_loss = CLfn(truthimg, preds)
                total_loss = vaegen_loss + kl_loss*tf.constant(self.kl_weight) + content_loss*tf.constant(self.content_loss_weight)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.vaegen_loss_tracker.update_state(vaegen_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        if self.ensemble_size is not None:
            self.content_loss_tracker.update_state(content_loss)
        return [tracker.result() for tracker in self.metrics]
        # tf.print(self.total_loss_tracker.result(), self.vaegen_loss_tracker.result(), self.kl_loss_tracker.result())
#         return {
#             "loss": self.total_loss_tracker.result(),
#             "vaegen loss": self.vaegen_loss_tracker.result(),
#             "kl_loss": self.kl_loss_tracker.result(),
#         }

    def predict(self, *args):
        raise RuntimeError("Should not be calling .predict on VAE_trainer")
