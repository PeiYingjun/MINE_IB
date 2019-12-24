import tensorflow as tf
import numpy as np
layers = tf.contrib.layers
ds = tf.contrib.distributions
from tools import fc

class Networks(object):
    def __init__(self, K=int(256), batch_size=100, reuse=tf.AUTO_REUSE):
        images = tf.placeholder(tf.float32, [batch_size, 784], 'images')
        images_shuffle = tf.placeholder(tf.float32, [batch_size, 784], 'images_shuffle')
        joint_eta = tf.placeholder(tf.float32, shape=[batch_size, 784 + 256])
        marginal_eta = tf.placeholder(tf.float32, shape=[batch_size, 784 + 256])

        with tf.variable_scope("model", reuse=reuse):
            activ_Classifier = tf.nn.relu
            activ_MINE = tf.nn.elu

            def encoder(images, scope='encoder', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    h1 = activ_Classifier(fc(2 * images - 1, 'h1', nh=1024))
                    h2 = activ_Classifier(fc(h1, 'h2', nh=1024))
                    Gaussian_params = activ_Classifier(fc(h2, 'Gaussian_params', nh=2*K))
                    mu, logvar = Gaussian_params[:, :K], Gaussian_params[:, K:]
                    eps = tf.random_normal(shape=tf.shape(mu))
                    stddev = tf.nn.softplus(0.5 * logvar - 5.0)
                    return mu + eps * stddev

            def decoder(encoding_sample, scope='decoder', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    logits = fc(encoding_sample, 'logits', nh=10)
                    return logits

            def T(inputs, scope='T', reuse=reuse):
                with tf.variable_scope(scope, reuse=reuse):
                    inputs += tf.random_normal(shape=tf.shape(inputs))*0.3
                    t1 = activ_MINE(fc(inputs, 't1', nh=512))
                    t1 += tf.random_normal(shape=tf.shape(t1))*0.5
                    t2 = activ_MINE(fc(t1, 't2', nh=512))
                    t2 += tf.random_normal(shape=tf.shape(t2)) * 0.5
                    return fc(t2, 't3', nh=1)

            z = encoder(images)
            logits = decoder(z)
            #eta
            t_eta, et_eta = T(joint_eta)[:,0], tf.exp(T(marginal_eta))[:,0]
            #theta and phi
            joint = tf.concat([images,z],axis=-1)
            # idx = tf.reshape(tf.range(start=0, limit=batch_size, dtype=tf.int32), [-1, 1])
            # idx_shuffle = tf.random_shuffle(idx)
            # images_shuffle = tf.gather_nd(images, indices=idx_shuffle)
            z_shuffle = encoder(images_shuffle)
            logits_shuffle = decoder(z_shuffle)
            marginal = tf.concat([images,z_shuffle],axis=-1)
            t, et= T(joint), tf.exp(T(marginal))

        self.images, self.images_shuffle, self.joint_eta, self.marginal_eta = images, images_shuffle, joint_eta, marginal_eta
        self.z, self.z_shuffle, self.logits, self.logits_shuffle, self.t_eta, self.et_eta, self.t, self.et = z, z_shuffle, logits, logits_shuffle, t_eta, et_eta, t, et
        # self.encoder, self.decoder, self.T = encoder, decoder, T