import datetime
import os
import sys
import argparse
import tensorflow as tf
import numpy as np
ds = tf.contrib.distributions
import math
import random
from net import Networks

from tools import grad_clip

class Model(object):

    def __init__(self, Nets, beta=2e-3, K=int(256), batch_size=100, test_batch_size=10000, num_epochs=200, num_test_iters=10,
                 lr_new=1e-4, lr_for_mine=1e-4, algo='MINE_IB', moving_avg=1., moving_rate=0.01, max_grad_norm=None):
        # sess = tf_util.make_session()
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess = tf.InteractiveSession(config=config)
        test_labels = tf.placeholder(tf.int64, [None], 'test_labels')
        labels = tf.placeholder(tf.int64, [batch_size], 'labels')
        lr = tf.placeholder(tf.float32, shape=None)
        self.moving_avg = tf.convert_to_tensor(moving_avg)

        nets = Nets(K=K, batch_size=batch_size)
        test_nets = Nets(K=K, batch_size=test_batch_size)

        #regular_loss
        sample_wise_regular_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=nets.logits, labels=labels)
        regular_loss = tf.reduce_mean(sample_wise_regular_loss)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(nets.logits, 1), labels), tf.float32))
        accuracy_test = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(test_nets.logits, 1), test_labels), tf.float32))
        try:
            if algo=='MINE_IB':
                #construct the graph for eta first
                print(algo, 'lr_for_mine: ', str(lr_for_mine), ' beta: ', str(beta))
                self.moving_avg = (1 - moving_rate) * self.moving_avg + moving_rate * tf.reduce_mean(nets.et_eta)
                # MINE_loss = -(tf.reduce_mean(nets.t_eta) - (1. / tf.stop_gradient(self.moving_avg)) * (tf.reduce_mean(nets.et_eta)))
                MINE_loss = -beta*(tf.reduce_mean(nets.t_eta) - tf.log(tf.reduce_mean(nets.et_eta)))
                #construct the graph for phi and theta
                mutual_info_lowerbound = tf.reduce_mean(nets.t) - tf.log(tf.reduce_mean(nets.et))
                loss = regular_loss + beta*mutual_info_lowerbound

                grads_eta, global_norm_eta = grad_clip(MINE_loss, max_grad_norm, ['model/T'])
                trainer_eta = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999)
                _train_eta = trainer_eta.apply_gradients(grads_eta)
                grads, global_norm = grad_clip(loss, max_grad_norm, ['model/encoder', 'model/decoder'])
                trainer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999)
                _train = trainer.apply_gradients(grads)
            elif algo == 'regular':
                loss = regular_loss
                grads, global_norm = grad_clip(loss, max_grad_norm, ['model/encoder', 'model/decoder'])
                trainer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999)
                _train = trainer.apply_gradients(grads)
            elif algo == 'VRIB':
                labels_shuffle = tf.placeholder(tf.int64, [batch_size], 'labels_shuffle')
                sample_wise_regular_loss_shuffle = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=nets.logits_shuffle, labels=labels_shuffle)
                MINE_loss = - beta*(tf.reduce_mean(nets.t_eta) - tf.log(tf.reduce_mean(nets.et_eta*tf.exp(sample_wise_regular_loss_shuffle))))
                loss = beta*(tf.reduce_mean(nets.t) - tf.log(tf.reduce_mean(nets.et*tf.exp(sample_wise_regular_loss_shuffle))))

                grads_eta, global_norm_eta = grad_clip(MINE_loss, max_grad_norm, ['model/T'])
                trainer_eta = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999)
                _train_eta = trainer_eta.apply_gradients(grads_eta)
                grads, global_norm = grad_clip(loss, max_grad_norm, ['model/encoder', 'model/decoder'])
                trainer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999)
                _train = trainer.apply_gradients(grads)
            else:
                raise Exception('No such algorithm')
        except Exception as e:
            print(e)

        def evaluate(data, num_test_iters=num_test_iters):
            acc = 0.
            for i in range(num_test_iters):
                acc += sess.run(accuracy_test,feed_dict={test_nets.images: data.test.images,
                                                    test_labels: data.test.labels})
            return acc/float(num_test_iters)

        def train(data, batch_size=batch_size, num_epochs=num_epochs, lr_new=lr_new):
            logiter = 1000
            training_iter_with_train_data = data.train.num_examples // batch_size

            for epoch in range(num_epochs):
                if epoch % 2 == 1:
                    lr_new *= 0.97

                for i in range(training_iter_with_train_data):
                    img, label = data.train.next_batch(batch_size)

                    if algo == 'MINE_IB':
                        runop_eta = [MINE_loss, _train_eta]
                        runop = [accuracy, loss, mutual_info_lowerbound, _train]

                        index = range(repr.shape[0])
                        ml = 0.
                        for j in range(25):
                            repr = sess.run(nets.z, feed_dict={nets.images: img})
                            jnt = np.concatenate([img, repr], axis=-1)
                            marginal_index = np.random.choice(index, size=repr.shape[0], replace=False)
                            mgn = np.concatenate([img, repr[marginal_index]], axis=-1)
                            ml, ___ = sess.run(runop_eta, feed_dict={nets.joint_eta:jnt, nets.marginal_eta:mgn, lr:lr_for_mine})
                        img_shuffle_index = np.random.choice(index, size=repr.shape[0], replace=False)
                        acc, l, mi_lb, ____ = sess.run(runop, feed_dict={nets.images:img, labels:label, nets.images_shuffle:img[img_shuffle_index], lr:lr_new})

                        if i % logiter == 0:
                            print('{} epoch/{} iter\tLoss: {},  MINE Loss:{},  accuracy: {},  Mutual Information Lowerbound: {}'.format(
                                epoch, i, l, ml/beta, acc, mi_lb))
                    elif algo == 'regular':
                        runop = [accuracy, loss, _train]
                        acc, l, ____ = sess.run(runop, feed_dict={nets.images: img, labels: label, lr: lr_new})
                        if i % logiter == 0:
                            print('{} epoch/{} iter\tLoss: {}, accuracy: {}'.format(
                                epoch, i, l, acc))
                    else:
                        runop_eta = [MINE_loss, _train_eta]
                        runop = [loss, _train]

                        repr = sess.run(nets.z, feed_dict={nets.images: img})
                        jnt = np.concatenate([img, repr], axis=-1)
                        index = range(repr.shape[0])
                        ml = 0.
                        for j in range(1):
                            marginal_index = np.random.choice(index, size=repr.shape[0], replace=False)
                            mgn = np.concatenate([img, repr[marginal_index]], axis=-1)
                            ml, ___ = sess.run(runop_eta, feed_dict={nets.joint_eta:jnt, nets.marginal_eta:mgn, lr:lr_for_mine,
                                                                     nets.z_shuffle:repr[marginal_index], labels_shuffle:label[marginal_index]})
                        img_shuffle_index = np.random.choice(index, size=repr.shape[0], replace=False)
                        l, ____ = sess.run(runop, feed_dict={nets.images:img, nets.images_shuffle:img[img_shuffle_index],
                                                                  labels_shuffle:label[img_shuffle_index], lr:lr_new})

                        if i % logiter == 0:
                            print('{} epoch/{} iter\tLoss: {},  MINE Loss:{}'.format(
                                epoch, i, l, ml))

        self.evaluate = evaluate
        self.train = train
        tf.global_variables_initializer().run(session=sess)

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def learn(Nets, data, seed=0, beta=2e-3, K=int(256), batch_size=100, test_batch_size=10000, num_epochs=200,
          num_test_iters=10, lr_new=1e-4, lr_for_mine=1e-4, algo='MINE_IB', moving_avg=1., moving_rate=0.01, max_grad_norm=0.5):
    tf.reset_default_graph()
    set_global_seeds(seed)
    model = Model(Nets=Nets, beta=beta, K=K, batch_size=batch_size, test_batch_size=test_batch_size, num_epochs=num_epochs,
                  num_test_iters=num_test_iters, lr_new=lr_new, lr_for_mine=lr_for_mine, algo=algo, moving_avg=moving_avg,
                  moving_rate=moving_rate, max_grad_norm=max_grad_norm)
    model.train(data)
    acc = model.evaluate(data)
    return acc

def train(args):
    acc = 0.
    for seed in range(args.num_seeds):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist_data = input_data.read_data_sets('/tmp/mnistdata', validation_size=0)
        acc += learn(Networks, data=mnist_data, seed=args.seed_start+args.num_seeds, beta=args.beta,
                     batch_size=args.batch_size, test_batch_size=int(mnist_data.test.labels.shape[0]), num_epochs=args.epochs,
                     num_test_iters=args.num_test_iters, lr_new=args.lr, lr_for_mine=args.lr_for_mine, algo=args.algo,)
    return acc / float(args.num_seeds)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--beta', help='regularization coefficient', type=float, default=1e-3)
    parser.add_argument('--batch_size', help='number of samples at one batch', type=int, default=100)
    parser.add_argument('--epochs', help='number of epochs', type=int, default=200)
    parser.add_argument('--num_test_iters', help='number of test iterations', type=int, default=10)
    parser.add_argument('--seed_start', help='RNG seed', type=int, default=1)
    parser.add_argument('--num_seeds', help='number of random seeds', type=int, default=5)
    parser.add_argument('--algo', help='algorithm', default='MINE_IB')
    parser.add_argument('--data', help='which dataset to use', default='MNIST')
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--lr_for_mine', help='learning rate for mine', type=float, default=1.5e-4)
    args = parser.parse_args()
    acc = train(args)
    print('beta: ', str(args.beta), 'accuracy: ', str(acc), 'test error: ', str(1-acc))

if __name__ == "__main__":
    all=main()



