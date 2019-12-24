import tensorflow as tf
from tensorflow.contrib.solvers.python.ops.util import l2norm
import numpy as np

def fc(x, scope, nh, *, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b

def grad_clip(loss, max_grad_norm, scope_list):
    '''
    :param loss:
    :param params:
    :param max_grad_norm:
    :param scope: a list consist of variable scopes
    :return:
    '''
    params_list = []
    for scope in scope_list:
        List = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope)
        print(len(List))
        params_list += List
    # print(len(params_list))
    grads = tf.gradients(loss, params_list)
    # for i, grad in enumerate(grads):
    #     if grad is None:
    #         grads[i] = tf.zeros(shape=params_list[i].get_shape(), dtype=params_list[i].dtype)
    global_norm = 0.
    if max_grad_norm is not None:
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        global_norm = tf.sqrt(sum([l2norm(t) ** 2 for t in grads]))
    grads = list(zip(grads, params_list))
    return grads, global_norm
