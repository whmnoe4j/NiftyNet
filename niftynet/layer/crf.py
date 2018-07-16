# -*- coding: utf-8 -*-
"""
Re-implementation of [1] in Tensorflow for volumetric image processing.


[1] Zheng, Shuai, et al.
"Conditional random fields as recurrent neural networks." CVPR 2015.
https://arxiv.org/abs/1502.03240
"""
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.layer_util import infer_spatial_rank, expand_spatial_params


class CRFAsRNNLayer(TrainableLayer):
    """
    This class defines a layer implementing CRFAsRNN described in [1] using
    a bilateral and a spatial kernel as in [2].
    Essentially, this layer smooths its input based on a distance in a feature
    space comprising spatial and feature dimensions.

    [1] Zheng et al., https://arxiv.org/abs/1502.03240
    [2] Krahenbuhl and Koltun, https://arxiv.org/pdf/1210.5644.pdf
    """

    def __init__(self,
                 alpha=5.,
                 beta=5.,
                 gamma=5.,
                 T=5,
                 aspect_ratio=None,
                 name="crf_as_rnn"):
        """

        :param alpha: bandwidth for spatial coordinates in bilateral kernel.
                      Higher values cause more spatial blurring
        :param beta: bandwidth for feature coordinates in bilateral kernel
                      Higher values cause more feature blurring
        :param gamma: bandwidth for spatial coordinates in spatial kernel
                      Higher values cause more spatial blurring
        :param T: number of stacked layers in the RNN
        :param aspect_ratio: spacing of adjacent voxels
            (allows isotropic spatial smoothing when voxels are not isotropic
        :param name:
        """

        super(CRFAsRNNLayer, self).__init__(name=name)
        self._T = T
        self._aspect_ratio = aspect_ratio
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    def layer_op(self, I, U):
        """
        Compute `T` iterations of mean field update given a dense CRF.

        This layer maintains trainable CRF model parameters
        (a compatibility function and `m` kernel weights).

        :param I: feature maps used in the dense pairwise term of CRF
        :param U: activation maps used in the unary term of CRF (before softmax)
        :return: Maximum a posteriori labeling (before softmax)
        """

        spatial_dim = infer_spatial_rank(U)
        all_shape = U.shape.as_list()
        batch_size, spatial_shape, nCh = \
            all_shape[0], all_shape[1:-1], all_shape[-1]
        n_feat = I.shape.as_list()[-1]
        if self._aspect_ratio is None:
            self._aspect_ratio = [1.] * spatial_dim

        # constructing the scaled regular grid
        spatial_coords = tf.meshgrid(
            *[np.arange(i, dtype=np.float32) * a
                for i, a in zip(spatial_shape, self._aspect_ratio)],
            indexing='ij')
        spatial_coords = tf.stack(spatial_coords, spatial_dim)
        spatial_coords = tf.tile(
            tf.expand_dims(spatial_coords, 0),
            [batch_size] + [1] * spatial_dim + [1])
        print(spatial_coords.shape, I.shape)

        # concatenating spatial coordinates and features
        # (and squeeze spatially)
        # for the appearance kernel
        bilateral_coords = tf.reshape(
            tf.concat([spatial_coords / self._alpha, I / self._beta], -1),
            [batch_size, -1, n_feat + spatial_dim])
        # for the smoothness kernel
        spatial_coords = tf.reshape(
            spatial_coords / self._gamma, [batch_size, -1, spatial_dim])

        # Build permutohedral structures for smoothing
        permutohedrals = [
            permutohedral_prepare(coords)
            for coords in (bilateral_coords, spatial_coords)]

        # trainable compatibility matrix mu (initialised as identity)
        mu_shape = [1] * spatial_dim + [nCh, nCh]
        mu_init = np.reshape(np.eye(nCh), mu_shape)
        mu = tf.get_variable(
            'Compatibility', initializer=tf.constant(mu_init, dtype=tf.float32))

        # trainable kernel weights
        weight_shape = [1] * spatial_dim + [1, nCh]
        kernel_weights = [tf.get_variable(
            'FilterWeights{}'.format(idx),
            shape=weight_shape, initializer=tf.ones_initializer())
            for idx, k in enumerate(permutohedrals)]

        H1 = U
        for t in range(self._T):
            H1 = ftheta(U, H1, permutohedrals, mu, kernel_weights,
                        name='{}{}'.format(self.name, t))
        return H1


def ftheta(U, H1, permutohedrals, mu, kernel_weights, name):
    """
    A mean-field update

    :param U: the unary potentials (before softmax)
    :param H1: the previous mean-field approximation to be updated
    :param permutohedrals: fixed position vectors for fast filtering
    :param mu: compatibility function
    :param kernel_weights: weights apperance/smoothness kernels
    :param name: layer name
    :return: updated mean-field distribution
    """
    batch_size = U.shape.as_list()[0]
    nCh = U.shape.as_list()[-1]

    # Message Passing
    data = tf.reshape(tf.nn.softmax(H1), [batch_size, -1, nCh])
    Q1 = [None] * len(permutohedrals)
    with tf.device('/cpu:0'):
        for idx, permutohedral in enumerate(permutohedrals):
            Q1[idx] = tf.reshape(
                permutohedral_gen(permutohedral, data, name + str(idx)),
                U.shape)

    # Weighting Filter Outputs
    Q2 = tf.add_n([Q1 * w for Q1, w in zip(Q1, kernel_weights)])

    # Compatibility Transform
    spatial_dim = infer_spatial_rank(U)
    assert spatial_dim == 2 or 3, \
        'Currently CRFAsRNNLayer supports 2D/3D images.'
    full_stride = expand_spatial_params(1, spatial_dim)
    Q3 = tf.nn.convolution(input=Q2,
                           filter=mu,
                           strides=full_stride,
                           padding='SAME')
    # Adding Unary Potentials
    Q4 = U - Q3
    # output logits, not the softmax
    return Q4


def permutohedral_prepare(position_vectors):
    """
    Splatting the position vector.

    :param position_vectors: N x d position
    :return: barycentric weights, blur points in the hyperplane
    """
    batch_size = int(position_vectors.shape[0])
    nCh = int(position_vectors.shape[-1])
    n_voxels = int(position_vectors.shape.num_elements()) // batch_size // nCh
    # reshaping batches and voxels into one dimension
    # means we can use 1D gather and hashing easily
    position_vectors = tf.reshape(position_vectors, [-1, nCh])

    # Generate position vectors in lattice space
    x = position_vectors * (np.sqrt(2. / 3.) * (nCh + 1))

    # Embed in lattice space
    inv_std_dev = np.sqrt(2 / 3.) * (nCh + 1)
    scale_factor = [
        inv_std_dev / np.sqrt((i + 1) * (i + 2)) for i in range(nCh)]
    Ex = [None] * (nCh + 1)
    Ex[nCh] = -nCh * x[:, nCh - 1] * scale_factor[nCh - 1]
    for dit in range(nCh - 1, 0, -1):
        Ex[dit] = Ex[dit + 1] - \
                  dit * x[:, dit - 1] * scale_factor[dit - 1] + \
                  (dit + 2) * x[:, dit] * scale_factor[dit]
    Ex[0] = 2 * x[:, 0] * scale_factor[0] + Ex[1]
    Ex = tf.stack(Ex, 1)

    # Compute coordinates
    # Get closest remainder-0 point
    v = tf.to_int32(tf.round(Ex / float(nCh + 1)))
    rem0 = v * (nCh + 1)
    sumV = tf.reduce_sum(v, 1, True)

    # Find the simplex we are in and store it in rank
    # (where rank describes what position coordinate i has
    # in the sorted order of the features values)
    di = Ex - tf.to_float(rem0)
    # This can be done more efficiently
    # if necessary following the permutohedral paper
    _, index = tf.nn.top_k(di, nCh + 1, sorted=True)
    _, rank = tf.nn.top_k(-index, nCh + 1, sorted=True)

    # if the point doesn't lie on the plane (sum != 0) bring it back
    rank = tf.to_int32(rank) + sumV
    addMinusSub = tf.to_int32(rank < 0) * (nCh + 1) - \
                  tf.to_int32(rank >= nCh + 1) * (nCh + 1)
    rank = rank + addMinusSub
    rem0 = rem0 + addMinusSub

    # Compute the barycentric coordinates (p.10 in [Adams et al 2010])
    v2 = (Ex - tf.to_float(rem0)) / float(nCh + 1)
    # the barycentric coordinates are
    # v_sorted - v_sorted[...,[-1,1:-1]] + [1,0,0,...]
    # CRF2RNN uses the calculated ranks to get v2 sorted in O(nCh) time
    # We cheat here by using the easy to implement
    # but slower method of sorting again in O(nCh log nCh)
    # we might get this even more efficient
    # if we correct the original sorted data above
    # v_sortedDesc, _ = tf.nn.top_k(v2, nCh + 1, sorted=True)
    # v_sorted = tf.reverse(v_sortedDesc, [1])

    v_sorted = tf.contrib.framework.sort(v2)
    v_sorted.set_shape(v2.shape)
    barycentric = \
        v_sorted - tf.concat([v_sorted[:, -1:] - 1., v_sorted[:, :nCh]], 1)

    # Compute all vertices and their offset
    # canonical simplex (p.4 in [Adams et al 2010])
    canonical = \
        [[i] * (nCh + 1 - i) + [i - nCh - 1] * i for i in range(nCh + 1)]

    def _simple_hash(key):
        # WARNING: This hash function does not guarantee
        # uniqueness of different position_vectors
        hashVector = np.power(
            int(np.floor(np.power(tf.int64.max, 1. / (nCh + 2)))),
            [range(1, nCh + 1)])
        hashVector = tf.constant(hashVector, dtype=tf.int64)
        return tf.reduce_sum(tf.to_int64(key) * hashVector, 1)

    hashtable = tf.contrib.lookup.MutableDenseHashTable(
        tf.int64, tf.int64,
        default_value=tf.constant([-1] * nCh, dtype=tf.int64),
        empty_key=-1,
        initial_num_buckets=8,
        checkpoint=False)
    indextable = tf.contrib.lookup.MutableDenseHashTable(
        tf.int64, tf.int64,
        default_value=0,
        empty_key=-1,
        initial_num_buckets=8,
        checkpoint=False)

    insertOps = []
    keys = [None] * (nCh + 1)
    i64keys = [None] * (nCh + 1)
    for scit in range(nCh + 1):
        keys[scit] = tf.gather(canonical[scit], rank[:, :-1]) + rem0[:, :-1]
        i64keys[scit] = _simple_hash(keys[scit])
        insertOps.append(
            hashtable.insert(i64keys[scit], tf.to_int64(keys[scit])))

    with tf.control_dependencies(insertOps):
        fusedI64Keys, fusedKeys = hashtable.export()
        fusedKeys = tf.boolean_mask(
            fusedKeys, tf.not_equal(fusedI64Keys, -1))
        fusedI64Keys = tf.boolean_mask(
            fusedI64Keys, tf.not_equal(fusedI64Keys, -1))

    range_id = tf.range(
        1, tf.size(fusedI64Keys, out_type=tf.int64) + 1, dtype=tf.int64)
    insertIndices = indextable.insert(
        fusedI64Keys, tf.expand_dims(tf.transpose(range_id), 1))
    blurNeighbours1 = [None] * (nCh + 1)
    blurNeighbours2 = [None] * (nCh + 1)
    indices = [None] * (nCh + 1)
    with tf.control_dependencies([insertIndices]):
        for dit in range(nCh + 1):
            offset = [nCh if i == dit else -1 for i in range(nCh)],
            offset = tf.constant(offset, dtype=tf.int64)
            blurNeighbours1[dit] = \
                indextable.lookup(_simple_hash(fusedKeys + offset))
            blurNeighbours2[dit] = \
                indextable.lookup(_simple_hash(fusedKeys - offset))
            batch_index = tf.meshgrid(
                tf.range(batch_size), tf.zeros([n_voxels], dtype=tf.int32))
            batch_index = tf.reshape(batch_index[0], [-1])
            # where in the splat variable each simplex vertex is
            indices[dit] = tf.stack(
                [tf.to_int32(indextable.lookup(i64keys[dit])), batch_index], 1)
    return barycentric, blurNeighbours1, blurNeighbours2, indices


def permutohedral_compute(data_vectors,
                          barycentric,
                          blurNeighbours1,
                          blurNeighbours2,
                          indices,
                          name,
                          reverse):
    """
    Gaussian blur and slice

    :param data_vectors:
    :param barycentric:
    :param blurNeighbours1:
    :param blurNeighbours2:
    :param indices:
    :param name:
    :param reverse:
    :return:
    """

    batch_size = tf.shape(data_vectors)[0]
    numSimplexCorners = int(barycentric.shape[-1])
    nCh = numSimplexCorners - 1
    nChData = tf.shape(data_vectors)[-1]
    data_vectors = tf.reshape(data_vectors, [-1, nChData])
    # Convert to homogenous coordinates
    data_vectors = tf.concat(
        [data_vectors, tf.ones_like(data_vectors[:, 0:1])], 1)

    # Splatting
    with tf.variable_scope(name):
        splat = tf.contrib.framework.local_variable(
            tf.ones([0, 0]), validate_shape=False, name='splatbuffer')

    with tf.control_dependencies([splat.initialized_value()]):
        initial_splat = tf.zeros(
            [tf.shape(blurNeighbours1[0])[0] + 1, batch_size, nChData + 1])
        resetSplat = tf.assign(splat, initial_splat, validate_shape=False)

    # This is needed to force tensorflow to update the cache
    with tf.control_dependencies([resetSplat]):
        uncachedSplat = splat.read_value()

    for scit in range(numSimplexCorners):
        data = data_vectors * barycentric[:, scit:scit + 1]
        with tf.control_dependencies([uncachedSplat]):
            splat = tf.scatter_nd_add(splat, indices[scit], data)

    # Blur with 1D kernels
    with tf.control_dependencies([splat]):
        blurred = [splat]
        order = range(nCh, -1, -1) if reverse else range(nCh + 1)
        for dit in order:
            with tf.control_dependencies([blurred[-1]]):
                b1 = 0.5 * tf.gather(blurred[-1], blurNeighbours1[dit])
                b2 = blurred[-1][1:, :, :]
                b3 = 0.5 * tf.gather(blurred[-1], blurNeighbours2[dit])
                blurred.append(
                    tf.concat([blurred[-1][0:1, :, :], b2 + b1 + b3], 0))

    # Alpha is a magic scaling constant from CRFAsRNN code
    alpha = 1. / (1. + np.power(2., -nCh))
    normalized = blurred[-1][:, :, :-1] / blurred[-1][:, :, -1:]

    # Slice
    sliced = tf.gather_nd(normalized, indices[0]) * barycentric[:, 0:1] * alpha
    for scit in range(1, numSimplexCorners):
        sliced = sliced + \
                 tf.gather_nd(normalized, indices[scit]) * \
                 barycentric[:, scit:scit + 1] * alpha
    return sliced


def py_func_with_grads(func, inp, Tout, stateful=True, name=None, grad=None):
    """
    To get this to work with automatic differentiation
    we use a hack attributed to Sergey Ioffe
    mentioned here: http://stackoverflow.com/questions/36456436

    Define custom py_func_with_grads which takes also a grad op as argument:
    from https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342

    :param func:
    :param inp:
    :param Tout:
    :param stateful:
    :param name:
    :param grad:
    :return:
    """
    # Need to generate a unique name to avoid duplicates:
    import uuid
    rnd_name = 'PyFuncGrad' + str(uuid.uuid4())
    tf.logging.info('CRFasRNN layer iteration {}'.format(rnd_name))
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    with tf.get_default_graph().gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def gradientStub(data_vectors,
                 barycentric,
                 blurNeighbours1,
                 blurNeighbours2,
                 indices,
                 name):
    """
    This is a stub operator whose purpose is
    to allow us to overwrite the gradient.
    The forward pass gives zeros and
    the backward pass gives the correct gradients
    for the permutohedral_compute function

    :param data_vectors:
    :param barycentric:
    :param blurNeighbours1:
    :param blurNeighbours2:
    :param indices:
    :param name:
    :return:
    """

    def _dummy_wrapper(*_unused):
        return np.float32(0.0)

    def _permutohedral_grad_wrapper(op, grad):
        # Differentiation can be done using permutohedral lattice
        # with gaussion filter order reversed
        filtering_grad = permutohedral_compute(
            grad, op.inputs[1], op.inputs[2], op.inputs[3], op.inputs[4],
            name, reverse=True)
        return [filtering_grad] + [tf.zeros_like(i) for i in op.inputs[1:]]

    partial_grads_func = py_func_with_grads(
        _dummy_wrapper,
        [data_vectors, barycentric, blurNeighbours1, blurNeighbours2, indices],
        [tf.float32],
        name=name,
        grad=_permutohedral_grad_wrapper)

    return partial_grads_func


def permutohedral_gen(permutohedral, data_vectors, name):
    """
    a wrapper combines permutohedral_compute and a customised gradient op.

    :param permutohedral:
    :param data_vectors:
    :param name:
    :return:
    """
    barycentric, blurNeighbours1, blurNeighbours2, indices = permutohedral
    backward_branch = gradientStub(
        data_vectors,
        barycentric,
        blurNeighbours1,
        blurNeighbours2,
        indices,
        name)
    forward_branch = permutohedral_compute(
        data_vectors,
        barycentric,
        blurNeighbours1,
        blurNeighbours2,
        indices,
        name,
        reverse=False)
    forward_branch = tf.reshape(forward_branch, data_vectors.shape)
    return backward_branch + tf.stop_gradient(forward_branch)
