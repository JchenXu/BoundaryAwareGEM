"""
PointConv operation
Author: Wenxuan Wu
Date: July 2018
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/3d_interpolation'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/grouping'))
from tf_interpolate import three_nn, three_interpolate
import tf_grouping
import pointconv_util
import tf_util

def weight_net_hidden(xyz, hidden_units, scope, is_training, bn_decay=None, weight_decay = None, activation_fn=tf.nn.relu):

    with tf.variable_scope(scope) as sc:
        net = xyz
        for i, num_hidden_units in enumerate(hidden_units):
            net = tf_util.conv2d(net, num_hidden_units, [1, 1],
                                padding = 'VALID', stride=[1, 1],
                                bn = True, is_training = is_training, activation_fn=activation_fn,
                                scope = 'wconv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

            #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='wconv_dp%d'%(i))
    return net

def weight_net(xyz, hidden_units, scope, is_training, bn_decay=None, weight_decay = None, activation_fn=tf.nn.relu):

    with tf.variable_scope(scope) as sc:
        net = xyz
        for i, num_hidden_units in enumerate(hidden_units):
            if i != len(hidden_units) -1:
                net = tf_util.conv2d(net, num_hidden_units, [1, 1],
                                    padding = 'VALID', stride=[1, 1],
                                    bn = True, is_training = is_training, activation_fn=activation_fn,
                                    scope = 'wconv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)
            else:
                net = tf_util.conv2d(net, num_hidden_units, [1, 1],
                                    padding = 'VALID', stride=[1, 1],
                                    bn = False, is_training = is_training, activation_fn=None,
                                    scope = 'wconv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)
            #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='wconv_dp%d'%(i))
    return net

def nonlinear_transform(data_in, mlp, scope, is_training, bn_decay=None, weight_decay = None, activation_fn = tf.nn.relu):

    with tf.variable_scope(scope) as sc:

        net = data_in
        l = len(mlp)
        if l > 1:
            for i, out_ch in enumerate(mlp[0:(l-1)]):
                net = tf_util.conv2d(net, out_ch, [1, 1],
                                    padding = 'VALID', stride=[1, 1],
                                    bn = True, is_training = is_training, activation_fn=tf.nn.relu,
                                    scope = 'nonlinear%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

                #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp_nonlinear%d'%(i))
        net = tf_util.conv2d(net, mlp[-1], [1, 1],
                            padding = 'VALID', stride=[1, 1],
                            bn = False, is_training = is_training,
                            scope = 'nonlinear%d'%(l-1), bn_decay=bn_decay,
                            activation_fn=tf.nn.sigmoid, weight_decay = weight_decay)

    return net

def feature_encoding_layer(xyz, feature, npoint, radius, sigma, K, mlp, local_num_out_channel, is_training, bn_decay, weight_decay, scope, bn=True, use_xyz=True, boundary_label=None):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            sigma: float32 -- KDE bandwidth
            K: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    batch_size = xyz.shape[0]

    with tf.variable_scope(scope) as sc:
        num_points = xyz.get_shape()[1]
        if num_points == npoint:
            new_xyz = xyz
            sub_boundary_label = boundary_label
        else:
            if boundary_label is None:
                new_xyz = pointconv_util.sampling(npoint, xyz)
                sub_boundary_label = None
            else:
                new_xyz, sub_boundary_label = pointconv_util.sampling_with_boundary_label(npoint, xyz, boundary_label)

        if boundary_label is None:
            pass
        else:
            tmp_boundary_label = tf.tile(tf.expand_dims(boundary_label,[-1]), [1, 1, feature.shape[2]])
            feature = feature * tmp_boundary_label

        if local_num_out_channel is not None:
            xyz_4n, feature_4n, idx_4n, _ = pointconv_util.grouping(feature, 4, xyz, xyz)
            xyz_3n = xyz_4n[:,:,1:4,:]
            local_feature = tf_util.local_feature(xyz_3n, local_num_out_channel, bn, is_training, 0, bn_decay, weight_decay)
            feature = tf.concat([feature, local_feature], axis=2)
        grouped_xyz, grouped_feature, idx, grouped_boundary_label = pointconv_util.grouping(feature, K, xyz, new_xyz, boundary_label=boundary_label)

        for i, num_out_channel in enumerate(mlp):
            if i != len(mlp) - 1:
                grouped_feature = tf_util.conv2d(grouped_feature, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1],
                                                bn=bn, is_training=is_training,
                                                scope='conv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

        if grouped_boundary_label is not None:
            tmp_grouped_boundary_label = tf.tile(tf.expand_dims(grouped_boundary_label,[-1]),[1,1,1,grouped_feature.shape[3]])
            grouped_feature = grouped_feature * tmp_grouped_boundary_label

        weight = weight_net_hidden(grouped_xyz, [32], scope = 'weight_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)

        new_points = tf.transpose(grouped_feature, [0, 1, 3, 2])
        new_points = tf.matmul(new_points, weight)
        new_points = tf_util.conv2d(new_points, mlp[-1], [1,new_points.get_shape()[2].value],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='after_conv', bn_decay=bn_decay, weight_decay = weight_decay)

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])

        return new_xyz, new_points, sub_boundary_label

def feature_encoding_layer_extra(xyz, sample_xyz, sample_boundary, feature, npoint, radius, sigma, K, mlp, local_num_out_channel, is_training, bn_decay, weight_decay, scope, bn=True, use_xyz=True, boundary_label=None):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            sigma: float32 -- KDE bandwidth
            K: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    batch_size = xyz.shape[0]

    with tf.variable_scope(scope) as sc:
        num_points = xyz.get_shape()[1]
        if num_points == npoint:
            new_xyz = xyz
            sub_boundary_label = boundary_label
        else:
            if boundary_label is None:
                #new_xyz = pointconv_util.sampling(npoint, xyz)
                new_xyz = sample_xyz
                sub_boundary_label = None
            else:
                #new_xyz, sub_boundary_label = pointconv_util.sampling_with_boundary_label(npoint, xyz, boundary_label)
                new_xyz = sample_xyz
                sub_boundary_label = sample_boundary

        if boundary_label is None:
            pass
        else:
            tmp_boundary_label = tf.tile(tf.expand_dims(boundary_label,[-1]), [1, 1, feature.shape[2]])
            feature = feature * tmp_boundary_label

        if local_num_out_channel is not None:
            xyz_4n, feature_4n, idx_4n, _ = pointconv_util.grouping(feature, 4, xyz, xyz)
            xyz_3n = xyz_4n[:,:,1:4,:]
            local_feature = tf_util.local_feature(xyz_3n, local_num_out_channel, bn, is_training, 0, bn_decay, weight_decay)
            feature = tf.concat([feature, local_feature], axis=2)
        grouped_xyz, grouped_feature, idx, grouped_boundary_label = pointconv_util.grouping(feature, K, xyz, new_xyz, boundary_label=boundary_label)

        for i, num_out_channel in enumerate(mlp):
            if i != len(mlp) - 1:
                grouped_feature = tf_util.conv2d(grouped_feature, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1],
                                                bn=bn, is_training=is_training,
                                                scope='conv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

        if grouped_boundary_label is not None:
            tmp_grouped_boundary_label = tf.tile(tf.expand_dims(grouped_boundary_label,[-1]),[1,1,1,grouped_feature.shape[3]])
            grouped_feature = grouped_feature * tmp_grouped_boundary_label

        weight = weight_net_hidden(grouped_xyz, [32], scope = 'weight_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)

        new_points = tf.transpose(grouped_feature, [0, 1, 3, 2])
        new_points = tf.matmul(new_points, weight)
        new_points = tf_util.conv2d(new_points, mlp[-1], [1,new_points.get_shape()[2].value],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='after_conv', bn_decay=bn_decay, weight_decay = weight_decay)

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])

        return new_xyz, new_points, sub_boundary_label

def feature_decoding_layer(xyz1, xyz2, points1, points2, radius, sigma, K, mlp, is_training, bn_decay, weight_decay, scope, bn=True, use_xyz = True, boundary_label=None):
    ''' Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            sigma: float32 -- KDE bandwidth
            K: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keepdims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        #setup for deConv
        if boundary_label is None:
            pass
        else:
            tmp_boundary_label = tf.tile(tf.expand_dims(boundary_label,[-1]),[1,1,interpolated_points.shape[2]])
            interpolated_points = interpolated_points * tmp_boundary_label
        grouped_xyz, grouped_feature, idx, grouped_boundary_label = pointconv_util.grouping(interpolated_points, K, xyz1, xyz1, use_xyz=use_xyz, boundary_label=boundary_label)

        weight = weight_net_hidden(grouped_xyz, [32], scope = 'decode_weight_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)

        new_points = tf.transpose(grouped_feature, [0, 1, 3, 2])
        new_points = tf.matmul(new_points, weight)
        new_points = tf_util.conv2d(new_points, mlp[0], [1,new_points.get_shape()[2].value],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='decode_after_conv', bn_decay=bn_decay, weight_decay = weight_decay)

        if points1 is not None:
            new_points1 = tf.concat(axis=-1, values=[new_points, tf.expand_dims(points1, axis = 2)]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = new_points

        for i, num_out_channel in enumerate(mlp):
            if i != 0:
                new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1

def placeholder_inputs(batch_size, num_point, channel):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    feature_pts_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, channel))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, feature_pts_pl, labels_pl

