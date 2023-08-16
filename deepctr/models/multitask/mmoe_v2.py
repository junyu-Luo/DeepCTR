import sys, os, site, glob, gc
import glob
import random
import datetime, pytz, time
from multiprocessing import Pool, cpu_count, Process, Queue
import math
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Lambda, Multiply

from ...layers.interaction import FM
# from ...feature_column import build_input_features, input_from_feature_columns
from ...layers.core import PredictionLayer, DNN
from ...layers.utils import combined_dnn_input, reduce_sum
from ...feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
from ...layers.convlayer import ArithmeticLayer, VocabLayer, AutoDis


# # Util function
def sequence_embs_combiner(fc, emb, mask_init, relavant_obj=None, method=None):
    if method is None:
        method = fc.combiner
    maxlen = fc.maxlen
    fn = fc.name

    if method in ('avg', 'mean', 'sum', 'max', 'test_mean'):
        mask = tf.cast(mask_init, tf.float32)
        if method == 'sum':
            mask = tf.expand_dims(mask, axis=1)
            sumed = tf.matmul(mask, emb)
            merged_emb = sumed
        elif method in ('avg', 'mean'):
            mask = tf.expand_dims(mask, axis=1)
            sumed = tf.matmul(mask, emb)
            avged = sumed / (tf.reduce_sum(mask, axis=-1, keepdims=True) + 1e-6)
            merged_emb = avged
        elif method == 'test_mean':
            #             mask = tf.expand_dims(mask, axis=1)
            user_behavior_length = tf.reduce_sum(mask, axis=-1, keepdims=True)
            print('user_behavior_length:', user_behavior_length.shape)
            mask = tf.expand_dims(mask, axis=2)
            print('mask:', mask.shape)
            emb_size = emb.shape[-1]
            print('emb_size:', emb_size)
            mask = tf.tile(mask, [1, 1, emb_size])
            print('mask:', mask.shape)
            print('emb:', emb.shape)
            sumed = tf.reduce_sum(emb * mask, 1, keepdims=False)
            print('sumed:', sumed.shape)
            avged = tf.divide(sumed, tf.cast(user_behavior_length, tf.float32) + 1e-6)
            print('avged:', avged.shape)
            avged = tf.expand_dims(avged, axis=1)
            merged_emb = avged
        elif method == 'max':
            user_behavior_length = tf.reduce_sum(mask, axis=-1, keepdims=True)
            mask = tf.expand_dims(mask, axis=2)
            emb_size = emb.shape[-1]
            mask = tf.tile(mask, [1, 1, emb_size])
            hist = emb - (1 - mask) * 1e9
            merged_emb = tf.reduce_max(hist, axis=1, keepdims=True)
            print('merged_emb:', merged_emb.shape)
        else:
            raise ValueError('method error')
    elif method == 'weighted_average':
        x = tf.strings.split(relavant_obj, ',', name=f'{fn}_weight_split').to_tensor('0', shape=[None, 1, maxlen],
                                                                                     name=f'{fn}_to_tensor')
        x = tf.compat.v1.string_to_number(tf.squeeze(x, axis=1), out_type=tf.float32, name=f'{fn}_to_number')
        mask = tf.expand_dims(x, axis=1)
        sumed = tf.matmul(mask, emb)
        merged_emb = sumed / (tf.reduce_sum(mask, axis=-1, keepdims=True) + 1e-6)
    elif method == 'attention':
        Q = layers.Dense(emb.shape[-1])(emb)
        K = layers.Dense(emb.shape[-1])(emb)
        V = layers.Dense(emb.shape[-1])(emb)
        embs = tf.keras.layers.Attention(use_scale=False)([Q, V, K], mask=[None, mask_init])
        _merged_emb = embs[:, :1, :]
        merged_emb = tf.keras.layers.Dense(4 * _merged_emb.shape[-1], activation='relu')(_merged_emb)
        merged_emb = tf.keras.layers.Dense(_merged_emb.shape[-1])(merged_emb)
    elif method == 'attention_no_mask':
        Q = layers.Dense(emb.shape[-1])(emb)
        K = layers.Dense(emb.shape[-1])(emb)
        V = layers.Dense(emb.shape[-1])(emb)
        embs = tf.keras.layers.Attention(use_scale=False)([Q, V, K])
        _merged_emb = embs[:, :1, :]
        merged_emb = tf.keras.layers.Dense(4 * _merged_emb.shape[-1], activation='relu')(_merged_emb)
        merged_emb = tf.keras.layers.Dense(_merged_emb.shape[-1])(merged_emb)
    elif method == 'attention_dot':
        attention_score = tf.reduce_sum(tf.multiply(emb, relavant_obj), axis=2) / (emb.get_shape()[-1] ** 0.5)
        score = tf.where(mask_init, attention_score, 1e-12)
        score = tf.nn.softmax(score)
        merged_emb = tf.matmul(tf.expand_dims(score, axis=1), emb)
    elif method == 'attention_din':
        # from tensorflow.keras import backend as K
        attention_hidden_units = (80, 40, 1)
        attention_activation = 'sigmoid'
        relavant_obj = tf.tile(relavant_obj, [1, maxlen, 1],
                               name=f'{fn}_din_attention_tile')  # (batch_size, maxlen * hidden_units)
        concat = K.concatenate([relavant_obj, emb, relavant_obj - emb, relavant_obj * emb],
                               axis=2)  # (batch_size, maxlen, emb_dim * 4) 点乘
        for i in range(len(attention_hidden_units)):
            activation = None if i == 2 else attention_activation
            outputs = layers.Dense(attention_hidden_units[i], activation=activation,
                                   name=f'{fn}_din_attention_dense_{i}')(concat)
            concat = outputs

        attention_score = tf.squeeze(outputs / (emb.get_shape()[-1] ** 0.5))
        score = tf.nn.softmax(tf.where(mask_init, attention_score, 1e-12), name=f'{fn}_din_attention_softmax')
        merged_emb = tf.matmul(tf.expand_dims(score, axis=1), emb)
    else:
        raise ValueError(
            'method must be in (mean, sum, max, min, attention, attention_no_mask, attention_dot, attention_din)')
    return merged_emb


def get_embeddings(feature_columns, inputs_dict, lookup_layer_dict, emb_layer_dict, return_dict=False):
    feature_dict = {fc.name: fc for fc in feature_columns}
    emb_dict = dict()
    for fn in emb_layer_dict.keys():
        if isinstance(feature_dict[fn], SparseFeat):
            if lookup_layer_dict[fn].table.key_dtype == tf.string and feature_dict[fn].dtype == 'int32':
                str_index = tf.strings.as_string(inputs_dict[fn], name=f'key_to_string_{fn}')
                sparse_index = lookup_layer_dict[fn](str_index)
            else:
                sparse_index = lookup_layer_dict[fn](inputs_dict[fn])
            emb = emb_layer_dict[fn](sparse_index)
            emb_dict[fn] = emb
        if isinstance(feature_dict[fn], VarLenSparseFeat):
            #             split_input = tf.keras.layers.Lambda(lambda x:tf.strings.split(x,','))(input_layer)[fn]

            x = tf.strings.split(inputs_dict[fn], ',', maxsplit=feature_dict[fn].maxlen, name=f'{fn}_split')
            x = tf.squeeze(x.to_tensor('', shape=[None, 1, feature_dict[fn].maxlen], name=f'{fn}_to_tensor'), axis=1)
            sparse_index = lookup_layer_dict[fn](x)
            emb = emb_layer_dict[fn](sparse_index)
            pooling_method = feature_dict[fn].combiner
            relavant_obj = None
            if pooling_method == 'weighted_average':
                if inputs_dict.get(fn + '_weight', None) is None:
                    raise ValueError(f'there is no weight column for feature {fn}')
                else:
                    relavant_obj = inputs_dict[fn + '_weight']

            mask_init = tf.not_equal(sparse_index, tf.constant(0, dtype=lookup_layer_dict[fn].table.value_dtype))
            if isinstance(pooling_method, list):
                combiner_emb = sequence_embs_combiner(feature_dict[fn], emb, mask_init, relavant_obj,
                                                      method=pooling_method[0])
                for method in pooling_method[1:]:
                    emb = sequence_embs_combiner(feature_dict[fn], emb, mask_init, relavant_obj, method=method)
                    combiner_emb = tf.concat([combiner_emb, emb], axis=-1)
            else:
                combiner_emb = sequence_embs_combiner(feature_dict[fn], emb, mask_init, relavant_obj,
                                                      method=pooling_method)
            emb_dict[fn] = combiner_emb
    return list(emb_dict.values()) if not return_dict else emb_dict


def MMOE_v2(feature_columns, share_embedding_list, fm_fea_groups=None,
            use_linear=True, use_fm=False, use_dnn=True,
            linear_reg=1e-5,
            num_experts=3, expert_dnn_hidden_units=(256, 128), tower_dnn_hidden_units=(64,),
            gate_dnn_hidden_units=(), l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
            dnn_activation='relu',
            dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'),
            ):

    share_embedding = {}
    for features in share_embedding_list:
        # put the feature used first in the first place
        if features:
            for fea in features[1:]:
                share_embedding[fea] = features[0]  # 'i_u_p_gender': 'u_p_gender'

    assert (use_linear or use_fm or use_dnn)
    use_dense = any(True if isinstance(fc, DenseFeat) else False for fc in feature_columns)
    use_sparse = any(True if isinstance(fc, SparseFeat) else False for fc in feature_columns)
    logits_list = list()

    # # inputs
    inputs_dict = dict()
    dense_op_layer_dict = dict()
    lookup_layer_dict = dict()
    emb_layer_dict = dict()
    linear_emb_layer_dict = dict()
    for fc in feature_columns:
        if isinstance(fc, DenseFeat):
            inputs_dict[fc.name] = layers.Input(name=fc.name, shape=(1,), dtype=fc.dtype)
            if fc.operation is not None:
                dense_op_layer_dict[fc.name] = ArithmeticLayer(fc.operation, fc.statistics,
                                                               name=f'{fc.operation}_{fc.name}')
        elif isinstance(fc, SparseFeat):
            inputs_dict[fc.name] = layers.Input(name=fc.name, shape=(1,), dtype=fc.dtype)
            if not share_embedding.get(fc.name, None) or not lookup_layer_dict.get(share_embedding[fc.name], None):
                lookup_layer_dict[fc.name] = VocabLayer(fc.vocab, name=f'lookup_{fc.name}')
                emb_layer_dict[fc.name] = layers.Embedding(1 + fc.vocabulary_size, fc.embedding_dim,
                                                           name=f'emb_{fc.name}')
                linear_emb_layer_dict[fc.name] = layers.Embedding(1 + fc.vocabulary_size, 1,
                                                                  name=f'linear_emb_{fc.name}',
                                                                  embeddings_regularizer=l2(linear_reg))
            else:
                lookup_layer_dict[fc.name] = lookup_layer_dict[share_embedding[fc.name]]
                emb_layer_dict[fc.name] = emb_layer_dict[share_embedding[fc.name]]
                linear_emb_layer_dict[fc.name] = layers.Embedding(1 + fc.vocabulary_size, 1,
                                                                  name=f'linear_emb_{fc.name}',
                                                                  embeddings_regularizer=l2(linear_reg))
        elif isinstance(fc, VarLenSparseFeat):
            inputs_dict[fc.name] = layers.Input(name=fc.name, shape=(1,), dtype=fc.dtype)
            lookup_layer_dict[fc.name] = VocabLayer(fc.vocab, name=f'lookup_{fc.name}')
            emb_layer_dict[fc.name] = layers.Embedding(1 + fc.vocabulary_size, fc.embedding_dim, name=f'emb_{fc.name}')

        #     inputs_dict[fc.name] = layers.Input(name=fc.name, shape=(1,), dtype=fc.dtype)
        #     if not fc.is_weight:
        #         if not share_embedding.get(fc.name, None) or not lookup_layer_dict.get(share_embedding[fc.name], None):
        #             lookup_layer_dict[fc.name] = VocabLayer(fc.vocab, name=f'lookup_{fc.name}')
        #             emb_layer_dict[fc.name] = layers.Embedding(1 + fc.vocabulary_size, fc.embedding_dim, name=f'emb_{fc.name}')
        #         #                     linear_emb_layer_dict[fc.name] = layers.Embedding(1+fc.vocabulary_size, 1, name=f'linear_emb_{fc.name}', embeddings_regularizer=l2(linear_reg))
        #         else:
        #             lookup_layer_dict[fc.name] = lookup_layer_dict[share_embedding[fc.name]]
        #             emb_layer_dict[fc.name] = emb_layer_dict[share_embedding[fc.name]]
        # #                     linear_emb_layer_dict[fc.name] = layers.Embedding(1+fc.vocabulary_size, 1, name=f'linear_emb_{fc.name}', embeddings_regularizer=l2(linear_reg))
        else:
            raise ValueError('feature_columns contains unknown Featrue-Type')

    if use_dense:
        normal_dense_list = [
            AutoDis(num_buckets=16, emb_dim=8, keepdim=True)(dense_op_layer_dict[fc.name](inputs_dict[fc.name]))
            if fc.operation is not None else AutoDis(num_buckets=16, emb_dim=8, keepdim=True)(inputs_dict[fc.name])
            for fc in feature_columns if isinstance(fc, DenseFeat)]
        concat_denses = layers.Concatenate(name='concat_denses')(normal_dense_list)
    if use_sparse:
        emb_list = get_embeddings(feature_columns, inputs_dict, lookup_layer_dict, emb_layer_dict)

    # Linear
    if use_linear:
        if use_dense:
            linear_dense_logits = layers.Dense(1, kernel_regularizer=l2(linear_reg), name='linear_dense_logits')(
                concat_denses)
        if use_sparse:
            weight_sparses = get_embeddings(feature_columns, inputs_dict, lookup_layer_dict, linear_emb_layer_dict)
            linear_sparse_logits = layers.Flatten(name='linear_sparse_logits')(
                layers.Add(name='linear_sparse_logits_')(weight_sparses))
        if use_dense and use_sparse:
            linear_logits = layers.Add(name='linear_logits')([linear_dense_logits, linear_sparse_logits])
        elif use_dense:
            linear_logits = linear_dense_logits
        elif use_sparse:
            linear_logits = linear_sparse_logits
        else:
            raise Exception('no linear_logits')
        logits_list.append(linear_logits)

    # fm
    if use_fm and use_sparse:
        emb_dict = get_embeddings(feature_columns, inputs_dict, lookup_layer_dict, emb_layer_dict, return_dict=True)
        for i, fea_group in enumerate(fm_fea_groups):
            fm_emb_list = [emb_dict[fea] for fea in fea_group]
            embs_concat = layers.Concatenate(axis=1, name=f'embs_concat_{i}')(fm_emb_list)
            fm_logits = FM(name=f'fm_{i}')(embs_concat)
            logits_list.append(fm_logits)

    # dnn
    if use_sparse and use_dense:
        concat_sparse = layers.Flatten(name='embs_flatten')(
            layers.Concatenate(axis=-1, name='embs_concat_last')(emb_list))
        dnn_inputs = layers.Concatenate(name='dnn_inputs')([concat_denses, concat_sparse])
    elif use_sparse:
        dnn_inputs = layers.Flatten(name='embs_flatten')(layers.Concatenate(axis=-1, name='embs_concat_last')(emb_list))
    elif use_dense:
        dnn_inputs = concat_denses
    else:
        raise Exception('no dnn_inputs')

    num_tasks = len(task_names)

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")

    if num_experts <= 1:
        raise ValueError("num_experts must be greater than 1")

    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")

    for task_type in task_types:

        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))

    # features = build_input_features(dnn_feature_columns)

    inputs_list = list(inputs_dict.values())

    # sparse_embedding_list, dense_value_list = input_from_feature_columns(process_features,dnn_feature_columns,

    #                                                                      l2_reg_embedding, seed)

    # dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    # build expert layer

    expert_outs = []

    for i in range(num_experts):
        expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,

                             name='expert_' + str(i))(dnn_inputs)

        expert_outs.append(expert_network)

    expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(expert_outs)  # None,num_experts,dim

    mmoe_outs = []

    for i in range(num_tasks):  # one mmoe layer: nums_tasks = num_gates

        # build gate layers

        gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,

                         name='gate_' + task_names[i])(dnn_inputs)

        gate_out = Dense(num_experts, use_bias=False, activation='softmax',

                         name='gate_softmax_' + task_names[i])(gate_input)

        gate_out = Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

        # gate multiply the expert

        gate_mul_expert = Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),

                                 name='gate_mul_expert_' + task_names[i])([expert_concat, gate_out])

        mmoe_outs.append(gate_mul_expert)

    task_outs = []

    for task_type, task_name, mmoe_out in zip(task_types, task_names, mmoe_outs):
        # build tower layer

        tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,

                           name='tower_' + task_name)(mmoe_out)

        logit = Dense(1, use_bias=False)(tower_output)

        output = PredictionLayer(task_type, name=task_name)(logit)

        task_outs.append(output)

    model = Model(inputs=inputs_list, outputs=task_outs)

    # else:
    #     print(f'Error: model_name was not defined: {model_name}, it should be one of (DNN, ESMM, PLE)')
    #     model = None

    return model

    # model.compile(optimizer=optimizer,
    #               loss={label: loss for label in labels},
    #               metrics={label: metrics for label in labels})
