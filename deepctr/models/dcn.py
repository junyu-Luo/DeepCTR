# -*- coding:utf-8 -*-

from itertools import chain

import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM, CrossNet, CrossNetMix
from ..layers.utils import concat_func, add_func, combined_dnn_input


def DCN(linear_feature_columns, dnn_feature_columns, fm_group=(DEFAULT_GROUP_NAME,), dnn_hidden_units=(256, 128, 64),
        l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0, l2_reg_cross=1e-5,
        dnn_activation='relu', dnn_use_bn=False, task='binary', cross_layer=3, if_fm=False, if_linear=True,
        parameterization='matrix', cross_method='parallel', if_mix=True, low_rank=32, num_experts=4):
    features, process_features, arithmetic_features = build_input_features(
        dnn_feature_columns)

    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(process_features, dnn_feature_columns,
                                                                        l2_reg_embedding,
                                                                        seed, support_group=True)

    shared_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)
    # use DCN-MIX
    # default low_rank=32, num_experts=4, layer_num=2
    if if_mix:
        cross_output = CrossNetMix(layer_num=cross_layer, num_experts=num_experts, low_rank=low_rank,
                                   l2_reg=l2_reg_cross)(shared_input)
    else:
        cross_output = CrossNet(layer_num=cross_layer, parameterization=parameterization, l2_reg=l2_reg_cross)(
            shared_input)
    if cross_method == 'stacked':
        dnn_input = cross_output
        final_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    elif cross_method == 'parallel':
        dnn_input = shared_input
        dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
        final_output = tf.keras.layers.Concatenate()([cross_output, dnn_output])
    else:
        raise ValueError("cross_method should be 'stacked' or 'parallel'")
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(final_output)

    final = [dnn_logit]
    linear_logit = get_linear_logit(process_features, linear_feature_columns, arithmetic_features, seed=seed,
                                    prefix='linear', l2_reg=l2_reg_linear)
    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])
    if if_linear: final.append(linear_logit)
    if if_fm: final.append(fm_logit)
    final_logit = add_func(final)

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
