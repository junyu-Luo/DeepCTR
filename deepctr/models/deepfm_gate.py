# -*- coding:utf-8 -*-

from itertools import chain

import tensorflow as tf

from ..feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM
from ..layers.utils import concat_func, add_func, combined_dnn_input


def DNNGATE(linear_feature_columns, dnn_feature_columns, fm_group=(DEFAULT_GROUP_NAME,),
            gate_feat=('u_user_kaihei_reg_time_day')
            , dnn_hidden_units=(256, 128, 64,), gate_hidden_units=(32,),
            l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_embedding_uid=0.001, l2_reg_dnn=0, seed=1024,
            dnn_dropout=0,
            dnn_activation='relu', gate_activation='sigmoid', dnn_use_bn=False, task='binary', if_fm=False,if_linear=True):
    features, process_features, arithmetic_features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    # 修改部分代码适配gate特征输入
    group_embedding_dict, dense_value_list, gate_embedding_dict, gate_dense_value_list = input_from_feature_columns(
        process_features, dnn_feature_columns, l2_reg_embedding,
        seed, support_group=True, gate_feat=gate_feat, l2_reg_embedding_uid=l2_reg_embedding_uid)
    # 主网络部分
    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)

    # 子网络部分
    gate_input = combined_dnn_input(list(chain.from_iterable(
        gate_embedding_dict.values())), gate_dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)
    gate_output = DNN(gate_hidden_units, gate_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(gate_input)
    if gate_activation == 'sigmoid':
        gate_output = tf.keras.layers.Lambda(lambda x: tf.multiply(x, 2),
                                             name='gate_mul_2')(gate_output)
    gate_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(gate_output)
    dnn_gate_logit = tf.keras.layers.Multiply(name='dnn_gate_logit')([dnn_logit, gate_logit])

    final=[dnn_gate_logit]
    # linear & fm
    if if_linear:
        linear_logit = get_linear_logit(process_features, linear_feature_columns, arithmetic_features, seed=seed,
                                        prefix='linear', l2_reg=l2_reg_linear)
        final.append(linear_logit)
    if if_fm:
        fm_logit = add_func([FM()(concat_func(v, axis=1))
                             for k, v in group_embedding_dict.items() if k in fm_group])
        final.append(fm_logit)

    final_logit = add_func(final)
    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
