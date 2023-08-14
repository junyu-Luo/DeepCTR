


import tensorflow as tf
from ...feature_column import  input_from_feature_columns,DEFAULT_GROUP_NAME,build_input_features,get_linear_logit
from ...layers.core import PredictionLayer, DNN
from ...layers.utils import concat_func, add_func, combined_dnn_input
from ...layers.interaction import FM
from itertools import chain


def ESM2(linear_feature_columns, dnn_feature_columns, fm_group=(DEFAULT_GROUP_NAME,),
           bottom_dnn_hidden_units=(256, 128, 64),
          l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_linear=0.00001,
         seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
         task_names=('ctr', 'ctavr','ctcvr'),if_fm=False):

    features,process_features, arithmetic_features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())
    sparse_embedding_dict, dense_value_list = input_from_feature_columns(process_features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed, support_group=True)
    sparse_embedding_list = list(chain.from_iterable(sparse_embedding_dict.values()))

    linear_logit = get_linear_logit(process_features, linear_feature_columns,arithmetic_features, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in sparse_embedding_dict.items() if k in fm_group])

    task_dnn_logit={k:None for k in task_names}
    for idx,task in enumerate(task_names):
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        dnn_output = DNN(bottom_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
        dnn_logit = tf.keras.layers.Dense(
            1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)
        if if_fm:
            final_logit = add_func([linear_logit, fm_logit, dnn_logit])
        else:
            final_logit = add_func([linear_logit, dnn_logit])
        task_dnn_logit[task]=final_logit

    # ctr 预测值
    predict_ctr=PredictionLayer('binary', name=task_names[0])(task_dnn_logit[task_names[0]])

    # ctavr 预测值
    ctacvr_output = tf.keras.layers.Multiply(name='ctacvr_output')(
        [task_dnn_logit[task_names[0]], task_dnn_logit[task_names[1]]])
    predict_ctacvr = PredictionLayer('binary', name=task_names[1])(ctacvr_output)

    # ctcvr 预测值
    # 点击&发生O行为=1-点击&发生D行为
    # oaction=1-ctacvr_output
    oaction = tf.keras.layers.Lambda(lambda x: tf.subtract(1.0, x),
                                         name='oaction')(ctacvr_output)
    # 无中间行为直接转化路径
    ctcvr1 = tf.keras.layers.Multiply(name='ctcvr1')([oaction, task_dnn_logit[task_names[2]]])
    # 有中间行为并转化
    ctcvr2 = tf.keras.layers.Multiply(name='ctcvr2')(
        [ctacvr_output, task_dnn_logit[task_names[2]]])
    ctcvr1_add_ctcvr2 = tf.keras.layers.Lambda(lambda x: tf.add(x[0], x[1]),
                                         name='ctcvr1_add_ctcvr2')([ctcvr1,ctcvr2])
    predict_ctcvr = PredictionLayer('binary', name=task_names[2])(ctcvr1_add_ctcvr2)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=[predict_ctr,predict_ctacvr,predict_ctcvr])
    return model

