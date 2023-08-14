
import tensorflow as tf

from ...feature_column import  input_from_feature_columns,DenseFeat,DEFAULT_GROUP_NAME,build_input_features,get_linear_logit
from ...layers.core import PredictionLayer, DNN
from ...layers.utils import concat_func, add_func, combined_dnn_input
from ...layers.interaction import FM
from itertools import chain


def DBMTL(linear_feature_columns, dnn_feature_columns, fm_group=(DEFAULT_GROUP_NAME,),
          tower_dnn_hidden_units=(256, 128, 64, 32),bayes_dnn_hidden_units=(32,),
          l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_linear=0.00001,
         seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
         task_names=('ctr', 'ctcvr'),if_fm=False):



    features,process_features, arithmetic_features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())
    sparse_embedding_dict, dense_value_list = input_from_feature_columns(process_features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed, support_group=True)

    sparse_embedding_list = list(chain.from_iterable(sparse_embedding_dict.values()))
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)


    linear_logit = get_linear_logit(process_features, linear_feature_columns,arithmetic_features, seed=seed,
                                    prefix='linear_{}'.format('shared'),
                                    l2_reg=l2_reg_linear)

    fm_logit = add_func([FM(name='FM_{}'.format('shared'))(concat_func(v, axis=1))
                         for k, v in sparse_embedding_dict.items() if k in fm_group])

    # towers
    tower_logit_ls={k:None for k in task_names}
    for task in task_names:
        dnn_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                        name='tower_DNN_{}'.format(task), seed=seed)(
            dnn_input)
        dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None) \
            (dnn_output)
        if if_fm:
            final_logit = add_func([linear_logit, fm_logit, dnn_logit])
        else:
            final_logit = add_func([linear_logit, dnn_logit])
        tower_logit_ls[task]=final_logit

    # Bayes
    predict_ls={k:None for k in task_names}
    tmp_logit=tower_logit_ls[task_names[0]]
    for idx,task in enumerate(tower_logit_ls):
        if idx==0:
            tmp_logit=tower_logit_ls[task]
            predict_ls[task]=PredictionLayer('binary', name=task)(tmp_logit)
        else:
            concat_input = tf.keras.layers.concatenate([tmp_logit, tower_logit_ls[task]], axis=1)
            tmp_logit=concat_input
            concat_output=tf.keras.layers.Dense(1, use_bias=False, activation=None)\
                (DNN(bayes_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(concat_input))
            # 先验知识后预测
            predict_ls[task] = PredictionLayer('binary', name=task)(concat_output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=predict_ls.values())
    return model
