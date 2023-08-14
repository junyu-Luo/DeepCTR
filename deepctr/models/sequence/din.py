# -*- coding:utf-8 -*-
import tensorflow as tf

from ...feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features,get_linear_logit,DEFAULT_GROUP_NAME
from ...layers.core import DNN, PredictionLayer
from ...layers.sequence import AttentionSequencePoolingLayer,SequencePoolingLayer
from ...layers.utils import NoMask, combined_dnn_input,add_func
from ...layers.interaction import FM
from ...inputs import create_embedding_matrix, embedding_lookup, get_dense_input,varlen_embedding_lookup, \
    get_varlen_pooling_list, mergeDict

def concat_func(inputs, axis=-1, mask=False):
    # if mask:
    #     raise Exception('mask 只有onn跟din用到了，如果用这两个模型记得把下面注释打开')
    if not mask:
        # py的map指针指向同一个，多个调用同一个混乱
        #### inputs = list(map(NoMask(), inputs))
        inputs = [NoMask()(x) for x in inputs]
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)

def TTDIN(linear_feature_columns, dnn_feature_columns, history_feature_list=None,fm_group=(DEFAULT_GROUP_NAME,), dnn_use_bn=False,
          dnn_hidden_units=(256, 128, 64), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice",
          att_weight_normalization=False, l2_reg_linear=0.00001,l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024,
          task='binary',attention_method='din',if_fm=False):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """

    if history_feature_list is None:
        assert '需要对应的key和query'
    features,process_features,arithmetic_features = build_input_features(linear_feature_columns+dnn_feature_columns)
    inputs_list = list(features.values())
    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, seed, prefix="")


    SparseFeatures = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    DenseFeatures = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    VarLenSparseFeatures = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    hist_ls=[]
    for key in history_feature_list:
        query_emb_list = embedding_lookup(embedding_dict, process_features,
                                          VarLenSparseFeatures, history_feature_list[key], history_feature_list[key],
                                          to_list=True)

        keys_emb_list = embedding_lookup(embedding_dict, process_features,
                                         VarLenSparseFeatures, [key], [key], to_list=True)


        keys_emb = concat_func(keys_emb_list, mask=True)
        query_emb = concat_func(query_emb_list, mask=True)
        # query也是变长序列
        poollayer = SequencePoolingLayer('mean', supports_masking=True)
        item_mean=poollayer(query_emb)
        if attention_method == 'din':
            hist = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
                                                 weight_normalization=att_weight_normalization, supports_masking=True)([
                item_mean, keys_emb])
            hist_ls.append(hist)
        elif attention_method == 'attention':
            hist=tf.keras.layers.Attention(use_scale=False)([item_mean, keys_emb])
            hist_ls.append(hist)


    dnn_input_emb_list = embedding_lookup(embedding_dict, process_features, SparseFeatures, to_list=True)
    dense_value_list = get_dense_input(process_features, DenseFeatures)
    deep_input_emb = concat_func(dnn_input_emb_list)
    deep_input_emb = tf.keras.layers.Concatenate()([NoMask()(deep_input_emb)]+hist_ls)
    deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)
    # 链接dense、sparse部分
    dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)

    dnn_output = DNN(dnn_hidden_units, dnn_activation, 0, 0, False, seed=seed)(dnn_input)
    dnn_logit=tf.keras.layers.Dense(1, use_bias=False,
                                        kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_output)

    # FM和Linear部分
    linear_logit = get_linear_logit(process_features, linear_feature_columns, arithmetic_features, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)


    group_sparse_embedding_dict = embedding_lookup(embedding_dict, process_features, SparseFeatures)

    sequence_embed_dict = varlen_embedding_lookup(embedding_dict, process_features, VarLenSparseFeatures)
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                                 VarLenSparseFeatures)
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)

    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])


    if if_fm:
        final_logit = add_func([ fm_logit, dnn_logit,linear_logit])
    else:
        final_logit = add_func([dnn_logit, linear_logit])
    # final_logit=fm_logit
    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
