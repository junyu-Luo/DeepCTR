# -*- coding: utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com

"""

from collections import defaultdict
from itertools import chain

from tensorflow.python.keras.layers import Embedding, Lambda
from tensorflow.python.keras.regularizers import l2

from .layers.sequence import SequencePoolingLayer, WeightedSequenceLayer
from .layers.utils import Hash


# 过滤输入中的空值并返回列表形式的输入
def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))



# 为每个稀疏特征创建可训练的嵌入矩阵，使用字典存储所有特征列的嵌入矩阵，并返回该字典
def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True,l2_reg_embedding_uid=None):
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        if l2_reg_embedding_uid and feat.name=='user_id':
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(l2_reg_embedding_uid),
                            name=prefix + '_emb_' + feat.embedding_name)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb
        else:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(l2_reg),
                            name=prefix + '_emb_' + feat.embedding_name)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(
                                l2_reg),
                            name=prefix + '_seq_emb_' + feat.name,
                            mask_zero=seq_mask_zero)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding


# 从所有稀疏特征列中查询指定稀疏特征列(参数return_feat_list）的嵌入矩阵，以列表形式返回查询结果
def get_embedding_vec_list(embedding_dict, input_dict, sparse_feature_columns, return_feat_list=(), mask_feat_list=()):
    '''
    :param embedding_dict: type->dict；存储着所有特征列的嵌入矩阵的字典
    :param input_dict: type->dict；存储着特征列和对应的嵌入矩阵索引的字典，在没有使用hash查询时使用
    :param sparse_feature_columns: type->list；所有稀疏特征列
    :param return_feat_list: 需要查询的特征列，默认为空，为空则返回所有稀疏特征列的嵌入矩阵，不为空则仅返回该元组中的特征列的嵌入矩阵
    :param mask_feat_list: 用于哈希查询，默认为空
    :return:
    '''
    embedding_vec_list = []
    for fg in sparse_feature_columns:
        feat_name = fg.name
        if len(return_feat_list) == 0 or feat_name in return_feat_list:
            if fg.use_hash:
                lookup_idx = Hash(fg.vocabulary_size, mask_zero=(feat_name in mask_feat_list),
                                  vocabulary_path=fg.vocabulary_path)(input_dict[feat_name])
            else:
                lookup_idx = input_dict[feat_name]

            embedding_vec_list.append(embedding_dict[feat_name](lookup_idx))

    return embedding_vec_list



# 从所有特征列中筛选出SparseFeat和VarLenSparseFeat，然后调用函数create_embedding_dict为筛选的特征列创建嵌入矩阵
def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True,l2_reg_embedding_uid=None):
    from . import feature_column as fc_lib
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero,l2_reg_embedding_uid=l2_reg_embedding_uid)
    return sparse_emb_dict


# 从所有稀疏特征列中查询指定稀疏特征列(参数return_feat_list）的嵌入矩阵，以字典形式返回查询结果
def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False,gate_feat=None):
    '''
    :param sparse_embedding_dict: 存储稀疏特征列的嵌入矩阵的字典
    :param sparse_input_dict: 存储稀疏特征列的名字和索引的字典
    :param sparse_feature_columns: 稀疏特征列列表，元素为SparseFeat
    :param return_feat_list: 需要查询的稀疏特征列，如果元组为空，默认返回所有特征列的嵌入矩阵
    :param mask_feat_list: 用于哈希查询
    :param to_list: 是否以列表形式返回查询结果，默认是False
    :return:
    '''
    group_embedding_dict = defaultdict(list)
    gate_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            if fc.use_hash:
                lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list),
                                  vocabulary_path=fc.vocabulary_path)(
                    sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]
            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
            if gate_feat and embedding_name in gate_feat:
                gate_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    if not gate_feat:
        if to_list:
            return list(chain.from_iterable(group_embedding_dict.values()))
        return group_embedding_dict
    else:
        if to_list:
            return list(chain.from_iterable(group_embedding_dict.values())),list(chain.from_iterable(gate_embedding_dict.values()))
        return group_embedding_dict,gate_embedding_dict



# 获取varlen_sparse_feature_columns的嵌入矩阵
def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=True, vocabulary_path=fc.vocabulary_path)(
                sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict


# 获取varlen_sparse_feature_columns池化后的嵌入向量
def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, to_list=False):
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name
        if feature_length_name is not None:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm)(
                    [embedding_dict[feature_name], features[feature_length_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
                [seq_input, features[feature_length_name]])
        else:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm, supports_masking=True)(
                    [embedding_dict[feature_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
                seq_input)
        pooling_vec_list[fc.group_name].append(vec)
    if to_list:
        return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list



# 从所有特征列中选出DenseFeat，并以列表形式返回结果
def get_dense_input(features, feature_columns,gate_feat=None):
    from . import feature_column as fc_lib
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    gate_dense_input_list = []
    for fc in dense_feature_columns:
        if fc.transform_fn is None:
            dense_input_list.append(features[fc.name])
            if gate_feat and fc.name in gate_feat:
                gate_dense_input_list.append(features[fc.name])
        else:
            transform_result = Lambda(fc.transform_fn)(features[fc.name])
            dense_input_list.append(transform_result)
            if gate_feat and fc.name in gate_feat:
                transform_result_gate = Lambda(fc.transform_fn)(features[fc.name])
                gate_dense_input_list.append(transform_result_gate)
    if not gate_feat:
        return dense_input_list
    else:
        return dense_input_list,gate_dense_input_list


# 将a、b两个字典合并
def mergeDict(a, b):
    c = defaultdict(list)
    for k, v in a.items():
        c[k].extend(v)
    for k, v in b.items():
        c[k].extend(v)
    return c
