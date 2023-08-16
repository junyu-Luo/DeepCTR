# -*- coding: utf-8 -*-
'''

Author:
    Weichen Shen,weichenswc@163.com
modify:
    luojunyu
'''

import tensorflow as tf
from collections import namedtuple, OrderedDict
import copy
from itertools import chain
import math
from tensorflow.python.keras.initializers import RandomNormal, Zeros
from tensorflow.python.keras.layers import Input, Lambda

from .inputs import create_embedding_matrix, embedding_lookup, get_dense_input, varlen_embedding_lookup, \
    get_varlen_pooling_list, mergeDict
from .layers import Linear
from .layers.utils import concat_func

DEFAULT_GROUP_NAME = "default_group"


class SparseFeat:
    '''
    处理类别特征，将其转为固定维度的稠密特征
    name：生成的特征列的名字
    vocab：词表list或者tuple
    vocabulary_size：不同特征值的个数或当use_hash=True时的哈希空间
    embedding_dim：嵌入向量的维度
    use_hash：是否使用哈希编码，默认False
    dtype：默认int32
    embeddings_initializer：嵌入矩阵初始化方式，默认随机初始化
    embedding_name：默认None，其名字与name保持一致
    group_name：特征列所属的组
    traninable：嵌入矩阵是否可训练，默认True
    '''

    def __init__(self, name, vocab=None, vocabulary_size=None,
                 embedding_dim=4, use_hash=False,
                 vocabulary_path=None, dtype="int32",
                 embeddings_initializer=None,
                 embedding_name=None,
                 group_name="default_group", trainable=True):

        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2020)

        if embedding_name is None:
            embedding_name = name

        self.name = name
        self.vocab = vocab

        if vocabulary_size:
            self.vocabulary_size = vocabulary_size
        elif vocab:
            self.vocabulary_size = len(vocab) + 1
        else:
            assert vocab or vocabulary_size is not None

        self.embedding_dim = embedding_dim
        self.use_hash = use_hash
        self.vocabulary_path = vocabulary_path
        self.dtype = dtype
        self.embeddings_initializer = embeddings_initializer
        self.embedding_name = embedding_name
        self.group_name = group_name
        self.trainable = trainable

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat:
    '''
    将稠密特征转为向量的形式，并使用transform_fn 函数对其做归一化操作或者其它的线性或非线性变换
    name: 特征列名字
    dimension: 嵌入特征维度，默认是1
    dtype: 特征类型，default="float32"，
    operation：特征运算处理
    transform_fn: 转换函数，可以是归一化函数，也可以是其它的线性变换函数，
    以张量作为输入，经函数处理后，返回张量比如： lambda x: (x - 3.0) / 4.2)
    '''

    def __init__(self, name, dimension=1, dtype="float32",
                 transform_fn=None, operation=None, statistics=None):
        self.name = name
        self.dimension = dimension
        self.dtype = dtype
        # 复杂的尽量别使用transform_fn,Lambda层有(反)序列化限制且容易出错,建议继承然后编写一个子类层
        if transform_fn:
            print('复杂的尽量别使用transform_fn,Lambda层有(反)序列化限制且容易出错,建议继承然后编写一个子类层')
        self.transform_fn = transform_fn

        self.developed_op = ['normal', 'min_max', 'log', 'log_normal', 'donothing']
        if operation is None or operation == 'None':
            operation = 'donothing'
        assert operation in self.developed_op, f'{operation} not in {self.developed_op}'
        self.operation = operation
        self.statistics = statistics

    def __hash__(self):
        return self.name.__hash__()


class EmbFeat:
    '''
    默认emb都是1维的
    离线训练好的向量，输入给模型
    name: 特征列名字
    embed_size: embed_size特征维度 == Sparse 中 embedding_dim
    dtype: 特征类型，default="float32"，
    '''

    def __init__(self, name, vocab, vocabulary_size, weights=None,
                 trainable=False, embed_size=4, dtype='string',
                 embedding_name=None,
                 group_name="default_group"):
        if embedding_name is None:
            embedding_name = name
        self.name = name
        self.embed_size = int(embed_size)
        self.dtype = dtype
        self.embedding_name = embedding_name
        self.group_name = group_name
        self.vocab = vocab  # 使用tf.lookup.TextFileInitializer是文件名
        self.weights = weights
        self.vocabulary_size = vocabulary_size
        self.trainable = trainable

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat:
    def __init__(self, sparsefeat, maxlen,
                 combiner="mean", length_name=None,
                 weight_name=None, weight_norm=True):
        self.sparsefeat = sparsefeat
        self.maxlen = maxlen
        self.combiner = combiner
        self.length_name = length_name
        self.weight_name = weight_name
        self.weight_norm = weight_norm

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocab(self):
        return self.sparsefeat.vocab

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def vocabulary_path(self):
        return self.sparsefeat.vocabulary_path

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embeddings_initializer(self):
        return self.sparsefeat.embeddings_initializer

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    @property
    def trainable(self):
        return self.sparsefeat.trainable

    def __hash__(self):
        return self.name.__hash__()


def get_feature_names(feature_columns):
    '''
    获取所有特征列的名字，以列表形式返回
    '''
    features = build_input_features(feature_columns)
    return list(features.keys())


def build_input_features(feature_columns, prefix=''):
    from .layers.convlayer import VocabLayer, StrSeqPadLayer, IntSeqPadLayer, ArithmeticLayer, AutoDis
    """
    为所有的特征列构造keras tensor，生成输入的映射map，结果以OrderDict形式返回

    Parameters:
      feature_columns - 特征list，内含有[SparseFeat,DenseFeat,VarLen...]

    Returns:
        特征的有序字典 OrderedDict格式

    exsample:
         build_input_features([<DenseFeat object at 0x000002005C8F7E88>, <SparseFeat object at 0x0000020073AD3648>])
         return OrderedDict([('featureName', <KerasTensor: shape=(None, 1) dtype=int32 (created by layer 'featureName')>)])
    """
    input_features = OrderedDict()
    arithmetic_features = OrderedDict()
    process_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
            process_features[fc.name] = VocabLayer(fc.vocab, name=f'lookup_{fc.name}')(input_features[fc.name])
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
            assert fc.operation is not None
            # work
            # process_features[fc.name] = ArithmeticLayer(fc.operation, fc.statistics, name=f'{fc.operation}_{fc.name}')(
            #     input_features[fc.name])

            arithmetic_features[fc.name] = ArithmeticLayer(fc.operation, fc.statistics,
                                                           name=f'{fc.operation}_{fc.name}')(input_features[fc.name])

            process_features[fc.name] = AutoDis(num_buckets=8, emb_dim=4, keepdim=True)(arithmetic_features[fc.name])

        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
            process_features[fc.name] = StrSeqPadLayer(fc.vocab, fc.maxlen, name=f'lookup_{fc.name}')(
                input_features[fc.name])
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(1,), name=prefix + fc.weight_name, dtype=fc.dtype)
                # 处理权重列
                process_features[fc.weight_name] = IntSeqPadLayer(fc.maxlen, name=f'weight_{fc.weight_name}')(
                    input_features[fc.weight_name])
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        elif isinstance(fc, EmbFeat):
            from .layers.convlayer import Str2VecLayer
            input_features[fc.name] = Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
            process_features[fc.name] = Str2VecLayer(embedding_dim=fc.embed_size, name=f'Str2Vec_{fc.name}')(
                input_features[fc.name])

            # if fc.dtype == 'string':
            #     input_features[fc.name] = Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
            #     process_features[fc.name] = Str2VecLayer(name=f'Str2Vec_{fc.name}')(input_features[fc.name])
            #
            # else:
            #     input_features[fc.name] = Input(shape=(fc.embed_size,), name=prefix + fc.name, dtype=fc.dtype)
            #     process_features[fc.name] = Input(shape=(fc.embed_size,), name=prefix + fc.name, dtype=fc.dtype)

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features, process_features, arithmetic_features


def get_linear_logit(features, feature_columns, arithmetic_features, units=1, use_bias=False, seed=1024,
                     prefix='linear', l2_reg=0, sparse_feat_refine_weight=None):
    """
    获取linear_logit（线性变换）的结果

    Parameters:
      features - 特征的有序字典 OrderedDict格式 [(key1,val1),(key2,val2)...]
      feature_columns - 特征list，内含有3种函数
        例：[SparseFeat,DenseFeat,VocabLayer]

    Returns:
        logit list

    """
    linear_feature_columns = copy.deepcopy(feature_columns)
    for i in range(len(linear_feature_columns)):
        if isinstance(linear_feature_columns[i], SparseFeat):
            linear_feature_columns[i].embedding_dim = 1
            linear_feature_columns[i].embeddings_initializer = Zeros()
        if isinstance(linear_feature_columns[i], VarLenSparseFeat):
            linear_feature_columns[i].sparsefeat.embedding_dim = 1
            linear_feature_columns[i].sparsefeat.embeddings_initializer = Zeros()

    linear_emb_list = [
        input_from_feature_columns(features, linear_feature_columns, l2_reg, seed, prefix=prefix + str(i))
        [0] for i in range(units)
    ]

    dictMerge = dict(list(features.items()) + list(arithmetic_features.items()))
    _, dense_input_list = input_from_feature_columns(
        dictMerge, linear_feature_columns, l2_reg, seed,
        prefix=prefix)


    linear_logit_list = []
    for i in range(units):
        if len(linear_emb_list[i]) > 0 and len(dense_input_list) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            dense_input = concat_func(dense_input_list)
            if sparse_feat_refine_weight is not None:
                sparse_input = Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=1))(
                    [sparse_input, sparse_feat_refine_weight])
            linear_logit = Linear(l2_reg, mode=2, use_bias=use_bias, seed=seed)([sparse_input, dense_input])
        elif len(linear_emb_list[i]) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            if sparse_feat_refine_weight is not None:
                sparse_input = Lambda(lambda x: x[0] * tf.expand_dims(x[1], axis=1))(
                    [sparse_input, sparse_feat_refine_weight])
            linear_logit = Linear(l2_reg, mode=0, use_bias=use_bias, seed=seed)(sparse_input)
        elif len(dense_input_list) > 0:
            dense_input = concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=1, use_bias=use_bias, seed=seed)(dense_input)
        else:  # empty feature_columns
            return Lambda(lambda x: tf.constant([[0.0]]))(list(features.values())[0])
        linear_logit_list.append(linear_logit)

    return concat_func(linear_logit_list)


# def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
#                                support_dense=True, support_group=False):
#     sparse_feature_columns = list(
#         filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
#     varlen_sparse_feature_columns = list(
#         filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
#
#     embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
#                                                     seq_mask_zero=seq_mask_zero)
#     group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
#     dense_value_list = get_dense_input(features, feature_columns)
#     if not support_dense and len(dense_value_list) > 0:
#         raise ValueError("DenseFeat is not supported in dnn_feature_columns")
#
#     sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
#     group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
#                                                                  varlen_sparse_feature_columns)
#     group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
#     if not support_group:
#         group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
#     return group_embedding_dict, dense_value_list

def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False, gate_feat=None, l2_reg_embedding_uid=None):
    """
    为所有特征列创建嵌入矩阵，并分别返回包含SparseFeat和VarLenSparseFeat的嵌入矩阵的字典，
    以及包含DenseFeat的数值特征的字典具体实现是通过调用inputs中的create_embedding_matrix、
    embedding_lookup、varlen_embedding_lookup等函数完成

    Parameters:
      features - 特征的有序字典 OrderedDict格式 [(key1,val1),(key2,val2)...]
      feature_columns - 特征list，内含有3种函数
        例：[SparseFeat,DenseFeat,VocabLayer]
      其他可设置默认

    Returns:
    embedding_dict 或者 dense_list（support_group参数决定）
    group_embedding_list/dict  -  embedding 后得到的 list
        [<KerasTensor: shape=(None, 1, 1) dtype=float32 (created by layer 'xxxx')>]

    dense_value_list  -  dense_list

    exception: support_group 默认是 false，返回list。 如果设置 support_group=true，则返回 dict
    """
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                    seq_mask_zero=seq_mask_zero,
                                                    l2_reg_embedding_uid=l2_reg_embedding_uid)
    gate_sparse_embedding_dict = None
    gate_dense_value_list = None
    if not gate_feat:
        group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
        dense_value_list = get_dense_input(features, feature_columns)
    else:
        group_sparse_embedding_dict, gate_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features,
                                                                                   sparse_feature_columns,
                                                                                   gate_feat=gate_feat)
        dense_value_list, gate_dense_value_list = get_dense_input(features, feature_columns, gate_feat=gate_feat)

    if not support_dense and len(dense_value_list) > 0:
        # only CCPM,AFM
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")
    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                                 varlen_sparse_feature_columns)
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
    if not support_group:
        group_embedding_list = list(chain.from_iterable(group_embedding_dict.values()))
        if not gate_feat:
            return group_embedding_list, dense_value_list
        else:
            return group_embedding_list, dense_value_list, list(
                chain.from_iterable(gate_sparse_embedding_dict.values())), gate_dense_value_list
    else:
        if not gate_feat:
            return group_embedding_dict, dense_value_list
        else:
            return group_embedding_dict, dense_value_list, gate_sparse_embedding_dict, gate_dense_value_list
