import sys, os, time, json

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Zeros, glorot_uniform
from tensorflow.keras import layers, activations
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import backend as K

from .feature_difinition import DenseFeature, SparseFeature, VarLenFeature, PersonaFeature

# # Model
# ## Layer
class VocabLayer(tf.keras.layers.Layer):
    ''' keys --> [1, len(keys)]， 缺失值/OOV --> 0
    '''
    def __init__(self, keys, **kwargs):
        super(VocabLayer, self).__init__(**kwargs)
        self.keys = keys
        vals = list(range(1, len(keys) + 1))
        keys = tf.constant(keys)
        vals = tf.constant(vals, dtype='int32')
        self.table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, vals), 0)

    def build(self, input_shape):
        self.built = True

    def get_config(self):
        base_config = super(VocabLayer, self).get_config()
        config = {'keys': self.keys}
        config.update(base_config)
        return config

    def call(self, inputs):
        return self.table.lookup(inputs)


class VocabLayerV2(tf.keras.layers.Layer):
    ''' keys --> [1, len(keys)]， 缺失值/OOV --> 0
    '''
    def __init__(self, keys, **kwargs):
        assert all(isinstance(x, str) for x in keys) or all(isinstance(x, int) for x in keys), 'vocab must be all string or all int'

        super(VocabLayerV2, self).__init__(**kwargs)
        self.keys = keys

        if all(isinstance(x, str) for x in keys):
            self.table = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=keys, mask_token=None)
        else:
            self.table = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=keys, mask_token=None)

    def build(self, input_shape):
        self.built = True

    def get_config(self):
        base_config = super(VocabLayerV2, self).get_config()
        config = {'keys': self.keys}
        config.update(base_config)
        return config

    def call(self, inputs):
        return tf.cast(self.table(inputs), tf.int32)


class ArithmeticLayer(tf.keras.layers.Layer):
    def __init__(self, op_type, num_x=None, num_y=None, **kwargs):
        super(ArithmeticLayer, self).__init__(**kwargs)
        self.op_type = op_type
        self.num_x = num_x
        self.num_y = num_y
        
            
    def build(self, input_shape):
        self.built = True
        
    def get_config(self):
        base_config = super(ArithmeticLayer, self).get_config()
        config = {
            'op_type': self.op_type, 
            'num_x': self.num_x, 
            'num_y': self.num_y,
           }
        config.update(base_config)
        return config
    
    def call(self, inputs):
        if self.op_type == 'add' and self.num_x is not None:
            return tf.math.add(inputs, self.num_x)
        elif self.op_type == 'divide' and self.num_x is not None and self.num_x!=0:
            return tf.math.divide(inputs, self.num_x)
        elif self.op_type == 'sqrt':
            return tf.math.sqrt(inputs)
        elif self.op_type == 'log':  # add_one_log
            if self.num_x and self.num_x > 0:
                return tf.math.divide(tf.math.log(tf.math.add(inputs, 1)), tf.math.log(self.num_x))
            else:
                return tf.math.log(tf.math.add(inputs, 1))
        elif self.op_type == 'clip_by_value' and self.num_x is not None and self.num_y is not None:
            return tf.clip_by_value(inputs, self.num_x, self.num_y)
        elif self.op_type == 'normal' and self.num_x is not None and self.num_y is not None and self.num_y!=0:
            return tf.math.divide(tf.math.subtract(inputs, tf.constant(self.num_x)), tf.constant(self.num_y))
        elif self.op_type == 'log_normal' and self.num_x is not None and self.num_y is not None and self.num_y!=0:
            return tf.math.divide(tf.math.subtract(tf.math.log(inputs+1), self.num_x), self.num_y)
        elif self.op_type == 'min_max' and self.num_x is not None and self.num_y is not None and self.num_y > self.num_x:
            return tf.math.divide(tf.math.subtract(inputs, self.num_x), self.num_y-self.num_x)
        elif self.op_type == 'ratio':
            # 平滑行归一化
            x = inputs + float(self.num_x)
            return x / tf.reduce_sum(x, axis=-1, keepdims=True)
        else:
            raise Exception("unknown op_type:%s or invalid num_x:%s or invalid num_y:%s"%(self.op_type, self.op_num_x, self.op_num_y))

class DNN(layers.Layer):
    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None, seed=666, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed
        super(DNN, self).__init__(**kwargs)


    def build(self, input_shape):
        if len(self.hidden_units) == 0:
            raise ValueError("hidden_units is empty")

        self.dense_layers = [layers.Dense(units, kernel_initializer=glorot_uniform(seed=self.seed+i), bias_initializer=Zeros(),
                             kernel_regularizer=l2(self.l2_reg)) for i, units in enumerate(self.hidden_units)]

        if self.use_bn:
            self.bn_layers = [layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.activation_layers = {str(i):layers.Activation(self.activation) for i in range(len(self.hidden_units))}  # 用list会出现Unable to save the object ListWrapper报错(MMoE中，但WideDeep不会)

        if self.dropout_rate > 0:
            self.dropout_layers = [layers.Dropout(self.dropout_rate, seed=self.seed+i) for i in range(len(self.hidden_units))]

        if self.output_activation:
            if self.output_activation == 'no':
                self.activation_layers[str(len(self.activation_layers)-1)] = None
            else:
                self.activation_layers[str(len(self.activation_layers)-1)] = layers.Activation(self.output_activation)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!


    def call(self, inputs, training=None, **kwargs):
        hidden = inputs
        for i in range(len(self.hidden_units)):
            hidden = self.dense_layers[i](hidden)
            if self.use_bn:  # TODO: after activation
                hidden = self.bn_layers[i](hidden, training=training)

            if self.activation_layers[str(i)] is not None:
                try:
                    hidden = self.activation_layers[str(i)](hidden, training=training)
                except TypeError as e: # TypeError: call() got an unexpected keyword argument 'training'
                    print("make sure the activation function use training flag properly", e)
                    hidden = self.activation_layers[str(i)](hidden)

            if self.dropout_rate > 0:
                hidden = self.dropout_layers[i](hidden, training=training)
        return hidden

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape
        return tuple(shape)

    def get_config(self):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class FM(layers.Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))
        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)

class AutoDis(layers.Layer):
    def __init__(self, num_buckets=4, emb_dim=16, keepdim=False, initializer='glorot_uniform', regularizer=None, **kwargs):
        super(AutoDis, self).__init__(**kwargs)
        self.num_buckets = num_buckets
        self.emb_dim = emb_dim
        self.keepdim = keepdim
        self.initializer = initializer  # random_normal, truncated_normal, glorot_normal, glorot_uniform, he_normal, he_uniform
        self.regularizer = regularizer
        super(AutoDis, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("Unexpected inputs dimensions % d, expect to be 2 dimensions" % (len(input_shape)))
        self.meta_embeddings = self.add_weight(shape=(self.num_buckets, self.emb_dim), initializer=self.initializer, regularizer=self.regularizer, trainable=True, name='autodis_meta_embeds')
        self.weight_hidden_1 = layers.Dense(max(self.emb_dim*4, 64), activation='relu')
        self.weight_hidden_2 = tf.keras.layers.Dense(self.num_buckets)
        self.weight_softmax = layers.Softmax()
        super(AutoDis, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        buckets_weight = self.weight_softmax(self.weight_hidden_2(self.weight_hidden_1(inputs)))  # (None, num_buckets)
        embedding = tf.matmul(tf.expand_dims(buckets_weight, axis=1), self.meta_embeddings)
        if self.keepdim:
            embedding = tf.squeeze(embedding, axis=1)
        return embedding

    def compute_output_shape(self, input_shape):
        return (None, 1, self.emb_dim)

    def get_config(self):
        config = {'num_buckets': self.num_buckets, 'emb_dim': self.emb_dim,
                  'keepdim': self.keepdim, 'initializer': self.initializer}
        base_config = super(AutoDis, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DotAttention(tf.keras.layers.Layer):
    def __init__(self, use_scale=False, support_mask=True, **kwargs):
        self.use_scale = use_scale
        self.support_mask = support_mask
        super(DotAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.WQ = self.add_weight(name='ATT_Q',
            shape=(input_shape[0][-1], input_shape[0][-1]),
            initializer="random_normal", trainable=True
        )
        self.WK = self.add_weight(name='ATT_K',
            shape=(input_shape[1][-1], input_shape[1][-1]),
            initializer="random_normal", trainable=True
        )
        self.WV = self.add_weight(name='ATT_V',
            shape=(input_shape[2][-1], input_shape[2][-1]),
            initializer="random_normal", trainable=True
        )
        self.d_k = tf.constant(input_shape[1][-1], dtype=tf.float32)
        super(DotAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.support_mask:
            raw_query, raw_key, raw_value, value_mask = inputs  # [None, seq_len_{q,k,v}, emb_dim]
        else:
            raw_query, raw_key, raw_value = inputs  # [None, seq_len_{q,k,v}, emb_dim]
        Q = tf.matmul(raw_query, self.WQ)  # [None, seq_len_q, emb_dim]
        K = tf.matmul(raw_key, self.WK)
        V = tf.matmul(raw_value, self.WV)

        attention = tf.matmul(Q, K, transpose_b=True)  # [None, seq_len_q, seq_len_k]
        if self.use_scale:
            attention = tf.divide(attention, tf.sqrt(self.d_k))
        if self.support_mask:
            padding = tf.zeros_like(attention)
            attention = tf.where(tf.expand_dims(value_mask, axis=2)>0, attention, padding)
        attention = tf.nn.softmax(attention, axis=-1)
        output = tf.matmul(attention, V)  # [None, seq_len_q, emb_dim]
        return output
    
    def compute_output_shape(self, input_shape):
        return [None, input_shape[0][1], input_shape[2][-1]]

    def get_config(self):
        config = {'use_scale': self.use_scale, 'support_mask': self.support_mask}
        base_config = super(DotAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def sequence_embs_pooling(multi_embs, mask=None, method='sum', emb_dict=None):
    ''' multi_embs: [None, max_len, emb_dim]
    idxes/mask: [None, max_len]
    '''
    if method in ('avg', 'mean', 'sum'):
        mask = tf.expand_dims(tf.cast(mask, 'float32'), axis=1)
        sumed = tf.matmul(mask, multi_embs)
        if method == 'sum':
            merged_emb = sumed
        elif method in ('avg', 'mean'):
            avged = sumed / (tf.reduce_sum(mask, axis=-1, keepdims=True)  + 1e-6)
            merged_emb = avged
        else:
            raise ValueError('method error')
    elif method.startswith('cross-attention:'):
        assert isinstance(emb_dict, dict), 'cross-attention must provide emb_dict'
        query_name = method.split(':', 1)[1]
        query_emb = emb_dict[query_name]
        if mask is not None:
            merged_emb = DotAttention(use_scale=0, support_mask=1)([query_emb, multi_embs, multi_embs, mask])
        else:
            merged_emb = DotAttention(use_scale=0, support_mask=0)([query_emb, multi_embs, multi_embs])
    elif method == 'attention':
        Q = layers.Dense(multi_embs.shape[-1])(multi_embs)
        K = layers.Dense(multi_embs.shape[-1])(multi_embs)
        V = layers.Dense(multi_embs.shape[-1])(multi_embs)
        multi_embs = tf.keras.layers.Attention(use_scale=False)([Q, V, K], mask=[None, mask])
        _merged_emb = multi_embs[:, :1, :]
        merged_emb = tf.keras.layers.Dense(4*_merged_emb.shape[-1], activation='relu')(_merged_emb)
        merged_emb = tf.keras.layers.Dense(_merged_emb.shape[-1])(merged_emb)
    elif method == 'attention_no_mask':
        Q = layers.Dense(multi_embs.shape[-1])(multi_embs)
        K = layers.Dense(multi_embs.shape[-1])(multi_embs)
        V = layers.Dense(multi_embs.shape[-1])(multi_embs)
        multi_embs = tf.keras.layers.Attention(use_scale=False)([Q, V, K])
        _merged_emb = multi_embs[:, :1, :]
        merged_emb = tf.keras.layers.Dense(4*_merged_emb.shape[-1], activation='relu')(_merged_emb)
        merged_emb = tf.keras.layers.Dense(_merged_emb.shape[-1])(merged_emb)
    else:
        raise ValueError('method must be in ()')
    return merged_emb


# def DNN(X, hidden_units, reg, activation='relu', initializer='glorot_uniform', name_prefix=None):
#     _hidden = X
#     for i, units in enumerate(hidden_units):
#         _name = None if name_prefix is None else f'{name_prefix}_{i+1}'
#         _hidden = layers.Dense(units, kernel_regularizer=l2(reg), activation=activation, kernel_initializer=initializer, name=_name)(_hidden)
#     return _hidden

# # Util function
def weighted_sum(embs, weights, squeeze=False):
    # embs: (None, seq_len, emb_dim)
    # weights: (None, seq_len)
    w_sum = tf.matmul(tf.expand_dims(weights, axis=1), embs)
    return w_sum if not squeeze else tf.squeeze(w_sum, axis=1)


def get_inputs_dict(feature_columns):
    inputs_dict = dict()
    for fc in feature_columns:
        inputs_dict[fc.name] = layers.Input(name=fc.name, shape=fc.shape, dtype=fc.dtype)
    return inputs_dict


def get_embeddings_dict(inputs_dict, feature_columns, middle_outputs=None, reg=0, **kwargs):
    return_middle = True if middle_outputs is not None and isinstance(middle_outputs, dict) else False
    share_layers = {'lookup': {}, 'embed': {}}
    embeddings_dict = dict()
    for fc in feature_columns:
        x = inputs_dict[fc.name]
        if isinstance(fc, DenseFeature):
            normal_dense = ArithmeticLayer(fc.op, fc.op_num_x, fc.op_num_y, name=f'{fc.op}_{fc.name}')(x) if fc.op is not None else x
            emb = AutoDis(num_buckets=8, emb_dim=fc.emb_dim, keepdim=False, regularizer=l2(reg))(normal_dense) if fc.name != 'i_text_emb' else \
                AutoDis(num_buckets=16, emb_dim=16, keepdim=True, regularizer=l2(reg))(normal_dense)  # TODO: num_buckets
            if return_middle: middle_outputs[fc.name] = normal_dense
        elif isinstance(fc, SparseFeature) or isinstance(fc, VarLenFeature):
            if kwargs.get('vocab_v2'):
                share_layers['lookup'][fc.name] = VocabLayerV2(fc.vocab, name=f'lookup_{fc.name}')
            else:
                share_layers['lookup'][fc.name] = VocabLayer(fc.vocab, name=f'lookup_{fc.name}')
            if 'pretrained_path' in fc.__dict__:
                pretrained = get_pretrained_embs(fc.pretrained_path, fc.vocab)
                share_layers['embed'][fc.name] = layers.Embedding(1+fc.vocab_size, fc.emb_dim, weights=[pretrained], trainable=True,
                                                                embeddings_regularizer=l2(reg), name=f'emb_pretrained_{fc.name}')
            else:
                share_layers['embed'][fc.name] = layers.Embedding(1+fc.vocab_size, fc.emb_dim, embeddings_regularizer=l2(reg), name=f'emb_{fc.name}')
            
            if isinstance(fc, SparseFeature):
                lookup_idx = share_layers['lookup'][fc.share_emb or fc.name](x)
                emb = share_layers['embed'][fc.share_emb or fc.name](lookup_idx)
                if return_middle: middle_outputs[fc.name] = lookup_idx
            else:
                if fc.dtype == 'string':
                    # TODO: to be a layer
                    x = tf.strings.split(x, sep=fc.sep, maxsplit=fc.max_len, name=f'{fc.name}_split')
                    x = tf.squeeze(x.to_tensor('',shape=[None, 1, fc.max_len]), axis=1)
                    lookup_idxs = share_layers['lookup'][fc.share_emb or fc.name](x)
                elif fc.dtype == 'int32':
                    lookup_idxs = x
                else:
                    raise ValueError(f'VarLenFeature({fc.name}): dtype must be string(shape=1) or int32(shape=max_len)')

                multi_embs = share_layers['embed'][fc.share_emb or fc.name](lookup_idxs)
                mask = tf.not_equal(lookup_idxs, tf.constant(0, dtype='int32'))
                emb = sequence_embs_pooling(multi_embs, mask=mask if not fc.pooling.startswith('cross') else None, method=fc.pooling, emb_dict=embeddings_dict)
                if return_middle: middle_outputs[fc.name] = lookup_idxs
        else:
            raise ValueError('Unknown: feature column must be DenseFeature, SparseFeature or VarLenFeature')
        embeddings_dict[fc.name] = emb
    return embeddings_dict

def get_pretrained_embs(emb_path, target_vocab, get_default=True):
    if emb_path.startswith('hdfs'):
        if os.system('rm -f .tmp_pretrained_embs.keyedvector') != 0:
            raise Exception('can not remove temp file: .tmp_pretrained_embs.keyedvector')
        if os.system(f'hdfs dfs -get {emb_path} .tmp_pretrained_embs.keyedvector') != 0:
            raise Exception(f'hdfs get failed: {emb_path}')
        emb_path = '.tmp_pretrained_embs.keyedvector'

    from gensim.models import KeyedVectors
    kv = KeyedVectors.load(emb_path)
    if emb_path == '.tmp_pretrained_embs.keyedvector':
        os.system('rm -f .tmp_pretrained_embs.keyedvector')

    idxes = [] if not get_default else [0]
    for w in target_vocab:
        if w not in kv.key_to_index:
            idxes.append(0); print('WARN:', w, 'not in pretrained-embs keys, will use default vector')
        else:
            idxes.append(kv.key_to_index[w])
    
    mean_emb = kv.vectors.mean(axis=0)  # 用平均向量替换第一个向量
    kv.vectors[0, :] = mean_emb
    
    embs = kv.vectors[idxes, :]
    return embs


def get_linear_logit(normal_inputs_dict, feature_columns, linear_reg):
    dense_list, sparse_list, varlen_list = list(), list(), list()
    for fc in feature_columns:
        if isinstance(fc, DenseFeature):
            dense_list.append(normal_inputs_dict[fc.name])  # (None, 1|n)
        elif isinstance(fc, SparseFeature):
            sparse_weight_layer = layers.Embedding(1+len(fc.vocab), 1, name=f'linear_emb_{fc.name}', embeddings_regularizer=l2(linear_reg))
            sparse_list.append(sparse_weight_layer(normal_inputs_dict[fc.name]))  # (None, 1, 1)
        elif isinstance(fc, VarLenFeature):
            varlen_weight_layer = layers.Embedding(1+len(fc.vocab), 1, name=f'linear_emb_{fc.name}', embeddings_regularizer=l2(linear_reg))
            varlen_list.append(varlen_weight_layer(normal_inputs_dict[fc.name]))  # (None, max_len, 1)
        else:
            raise ValueError('Unknown: feature column must be DenseFeature, SparseFeature or VarLenFeature')
    
    linear_logits = list()  # [(None, 1), ...]
    if len(dense_list) > 0:
        dense_concat = layers.Concatenate(axis=-1, name='linear_dense_concat')(dense_list) if len(dense_list)>1 else dense_list[0]
        dense_logit = layers.Dense(1, kernel_regularizer=l2(linear_reg), bias_regularizer=l2(linear_reg))(dense_concat)
        linear_logits.append(dense_logit)
    if len(sparse_list) > 0:
        sparse_concat = layers.Add(name='linear_sparse_add')(sparse_list) if len(sparse_list)>1 else sparse_list[0]
        sparse_logit = layers.Flatten(name='linear_sparse_logit')(sparse_concat)
        linear_logits.append(sparse_logit)
    if len(varlen_list) > 0:
        varlen_concat = layers.Concatenate(axis=1, name='linear_varlen_add')(varlen_list) if len(varlen_list)>1 else varlen_list[0]
        varlen_logit = layers.Lambda(lambda x: K.sum(x, axis=[1,2]), name='linear_varlen_logit')(varlen_concat)  # TODO: mask
        linear_logits.append(varlen_logit)

    if len(linear_logits) == 1:
        return linear_logits[0]
    else:
        return layers.Add(name='linear_logit')(linear_logits)

def build_WideDeep_model(feature_columns, dnn_hidden_units=(128, 128), dnn_activation='relu', #emb_dim=None,
                use_linear=True, use_fm=False, use_dnn=True,
                linear_reg=1e-5, emb_reg=1e-5, dnn_reg=1e-4, run_eagerly=None,
                optimizer='adam', loss='binary_crossentropy', metrics=['AUC', 'binary_crossentropy'], model_name='model'):
    # # checks
    assert use_linear or use_fm or use_dnn, 'must use_linear or use_fm or use_dnn'
    assert all(True if isinstance(fc, DenseFeature) or isinstance(fc, SparseFeature) or isinstance(fc, VarLenFeature) else False for fc in feature_columns)

    # # inputs and embeddings
    inputs_dict = get_inputs_dict(feature_columns)
    normal_inputs_dict = dict()  # 用于存储归一化后的DenseFeature和lookup后的SparseFeature和VarLenFeature
    embeddings_dict = get_embeddings_dict(inputs_dict, feature_columns, middle_outputs=normal_inputs_dict)

    logits_list = list()
    # # linear
    if use_linear:
        linear_feature_columns = []
        for fc in feature_columns:
            if isinstance(fc, VarLenFeature) and len(fc.vocab)>100:
                print('linear skip:', fc.name)
            else:
                linear_feature_columns.append(fc)
        if len(linear_feature_columns) >= 1:
            linear_logit = get_linear_logit(normal_inputs_dict, linear_feature_columns, linear_reg=linear_reg)
            logits_list.append(linear_logit)
    # # fm
    if use_fm:
        for x in ('gender', 'city', 'province', 'experience_level', 'glamour_level', 'wealth_level', 'nobility_level'):
            fm_feature_columns = [fc for fc in feature_columns if fc.name in (f'u_{x}', f'i_p_{x}')]
            if len(fm_feature_columns) >= 2:
                emb_list = [embeddings_dict[fc.name] for fc in fm_feature_columns]
                fm_logit = FM(name=f'fm_{x}')(layers.Concatenate(axis=1, name=f'{x}_concat')(emb_list))
                logits_list.append(fm_logit)
        fm_feature_columns = [fc for fc in feature_columns if fc.name in ('expo_tag_id', 'ui_room_music_tab_enter_cnt_1w')]
        if len(fm_feature_columns) >= 2:
            emb_list = [embeddings_dict[fc.name] for fc in fm_feature_columns]
            fm_logit = FM(name='fm_ui_room_music_tab_enter_cnt_1w')(layers.Concatenate(axis=1, name='ui_room_music_tab_enter_cnt_1w_concat')(emb_list))
            logits_list.append(fm_logit)
        fm_feature_columns = [fc for fc in feature_columns if fc.name in ('expo_tag_id', 'ui_room_music_tab_enter_cnt_3w')]
        if len(fm_feature_columns) >= 2:
            emb_list = [embeddings_dict[fc.name] for fc in fm_feature_columns]
            fm_logit = FM(name='fm_ui_room_music_tab_enter_cnt_3w')(layers.Concatenate(axis=1, name='ui_room_music_tab_enter_cnt_3w_concat')(emb_list))
            logits_list.append(fm_logit)


    # # dnn
    if use_dnn:
        dnn_feature_columns = feature_columns
        if len(dnn_feature_columns) >= 1:
            # TODO: flatten_concat
            dnn_input = layers.Concatenate(axis=-1, name='dnn_concat_embs')([layers.Flatten(name=f'flatten_{fc.name}')(embeddings_dict[fc.name]) for fc in feature_columns])
            # _hidden = DNN(dnn_input, hidden_units=dnn_hidden_units, activation=dnn_activation, reg=dnn_reg, name_prefix='dnn_hidden')
            _hidden = DNN(dnn_hidden_units, activation=dnn_activation, l2_reg=dnn_reg, name='dnn_hidden')(dnn_input)
            dnn_logits = layers.Dense(1, name='dnn_logit')(_hidden)
            logits_list.append(dnn_logits)

    # # combine linear, fm, dnn to calculate final sigmoid output
    final_logit = layers.Add(name='final_logit')(logits_list) if len(logits_list)>1 else logits_list[0]
    sigmoid_output = layers.Activation('sigmoid', name='lr_output')(final_logit)

    # build and compile model
    model = models.Model(inputs=inputs_dict, outputs=[sigmoid_output], name=model_name)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=run_eagerly)
    return model


def get_model_basic_infos(model, history=None):
    num_params = model.count_params()
    num_epochs = len(history.history['loss']) if history else None
    num_inputs = len(model.inputs)
    num_outputs = len(model.outputs)
    return {
        'num_params': num_params,
        'num_epochs': num_epochs,
        'num_inputs': num_inputs,
        'num_outputs': num_outputs,
    }

# 保存模型
def save_model(model, base_path, tfx_warmup=True, X_4_warmup=None, num_warmup=100, test_savedmodel=True):
    def test_load_saved_model(saved_path, test_data):
        """ 测试保存的模型是否正常 """
        from tensorflow.keras.models import load_model
        lmodel = load_model(saved_path)
        ly = lmodel.predict({k:v[:1000] for k, v in test_data.items()})
        y = model.predict({k:v[:1000] for k, v in test_data.items()})
        np.testing.assert_allclose(y, ly, rtol=1e-5)

    def generate_tfserving_warmup(X: dict, savedmodel_path, n=100):
        import os
        import tensorflow as tf
        try:
            from tensorflow_serving.apis import model_pb2
            from tensorflow_serving.apis import predict_pb2
            from tensorflow_serving.apis import prediction_log_pb2
        except Exception as e:
            print('WARN: save tfx_warmup failed, please install tensorflow_serving')
            return None

        extra_path = os.path.join(savedmodel_path, 'assets.extra')
        warmup_file = os.path.join(extra_path, 'tf_serving_warmup_requests')
        if not os.path.exists(extra_path):
            os.makedirs(extra_path)

        with tf.io.TFRecordWriter(warmup_file) as writer:
            request = predict_pb2.PredictRequest(
        #             model_spec=model_pb2.ModelSpec(name="ntu_50419_v1", signature_name="serving_default"),
                inputs={k: tf.make_tensor_proto(v[:n].reshape([-1, 1]) if len(v.shape)==1 else v[:n]) for k, v in X.items()}
            )
            log = prediction_log_pb2.PredictionLog(predict_log=prediction_log_pb2.PredictLog(request=request))
            writer.write(log.SerializeToString())
        return extra_path

    print('start saving model')
    import datetime, pytz
    now_str = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d%H%M")
    savedmodel_path = os.path.join(base_path, now_str)
    model.save(savedmodel_path)
    if tfx_warmup and X_4_warmup is not None:
        _extra_path = generate_tfserving_warmup(X_4_warmup, savedmodel_path, n=num_warmup)
    print('finish saving model')
    if test_savedmodel:
        test_load_saved_model(savedmodel_path, X_4_warmup)
    return savedmodel_path

