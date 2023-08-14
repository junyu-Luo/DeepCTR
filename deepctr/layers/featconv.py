import tensorflow as tf
import math
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import embedding_ops


class VocabLayer(tf.keras.layers.Layer):
    ''' keys --> [1, len(keys)]， 缺失值/OOV --> 0
    '''

    def __init__(self, keys, **kwargs):
        super(VocabLayer, self).__init__(**kwargs)
        self.keys = keys
        vals = list(range(1, len(keys) + 1))
        keys = tf.constant(keys)
        vals = tf.constant(vals, dtype=tf.int32)
        self.table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, vals), 0)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        return self.table.lookup(inputs)

    def get_config(self):
        base_config = super(VocabLayer, self).get_config()
        config = {'keys': self.keys}
        config.update(base_config)
        return config


class ChangeType(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ChangeType, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        return tf.cast(inputs, tf.float32)

    def get_config(self):
        base_config = super(ChangeType, self).get_config()
        config = {}
        config.update(base_config)
        return config


class EmbVocabLayer(tf.keras.layers.Layer):
    ''' keys --> [0, len(keys)]， 缺失值/OOV --> keys.index('-1')
    '''

    def __init__(self, file_name, default_numb, **kwargs):
        super(EmbVocabLayer, self).__init__(**kwargs)
        self.file_name = file_name

        init = tf.lookup.TextFileInitializer(
            filename=file_name,
            key_dtype=tf.string, key_index=0,
            value_dtype=tf.int64, value_index=1,
            delimiter=' ')

        self.table = tf.lookup.StaticHashTable(init, default_numb - 1)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        return self.table.lookup(inputs)

    def get_config(self):
        base_config = super(EmbVocabLayer, self).get_config()
        config = {'file_name': self.file_name}
        config.update(base_config)
        return config


class ArithmeticLayer(tf.keras.layers.Layer):
    def __init__(self, op_type, statistics=None, **kwargs):
        super(ArithmeticLayer, self).__init__(**kwargs)
        self.op_type = op_type
        self.statistics = statistics
        self.min = statistics.get('min')
        self.mean = statistics.get('mean')
        self.quantile95 = statistics.get('quantile95')
        self.max = statistics.get('max')
        self.std = statistics.get('std')

        # 可放入 statistics 中，但感觉没必要
        self.add_numb = 0
        self.divide_numb = 1
        self.log_numb = -1

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        if inputs.dtype == tf.int32 or inputs.dtype == tf.int64:
            inputs = tf.cast(inputs, tf.float32)

        if self.op_type == 'sqrt':
            return tf.math.sqrt(inputs)
        elif self.op_type == 'log':  # add_one_log
            if self.log_numb and self.log_numb > 0:
                return tf.math.divide(tf.math.log(tf.math.add(tf.abs(inputs), 1)), math.log(self.log_numb))
            else:
                return tf.math.log(tf.math.add(tf.abs(inputs), 1))
        elif self.op_type == 'clip_by_value':
            return tf.clip_by_value(inputs, self.min, self.max)
        elif self.op_type == 'normal':
            if self.std == 0:
                raise Exception('{} x / 0'.format(self.name))
            return tf.math.divide(tf.math.subtract(inputs, self.mean), self.std)
        elif self.op_type == 'log_normal':
            if self.std == 1:
                raise Exception('{} x / 0'.format(self.name))
            if self.std < 0 or self.mean < 0:
                raise Exception('{} negative number with log'.format(self.name))
            return tf.math.divide(tf.math.subtract(tf.math.log(tf.abs(inputs) + 1),
                                                   tf.math.log(self.mean)),
                                  tf.math.log(self.std))
        elif self.op_type == 'min_max':
            if self.min == 0 and self.max == 0:
                raise Exception('{} x / 0'.format(self.name))
            return tf.math.divide(tf.math.subtract(inputs, self.min), self.max - self.min)
        elif self.op_type == 'donothing':
            return inputs
        else:
            raise Exception("unknown op_type:%s or invalid statistics :%s" % (
                self.op_type, self.statistics))

    def get_config(self):
        base_config = super(ArithmeticLayer, self).get_config()
        config = {
            'op_type': self.op_type,
            'statistics': self.statistics,
        }
        config.update(base_config)
        return config


class AutoDis(tf.keras.layers.Layer):
    def __init__(self, num_buckets=4, emb_dim=16, keepdim=False, initializer='glorot_uniform', **kwargs):
        super(AutoDis, self).__init__(**kwargs)
        self.num_buckets = num_buckets
        self.emb_dim = emb_dim
        self.keepdim = keepdim
        self.initializer = initializer  # random_normal, truncated_normal, glorot_normal, glorot_uniform, he_normal, he_uniform
        super(AutoDis, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("Unexpected inputs dimensions % d, expect to be 2 dimensions" % (len(input_shape)))
        self.meta_embeddings = self.add_weight(shape=(self.num_buckets, self.emb_dim), initializer=self.initializer,
                                               trainable=True, name='autodis_meta_embeds')
        self.weight_hidden_1 = tf.keras.layers.Dense(max(self.emb_dim * 4, 64), activation='relu')
        self.weight_hidden_2 = tf.keras.layers.Dense(self.num_buckets)
        self.weight_softmax = tf.keras.layers.Softmax()
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


class StrSeqPadLayer(tf.keras.layers.Layer):
    # inputs = '王者荣耀_区服_不限,王者荣耀_游戏段位_最强王者'
    def __init__(self, keys, max_len=10, pad_value=0, sep=',', **kwargs):
        super(StrSeqPadLayer, self).__init__(**kwargs)
        self.keys = keys
        self.max_len = max_len
        self.pad_value = pad_value
        self.sep = sep
        vals = list(range(1, len(keys) + 1))
        keys = tf.constant(keys)
        vals = tf.constant(vals, dtype='int32')
        self.table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, vals), 0)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        inputs_split = tf.squeeze(tf.strings.split(inputs, sep=self.sep), axis=1)
        token_inputs = self.table.lookup(inputs_split)
        pad_inputs = token_inputs.to_tensor(self.pad_value, shape=[None, self.max_len])
        return pad_inputs

    def get_config(self):
        base_config = super(StrSeqPadLayer, self).get_config()
        config = {'keys': self.keys}
        config.update(base_config)
        return config


class IntSeqPadLayer(tf.keras.layers.Layer):
    # inputs = '1,5,43,23'
    def __init__(self, max_len=30, pad_value='0', sep=',', **kwargs):
        super(IntSeqPadLayer, self).__init__(**kwargs)
        self.max_len = max_len
        self.pad_value = pad_value
        self.sep = sep

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        input_full = tf.strings.regex_replace(inputs, '', '0')
        inputs_split = tf.squeeze(tf.strings.split(input_full, sep=self.sep), axis=1)
        pad_inputs = inputs_split.to_tensor(self.pad_value, shape=[None, self.max_len])
        out_tensor = tf.strings.to_number(pad_inputs)
        out_tensor_final = tf.expand_dims(out_tensor, -1)
        return out_tensor_final

    def get_config(self):
        base_config = super(IntSeqPadLayer, self).get_config()
        config = {}
        config.update(base_config)
        return config


class Str2VecLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, vecType='FloatVec', sep=',', **kwargs):
        super(Str2VecLayer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.vecType = vecType
        self.sep = sep

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        inputs_split = tf.strings.split(inputs, sep=self.sep)
        if self.vecType == 'FloatVec':
            float_vec_inputs = tf.strings.to_number(
                inputs_split,
                out_type=tf.float32
            ).to_tensor()
        elif self.vecType == 'IntVec':
            float_vec_inputs = tf.strings.to_number(
                inputs_split,
                out_type=tf.int32
            ).to_tensor()  # shape=()
        else:
            raise Exception('vecType must be FloatVec/IntVec')
        # out_tensor_final = tf.squeeze(float_vec_inputs)
        # return float_vec_inputs
        out = tf.reshape(float_vec_inputs, shape=(-1, 1, self.embedding_dim))
        # out = tf.expand_dims(float_vec_inputs,1)
        return out
        # return tf.squeeze(float_vec_inputs,1)

    def get_config(self):
        base_config = super(Str2VecLayer, self).get_config()
        config = {
            'embedding_dim': self.embedding_dim,
            'vecType': self.vecType,
            'sep': self.sep,
        }
        config.update(base_config)
        return config


class SliceCosSim(tf.keras.layers.Layer):
    def __init__(self, emb_size, **kwargs):
        super(SliceCosSim, self).__init__(**kwargs)
        self.emb_size = emb_size

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, **kwargs):
        assert len(inputs.shape) == 3
        # 把 concat 的 user id 跟 item id 求相似度后的值输入到下一层
        user_emb = tf.slice(inputs, [0, 0, 0], [-1, 1, self.emb_size])
        item_emb = tf.slice(inputs, [0, 0, self.emb_size], [-1, 1, self.emb_size])
        user_emb = nn.l2_normalize(user_emb, axis=-1)
        item_emb = nn.l2_normalize(item_emb, axis=-1)
        return math_ops.reduce_sum(user_emb * item_emb, axis=-1)

    def get_config(self):
        base_config = super(SliceCosSim, self).get_config()
        config = {
            'emb_size': self.emb_size
        }
        config.update(base_config)
        return config


class GetEmbVal(tf.keras.layers.Layer):
    """EMBEDDING"""

    def __init__(self, emb_dict, sep=',', **kwargs):
        self.mask_zero = False
        self.supports_masking = self.mask_zero
        self.emb_dict = emb_dict
        assert self.emb_dict, 'self.emb_dict is empty'
        if len(self.emb_dict) == 2:
            vocab = self.emb_dict.get('vocab')
            emb = self.emb_dict.get('emb')
            assert len(vocab) == len(emb)
        else:
            vocab = list(self.emb_dict.keys())
            emb = [[float(x) for x in emb_str.split(sep)] for emb_str in list(self.emb_dict.values())]

        self.emb = emb
        self.vocab = vocab
        self.dim = len(vocab)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(tf.constant(vocab),
                                                tf.constant(list(range(len(vocab))), dtype=tf.int32)),
            vocab.index('-1'))
        self.embeddings = tf.convert_to_tensor(emb, dtype=tf.float32)
        super(GetEmbVal, self).__init__(**kwargs)

    def build(self, input_shape):
        """bulid"""

        self.built = True

    def call(self, inputs, **kwargs):
        """call"""
        tf.keras.layers.Embedding(self.dim, len(self.vocab), weights=[self.emb], trainable=False)(inputs)
        out = embedding_ops.embedding_lookup(self.embeddings, self.table.lookup(inputs))
        return out

    def compute_mask(self, inputs, mask=None):
        """is mask"""
        if not self.mask_zero:
            return None
        return math_ops.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        """:return shape"""
        output_shape = (input_shape[0], input_shape[1], self.dim)
        return output_shape

    def get_config(self):
        base_config = super(GetEmbVal, self).get_config()
        config = {
            'dim': self.dim,
            # 'vocab': self.vocab,
        }
        config.update(base_config)
        return config

    # def get_config(self):
    #     """get_config"""
    #     base_config = super(GetEmbVal, self).get_config()
    #     base_config['dim'] = self.dim
    #     # base_config['weight'] = self.weight
    #     # base_config['table'] = self.table
    #     # base_config['embeddings'] = self.embeddings
    #     return base_config


if __name__ == '__main__':
    model = GetEmbVal(emb_file='../models/emb_pkl/emb_list_tes.pkl')
    inp = tf.constant([['0'],
                       ['-1'],
                       ['279536513'],
                       ['11111'],
                       ])

    print(model(inp))

    # model = SliceCosSim(emb_size=4)
    # inp = tf.constant([[[1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]],
    #                    [[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]]])
    # print(model(inp))

    # key = ['王者荣耀_区服_不限', '王者荣耀_区服_微信区', '王者荣耀_区服_QQ区', '王者荣耀_模式_排位上分局', '王者荣耀_模式_娱乐局', '王者荣耀_模式_内战', '王者荣耀_人数_5排',
    #        '王者荣耀_人数_3排', '王者荣耀_人数_双排', '王者荣耀_分路偏好_不限', '王者荣耀_分路偏好_中路', '王者荣耀_分路偏好_打野', '王者荣耀_分路偏好_对抗路', '王者荣耀_分路偏好_发育路',
    #        '王者荣耀_分路偏好_游走', '王者荣耀_游戏段位_不限', '王者荣耀_游戏段位_青铜', '王者荣耀_游戏段位_白银', '王者荣耀_游戏段位_黄金', '王者荣耀_游戏段位_铂金',
    #        '王者荣耀_游戏段位_钻石', '王者荣耀_游戏段位_星耀', '王者荣耀_游戏段位_最强王者', '王者荣耀_游戏段位_无双王者', '王者荣耀_游戏段位_荣耀王者', '王者荣耀_游戏段位_传奇王者']
    #
    # model = VocabLayer(keys=key)
    # inp = tf.constant(['王者荣耀_区服_不限', '王者荣耀_游戏段位_最强王者', 'sgegdfghe'])
    # print(model(inp))
    # # res = model.call(inp)
    # # print(res)
    #
    # model = ArithmeticLayer(op_type='normal', input_list=[0, 100])
    # inp = tf.constant([10, 50, 100])
    # print(model(inp))
    # # res = model.call(inp)
    # # print(res)
    #
    # model = AutoDis(num_buckets=2, emb_dim=4)
    # inp = tf.constant([[1, 2, 3, 4]])
    # print(model(inp))
    # # model.build(inp.shape)
    # # res = model.call(inp)
    # # print(res)
    #
    # model = StrSeqPadLayer(keys=key, max_len=5)
    # inp = tf.constant([['王者荣耀_区服_不限,王者荣耀_人数_双排,王者荣耀_游戏段位_最强王者'], ['aghhapwlt'], ['']])
    # print(model(inp))
    # # res = model.call(inp)
    # # print(res)
    #
    # model = IntSeqPadLayer(max_len=5)
    # inp = tf.constant([['0,4,1'], ['']])
    # print(model(inp))
    # # res = model.call(inp)
    # # print(res)
    #
    # model = Str2VecLayer()
    # inp = tf.constant(['0.56464,-0.486645,0.1897546', '0.56464,-0.486645,0.1897546'])
    # print(model(inp))

    # model = GetEmbVal(emb_file='./emb_pkl/emb_tes.pkl')
    # inp = tf.constant([['0'],
    #                    ['-1'],
    #                    ['279536513'],
    #                    ['11111'],
    #                    ])
    #
    # print(model(inp))
