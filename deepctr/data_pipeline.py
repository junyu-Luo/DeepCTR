from .utils import read_json, dir_file_name, flatList
from .feature_column import DenseFeat, SparseFeat, VarLenSparseFeat, EmbFeat
import time
import tensorflow as tf
import os
from tqdm import tqdm
import random
import pandas as pd
import numpy as np


def get_train_features(config_map, choose_label, choose_feat, ensure_uid_in_field=True):
    if isinstance(config_map, dict):
        conf_js = config_map
    else:
        conf_js = read_json(config_map)
        assert conf_js

    conf_label_cols = [x.get('out') for x in conf_js.get('label')]
    conf_features_cols = [x.get('out') for x in conf_js.get('features')]

    assert len(set(choose_label)) == len(choose_label)
    assert len(set(choose_feat)) == len(choose_feat)
    assert len(set(conf_label_cols)) == len(conf_label_cols)
    assert len(set(conf_features_cols)) == len(conf_features_cols)

    label = []
    for label_name in choose_label:
        try:
            label_index = conf_label_cols.index(label_name)
        except:
            raise Exception('label_name {} not in conf_js'.format(label_name))
        label.append(conf_js.get('label')[label_index])

    features = []
    for feat_name in choose_feat:
        try:
            feat_index = conf_features_cols.index(feat_name)
        except:
            raise Exception('feat_name {} not in conf_js'.format(feat_name))
        features.append(conf_js.get('features')[feat_index])

    if ensure_uid_in_field:
        field_col = [x.get('out') for x in conf_js['field']]
        # user_id
        if "user_id" not in field_col and "user_id" not in choose_feat:
            conf_js['field'].append(
                {
                    "out": "user_id",
                    "expr": "str(user_id)",
                    "feature_column": {
                        "dtype": "string",
                        "featureType": "field",
                        "comment": "主键user_id"
                    }
                },
            )

    conf_js['label'] = label
    conf_js['features'] = features
    return conf_js


class TFrecordBuilder:
    def __init__(self, config_map, add_field=True):
        if isinstance(config_map, dict):
            conf = config_map
        else:
            conf = read_json(config_map)
            assert conf

        self.labels = conf.get('label')
        self.conf_features = conf.get('features')

        if add_field:
            self.field = conf.get('field')
        else:
            self.field = []

        feature_description = {}
        self.int64toint32 = []
        for js in self.labels + self.conf_features + self.field:
            out = js.get('out')
            dtype = js.get('feature_column').get('dtype')

            # default_value=None，在数据处理过程中，已经不存在None了
            if dtype == 'int32':
                # FixedLenFeature 不支持int32,需要后续强转(ef约定只有int32)
                feature_description[out] = tf.io.FixedLenFeature([], tf.int64)
                self.int64toint32.append(out)
            elif dtype == 'int64':
                feature_description[out] = tf.io.FixedLenFeature([], tf.int64)
            elif dtype == 'float32':
                feature_description[out] = tf.io.FixedLenFeature([], tf.float32)
            elif dtype == 'string':
                feature_description[out] = tf.io.FixedLenFeature([], tf.string)
            elif dtype == 'vector':
                feature_description[out] = tf.io.VarLenFeature(tf.float32)
            else:
                raise Exception('{} name dtype is {},not in [int32/float32/string/vector]'.format(out, dtype))
        self.feature_description = feature_description

    def _parse_function(self, example_proto):
        example = tf.io.parse_example(example_proto, self.feature_description)
        if self.int64toint32:
            for name in self.int64toint32:
                example[name] = tf.cast(example[name], tf.int32)
        if len(self.labels) == 1:
            label_name = self.labels[0].get('out')
            label = example[label_name]
            example.pop(label_name)
            feature = example
            return feature, label
        else:
            labels = {}
            for label_js in self.labels:
                label_name = label_js.get('out')
                labels[label_name] = example[label_name]
                example.pop(label_name)
            feature = example
            return feature, labels

    def _get_raw_dataset(self, tfrecord_files, buffer_size=1024, num_parallel_reads=2):
        if isinstance(tfrecord_files, str):
            all_files = dir_file_name(tfrecord_files)
            assert len(all_files) > 0, 'dataset not exist,check tfrecord_files:{}'.format(tfrecord_files)
        elif isinstance(tfrecord_files, list):
            all_files = []
            for filename in tfrecord_files:
                if '.tfrecord.gz' in filename:
                    all_files.append(filename)
                else:
                    all_files.extend(dir_file_name(filename))
            # all_files = list(flatList([dir_file_name(x) for x in tfrecord_files]))
            assert len(all_files) > 0, 'dataset not exist,check tfrecord_files:{}'.format(tfrecord_files)
        else:
            raise Exception('tfrecord_files must str/list')

        return tf.data.TFRecordDataset(all_files,
                                       compression_type='GZIP',
                                       buffer_size=buffer_size,
                                       num_parallel_reads=num_parallel_reads)

    def generate(self, tfrecord_files, batch_size=1024, buffer_size=None):
        dataset = self._get_raw_dataset(tfrecord_files)
        if buffer_size:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size) \
            .map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def get_pred_y_uid(self, dataset, uid_field='user_id', return_dict=False):
        if len(self.labels) == 1:
            y_list = []
            uid_list = []
            for X, y in dataset:
                y_list.extend(y.numpy().tolist())
                uid_list.extend(X[uid_field].numpy().tolist())
            if return_dict:
                return {
                    self.labels[0]: y_list,
                    uid_field: uid_list
                }
            else:
                return y_list, uid_list
        else:
            y_dict = {}
            y_keys_list = []
            for X, y in dataset:
                y_keys_list = list(y.keys())
                for y_key in y_keys_list:
                    y_dict[y_key] = []
                break
            uid_list = []
            for X, y in dataset:
                for k in y_keys_list:
                    y_dict[k].extend(y[k].numpy().tolist())
                uid_list.extend(X[uid_field].numpy().tolist())
            if return_dict:
                y_dict[uid_field] = uid_list
                return y_dict
            else:
                return y_dict, uid_list


class ParquetBuilder:
    def __init__(self, config_map, add_field=True):
        self.seed = 42
        random.seed(self.seed)

        if isinstance(config_map, dict):
            conf = config_map
        else:
            conf = read_json(config_map)
            assert conf

        if add_field:
            self.features = conf.get('label') + conf.get('features') + conf.get('field')
        else:
            self.features = conf.get('label') + conf.get('features')

        self.columns = [x.get('out') for x in self.features]
        self.labels = conf.get('label')

    def generate_func(self, parquet_files):
        if isinstance(parquet_files, str):
            file_list = dir_file_name(parquet_files)
        elif isinstance(parquet_files, list):
            file_list = [dir_file_name(x) for x in parquet_files]
        else:
            raise Exception('parquet_files must str/list')
        random.shuffle(file_list)
        for file_name in file_list:
            df = pd.read_parquet(file_name, columns=self.columns)
            df = df.sample(frac=1, random_state=self.seed)  # frac 是抽样参数
            X = {js.get('out'): df[js.get('out')].values.reshape([-1, 1]) for js in self.features}
            if len(self.label) == 1:
                label_name = self.label[0].get('out')
                y = df[label_name].values
                yield X, y
            else:
                labels = {}
                for label_js in self.label:
                    label_name = label_js.get('out')
                    labels[label_name] = df[label_name].values
                yield X, labels

    def generate(self, parquet_files, task, batch_size=1024, buffer_size=None, cache_name="auto",
                 cache_dir='obs://pvc-obs-hw-bj-zt-rcmd-data/temp/parquet_cache/'):
        dataset = tf.data.Dataset.from_generator(
            lambda: self.generate_func(parquet_files),
            output_signature=(
                # features
                {js.get("out"): tf.TensorSpec(shape=(None, 1), dtype=js.get('feature_column').get("dtype")) for js in
                 self.features},
                # label
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        if buffer_size:
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.unbatch().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        if cache_name is not None:
            if cache_name == "auto":
                cache_name = '{}/{}_{}_{}'.format(cache_dir, task, batch_size,
                                                  time.strftime("%Y%m%d", time.localtime()))
            else:
                cache_name = '{}/{}_{}_{}'.format(cache_dir, task, batch_size, cache_name)

            dataset = dataset.cache(cache_name)

            if not os.path.exists(cache_name + '.data-00000-of-00001'):
                print('caching')
                for _ in tqdm(dataset): pass

        return dataset

    def get_pred_y_uid(self, dataset, uid_field='user_id', return_dict=False):
        if len(self.labels) == 1:
            y_list = []
            uid_list = []
            for X, y in dataset:
                y_list.extend(y.numpy().tolist())
                uid_list.extend(X[uid_field].numpy().tolist())
            if return_dict:
                return {
                    self.labels[0]: y_list,
                    uid_field: uid_list
                }
            else:
                return y_list, uid_list
        else:
            y_dict = {}
            y_keys_list = []
            for X, y in dataset:
                y_keys_list = list(y.keys())
                for y_key in y_keys_list:
                    y_dict[y_key] = []
                break
            uid_list = []
            for X, y in dataset:
                for k in y_keys_list:
                    y_dict[k].extend(y[k].numpy().tolist())
                uid_list.extend(X[uid_field].numpy().tolist())
            if return_dict:
                y_dict[uid_field] = uid_list
                return y_dict
            else:
                return y_dict, uid_list


class FeatureColumn:
    def __init__(self, config_map, default_embedding_dim=8):

        if isinstance(config_map, dict):
            conf = config_map
        else:
            conf = read_json(config_map)
            assert conf

        self.default_embedding_dim = default_embedding_dim

        self.conf = conf

        self.label_names = [x.get('out') for x in conf.get('label')]
        self.feature_names = [x.get('out') for x in conf.get('features')]
        self.field_names = [x.get('out') for x in conf.get('field')]
        self.column_names = self.label_names + self.feature_names + self.field_names

        self.DenseFeatures = []
        self.SparseFeatures = []
        self.VarLenSparseFeatures = []
        self.get_input_feature()
        self.features = self.DenseFeatures + self.SparseFeatures + self.VarLenSparseFeatures

    def get_input_feature(self):
        for js in self.conf.get('features'):
            name = js.get("out")

            fc = js.get('feature_column')
            featureType = fc.get('featureType')

            if featureType.lower() == 'dense':
                dimension = fc.get("dimension", 1)
                dtype = fc.get("dtype")
                operation = fc.get("operation", "normal")
                statistics = fc.get("statistics")

                assert dtype is not None
                assert statistics is not None

                self.DenseFeatures.append(
                    DenseFeat(name=name, dimension=dimension, dtype=dtype, operation=operation, statistics=statistics)
                )

            elif featureType.lower() == 'sparse':
                embedding_dim = fc.get("embedding_dim", self.default_embedding_dim)
                embedding_name = fc.get("embedding_name", name)
                vocab = fc.get("vocab")
                dtype = fc.get("dtype")

                assert embedding_name is not None
                assert vocab is not None
                assert dtype is not None
                self.SparseFeatures.append(
                    SparseFeat(name=name, embedding_dim=embedding_dim, embedding_name=embedding_name, vocab=vocab,
                               dtype=dtype)
                )

            elif featureType.lower() == 'varlensparse':
                embedding_dim = fc.get("embedding_dim", self.default_embedding_dim)
                embedding_name = fc.get("embedding_name", name)  # 为None的时候就是name
                vocab = fc.get("vocab")
                dtype = fc.get("dtype")
                maxlen = fc.get("maxlen")

                assert vocab is not None
                assert dtype is not None
                assert maxlen is not None

                length_name = fc.get("length_name", None)
                combiner = fc.get('combiner', "mean")
                weight_name = fc.get("weight_name", None)
                self.VarLenSparseFeatures.append(
                    VarLenSparseFeat(
                        SparseFeat(
                            name=name, embedding_dim=embedding_dim,
                            vocab=vocab,
                            dtype=dtype,
                            embedding_name=embedding_name
                        ),
                        maxlen=maxlen, combiner=combiner, length_name=length_name, weight_name=weight_name
                    ))

            # elif 需要再添加

            else:
                raise Exception('featureType support dense/sparse/varlensparse,')

            # emb 分两种，一种vocab很少直接，一种很多需要传文件加载不然速度太慢
            # elif featureType.lower() == 'emb':
            #
            #     weights = np.load(emb_file, allow_pickle=True)
            #     vocab
            #
            #     self.vocab_file = vocab_file
            #     if emb_file:
            #         print('loading pickle file')
            #         self.vocab = vocab_file
            #         self.weights = np.load(emb_file, allow_pickle=True)
            #         self.embed_size = len(self.weights[0])
            #         self.vocabulary_size = len(self.weights)  # 一般最后一个是平均向量
            #         print('load pickle file done')
            #
            #     name = js.get("out")
            #     dtype = js.get("dtype")
            #     embedding_name = js.get("embedding_name")
            #     trainable = js.get("trainable", False)
            #     res['EmbFeatures'].append(
            #         EmbFeat(name=name,
            #                 vocab=self.vocab,
            #                 weights=self.weights,
            #                 trainable=trainable,
            #                 embed_size=self.embed_size,
            #                 dtype=dtype,
            #                 vocabulary_size=self.vocabulary_size,
            #                 embedding_name=embedding_name)
            #     )


if __name__ == '__main__':
    conf_file = 'feature.json'
    files_path = './Dataset'
    builder = TFrecordBuilder(conf_file)

    dataset = builder.generate(files_path)
    for x, y in dataset:
        print(x)
        print(y)
        break
