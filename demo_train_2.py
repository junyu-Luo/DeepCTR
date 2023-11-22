# nn_v3_esmm
import os, sys, time, json, gc, pickle
from collections import Counter, defaultdict
from glob import glob
from multiprocessing import cpu_count, Pool, Queue, Process

from tqdm import tqdm
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import models
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score

from deepctr.utils import get_date, date_range
from deepctr.metrics import cal_group_auc
from v2.feature_difinition import DenseFeature, SparseFeature, VarLenFeature, PersonaFeature
from v2.tf_util import VocabLayer, ArithmeticLayer, AutoDis, DNN, FM, sequence_embs_pooling
from v2.tf_util import get_inputs_dict, get_embeddings_dict, get_linear_logit, save_model


# 数据读取
def prepare_trainset(hdfs_path, local_path, dt):
    dt_1 = get_date(-1, dt)
    os.system(f'rm -rf {local_path}/dt={dt_1}')

    if not os.path.exists(f'{local_path}/dt={dt}'):
        _signal = os.system(f'hdfs dfs -get {hdfs_path}/dt={dt} {local_path}')
        if _signal != 0:
            os.system(f'{local_path}/dt={dt}')
            _signal = os.system(f'hdfs dfs -get {hdfs_path}/dt={dt} {local_path}')
            if _signal != 0:
                raise Exception('从hdfs/obs拉取数据集失败')


def get_feature_columns(input_trainset_dir, end_dt):
    with open(f'{input_trainset_dir}/dt={end_dt}/features_info/persona_features.pkl', 'rb') as fr:
        persona_features = pickle.loads(
            fr.read().replace(b'zt_utils.feature_util_v2', b'v2.feature_difinition'))
    feature_columns = [pf.tf_input for pf in persona_features]
    return persona_features, feature_columns


def get_tfrecord_define(feature_columns):
    description = {
        'user_id': tf.io.FixedLenFeature((), 'int64'),
        'has_enter_room': tf.io.FixedLenFeature((), 'int64'),
        'duration': tf.io.FixedLenFeature((), 'int64', 0)
    }
    for fc in feature_columns:
        description[fc.name] = tf.io.FixedLenFeature(fc.shape, fc.dtype)
    return description


def parseTFRecord(decoded_example, description=None, only_xy=False):  # TODO: cast int64 as int32
    # 定义解析的字典
    if description is None:
        print('WARN: 没有提供description')
        description = {
            'id': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'IntegerCol': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'LongCol': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'FloatCol': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
            'DoubleCol': tf.io.FixedLenFeature([], tf.float32, default_value=-214.0),
            'VectorCol': tf.io.FixedLenFeature([2], tf.float32, default_value=tf.zeros(2, dtype='float32')),
            'StringCol': tf.io.FixedLenFeature([], tf.string, default_value='fillna'),
        }

    # 调用接口解析一行样本，得到一个字典，其中每个key是对应feature的名字，value是相应的feature解析值
    new_desc = dict()
    for k, v in description.items():
        if v.dtype in ('int32', tf.int32):
            new_desc[k] = tf.io.FixedLenFeature(v.shape, 'int64', v.default_value)
        elif v.dtype in ('float64', tf.float64, 'double'):
            new_desc[k] = tf.io.FixedLenFeature(v.shape, 'float32', v.default_value)
        else:
            new_desc[k] = v
    features = tf.io.parse_example(decoded_example, features=new_desc)
    for k, v in description.items():
        if v.dtype in ('int32', tf.int32):
            features[k] = tf.cast(features[k], 'int32')
        elif v.dtype in ('float64', tf.float64, 'double'):
            new_desc[k] = tf.cast(features[k], 'float64')

    # 根据任务需求对解析出来的数据做调整
    has_enter = tf.cast(features.pop('has_enter_room'), 'float32')
    has_convert = tf.cast(features.pop('duration') >= 180, 'float32')
    X, y, others = features, (has_enter, has_convert), {'user_id': features.pop('user_id')}
    if only_xy:
        return X, y
    return X, y, others


# 模型构建
def ESMM(feature_columns, dnn_hidden_units=(128, 128), dnn_activation='relu', emb_reg=1e-5, dnn_reg=1e-4,
         optimizer='adam', loss='binary_crossentropy', metrics=['AUC', 'binary_crossentropy'], run_eagerly=None,
         model_name='ESMM'):
    # # checks
    assert all(True if isinstance(fc, DenseFeature) or isinstance(fc, SparseFeature) or isinstance(fc,
                                                                                                   VarLenFeature) else False
               for fc in feature_columns)

    # # inputs and embeddings
    inputs_dict = get_inputs_dict(feature_columns)
    normal_inputs_dict = dict()  # 用于存储归一化后的DenseFeature和lookup后的SparseFeature和VarLenFeature
    embeddings_dict = get_embeddings_dict(inputs_dict, feature_columns, middle_outputs=normal_inputs_dict)

    dnn_input = layers.Concatenate(axis=-1, name='dnn_concat_embs')(
        [layers.Flatten(name=f'flatten_{fc.name}')(embeddings_dict[fc.name]) for fc in feature_columns])
    ctr_hidden = DNN(dnn_hidden_units, activation=dnn_activation, l2_reg=dnn_reg, name='ctr_dnn_hidden')(dnn_input)
    pCTR = layers.Dense(1, use_bias=False, activation='sigmoid', name='pCTR')(ctr_hidden)
    cvr_hidden = DNN(dnn_hidden_units, activation=dnn_activation, l2_reg=dnn_reg, name='cvr_dnn_hidden')(dnn_input)
    pCVR = layers.Dense(1, use_bias=False, activation='sigmoid', name='pCVR')(cvr_hidden)

    pCTCVR = layers.Multiply(name='pCTCVR')([pCTR, pCVR])  # pCTCVR = pCTR * pCVR
    esmm = models.Model(inputs=inputs_dict, outputs=[pCTR, pCTCVR], name=model_name)
    esmm.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=run_eagerly)
    return esmm


def MMoE(input_embs, num_tasks, num_experts=8, dnn_hidden_units=(128, 128), reg=0, gate_dropout=0.5):
    ''' input_embs: [None, d]
    '''
    input_embs = tf.expand_dims(input_embs, axis=1)

    experts_layers = [DNN(dnn_hidden_units, l2_reg=reg, output_activation='no', name=f'expert_{i}') for i in
                      range(num_experts)]
    gates_layers = [DNN([dnn_hidden_units[0], num_experts], output_activation='softmax', l2_reg=reg, name=f'gate_{i}')
                    for i in range(num_tasks)]
    dropout_layers = [layers.Dropout(rate=gate_dropout, name=f'gate_dropout_{i}') for i in range(num_tasks)]

    experts_outputs = [expert_layer(input_embs) for expert_layer in
                       experts_layers]  # [(None, 1, hidden_dim), ...] * num_experts
    experts_stack = layers.Concatenate(axis=1, name='experts_stack')(experts_outputs)  # (None, num_experts, hidden_dim)
    gates_outputs = [dropout_layer(gate_layer(input_embs)) for dropout_layer, gate_layer in
                     zip(dropout_layers, gates_layers)]  # [(None, 1, num_experts), ...] * num_tasks

    towers_inputs = list()
    for i in range(num_tasks):
        towers_inputs.append(
            tf.squeeze(tf.matmul(gates_outputs[i], experts_stack), axis=1))  # [(None, hidden_dim), ...] * num_tasks
    return towers_inputs


def ESMMoE(feature_columns, dnn_hidden_units=(128, 128), dnn_activation='relu', emb_reg=1e-5, dnn_reg=1e-4,
           optimizer='adam', loss='binary_crossentropy', metrics=['AUC', 'binary_crossentropy'], run_eagerly=None,
           num_experts=8, expert_dnn=(128, 128), model_name='ESMMoE'):
    # # checks
    assert all(True if isinstance(fc, DenseFeature) or isinstance(fc, SparseFeature) or isinstance(fc,
                                                                                                   VarLenFeature) else False
               for fc in feature_columns)

    # # inputs and embeddings
    inputs_dict = get_inputs_dict(feature_columns)
    normal_inputs_dict = dict()  # 用于存储归一化后的DenseFeature和lookup后的SparseFeature和VarLenFeature
    embeddings_dict = get_embeddings_dict(inputs_dict, feature_columns, middle_outputs=normal_inputs_dict)

    dnn_input = layers.Concatenate(axis=-1, name='dnn_concat_embs')(
        [layers.Flatten(name=f'flatten_{fc.name}')(embeddings_dict[fc.name]) for fc in feature_columns])
    tower_inputs = MMoE(dnn_input, 2, num_experts=num_experts, dnn_hidden_units=expert_dnn, reg=dnn_reg)

    ctr_hidden = DNN(dnn_hidden_units, activation=dnn_activation, l2_reg=dnn_reg, name='ctr_dnn_hidden')(
        tower_inputs[0])
    pCTR = layers.Dense(1, use_bias=False, activation='sigmoid', name='pCTR')(ctr_hidden)
    cvr_hidden = DNN(dnn_hidden_units, activation=dnn_activation, l2_reg=dnn_reg, name='cvr_dnn_hidden')(
        tower_inputs[1])
    pCVR = layers.Dense(1, use_bias=False, activation='sigmoid', name='pCVR')(cvr_hidden)

    pCTCVR = layers.Multiply(name='pCTCVR')([pCTR, pCVR])  # pCTCVR = pCTR * pCVR
    mmoe_esmm = models.Model(inputs=inputs_dict, outputs=[pCTR, pCTCVR], name=model_name)
    mmoe_esmm.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=run_eagerly)
    return mmoe_esmm


def get_y_and_uid(val_ds):
    def vstack(list_of_array):
        if len(list_of_array[0].shape) == 1:
            return np.concatenate(list_of_array)
        else:
            return np.vstack(list_of_array)

    uid_list, y_list, y2_list = [], [], []
    for data in val_ds:
        if isinstance(data[1], tuple):
            y_list.append(data[1][0].numpy())
            y2_list.append(data[1][1].numpy())
        else:
            y_list.append(data[1].numpy())
        uid_list.append(data[-1]['user_id'].numpy())

    if len(y2_list) > 0:
        return (vstack(y_list), vstack(y2_list)), vstack(uid_list)
    return vstack(y_list), vstack(uid_list)


# 读数据

dt, env, model_id, task_id, task_name = '20221205', 'test', 'music_list_nn_v3_esmm', 1064, 'music_list_nn_v3_esmm'
end_dt = dt
# dt, env, model_id, task_id, task_name = sys.argv[1:1 + 5]

start_dt = get_date(-7, end_dt)
start_1 = get_date(-1, dt)

input_trainset_dir = './v2/data/tfrecord_v1'
input_pretrain_expo_song_dir = f'./v2/data/expo_song_seq_i2v_{start_1}.KeyedVectors'
output_model_dir = './v2/model/nn_v3_esmm'

if __name__ == '__main__':
    # 读取数据
    _persona_features, feature_columns = get_feature_columns(input_trainset_dir, end_dt)
    for fc in feature_columns:
        if fc.name == 'i_expo_song_name':  # 采用预训练embedding
            fc.pretrained_path = input_pretrain_expo_song_dir
    description = get_tfrecord_define(feature_columns)

    train_ds = tf.data.Dataset.list_files(f'{input_trainset_dir}/dt={end_dt}/train/part*', shuffle=False).interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        cycle_length=6, block_length=1024, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    ).batch(1024).map(lambda x: parseTFRecord(x, description, only_xy=True),
                      num_parallel_calls=tf.data.AUTOTUNE).prefetch(100)

    val_ds = tf.data.Dataset.list_files(f'{input_trainset_dir}/dt={end_dt}/validation/part*', shuffle=False).interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        cycle_length=6, block_length=2048, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    ).batch(2048).map(lambda x: parseTFRecord(x, description), num_parallel_calls=tf.data.AUTOTUNE).prefetch(100)

    # esmm: 数据喂入模型训练
    K.clear_session();
    gc.collect()
    esmm = ESMM(
        feature_columns, dnn_hidden_units=(128, 128), emb_reg=1e-5, dnn_reg=1e-5,
        optimizer='Adamax', loss='binary_crossentropy', metrics=['binary_crossentropy', 'AUC'],
    )
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_pCTCVR_auc_1', patience=3, mode='max',
                                                  restore_best_weights=True), ]
    history = esmm.fit(train_ds, epochs=1, validation_data=val_ds.map(lambda *x: (x[0], x[1])), callbacks=callbacks,
                       verbose=1)

    # 计算最终auc和gauc
    y_true, uid_list = get_y_and_uid(val_ds)
    y_pred = esmm.predict(val_ds)
    metrics = dict()
    for i, task in enumerate(['ctr', 'ctcvr']):
        gauc = round(cal_group_auc(y_true[i], y_pred[i], uid_list), 6);
        print(f'{task}_gauc =', gauc)
        auc = round(roc_auc_score(y_true[i], y_pred[i]), 6);
        print(f'{task}_auc =', auc)
        metrics.update({f'{task}_test_auc': auc, f'{task}_test_gauc': gauc})

    # 保存模型
    inference_model = tf.keras.Model(inputs=esmm.inputs, outputs=[esmm.outputs[-1]])
    if True:
        X_val = next(iter(val_ds))[0]
        saved_path = save_model(inference_model, output_model_dir, tfx_warmup=True, X_4_warmup=X_val)
        if True:
            base_path = f'{input_trainset_dir}/dt={end_dt}/features_info'
            os.system('cp %s %s' % (os.path.join(base_path, 'feature.json'), os.path.join(saved_path, 'feature.json')))
            os.system(
                'cp %s %s' % (os.path.join(base_path, 'token_map.json'), os.path.join(saved_path, 'token_map.json')))
